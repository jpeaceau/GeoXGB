#include "geoxgb/geoxgb_base.h"
#include "geoxgb/noise.h"
#include "hvrt/types.h"
#include "hvrt/hvrt.h"
#include "hvrt/target.h"
#include "hvrt/reduce.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <unordered_map>

namespace geoxgb {

// ── Constructor ────────────────────────────────────────────────────────────────

GeoXGBBase::GeoXGBBase(GeoXGBConfig cfg) : cfg_(std::move(cfg)) {}

// ── Geometry accessors ────────────────────────────────────────────────────────

Eigen::MatrixXd GeoXGBBase::to_z(const Eigen::MatrixXd& X_new) const {
    if (!last_hvrt_ || !last_hvrt_->fitted())
        throw std::runtime_error("No geometry state: re-fit the model.");
    return last_hvrt_->to_z(X_new);
}

Eigen::VectorXi GeoXGBBase::apply(const Eigen::MatrixXd& X_new) const {
    if (!last_hvrt_ || !last_hvrt_->fitted())
        throw std::runtime_error("No geometry state: re-fit the model.");
    return last_hvrt_->apply(X_new);
}

// ── feature_importances ───────────────────────────────────────────────────────
// Aggregate impurity-based importance across all GBT weak learners.

std::vector<double> GeoXGBBase::feature_importances() const {
    if (trees_.empty()) return {};
    const int d = static_cast<int>(cont_cols_full_.size());
    std::vector<double> total(d, 0.0);
    for (const auto& wl : trees_) {
        const auto& fi = wl.tree.feature_importances();
        const int nd = static_cast<int>(std::min(fi.size(), total.size()));
        for (int i = 0; i < nd; ++i)
            total[i] += fi[i];
    }
    double s = 0.0;
    for (double v : total) s += v;
    if (s > 1e-12) for (double& v : total) v /= s;
    return total;
}

// ── Default targets_from_gradients (regression: y = preds + grads) ───────────

Eigen::VectorXd GeoXGBBase::targets_from_gradients(
        const Eigen::VectorXd& grads,
        const Eigen::VectorXd& preds) const {
    return preds + grads;
}

// ── k-NN y-assignment for synthetic samples ───────────────────────────────────
// Global inverse-distance weighted (k=3) in HVRT z-space.

Eigen::VectorXd GeoXGBBase::knn_assign_y(
        const Eigen::MatrixXd& X_syn_z,
        const Eigen::MatrixXd& X_red_z,
        const Eigen::VectorXd& y_red) const
{
    const int n_syn = static_cast<int>(X_syn_z.rows());
    const int n_red = static_cast<int>(X_red_z.rows());
    const int k     = std::min(3, n_red);

    Eigen::VectorXd y_syn(n_syn);

    // ── Large-n_red guard ─────────────────────────────────────────────────────
    // The n_syn × n_red distance matrix grows as O(n²) at scale:
    // n_syn=10k, n_red=70k (n=100k, er=0.1, rr=0.7) → 5.6 GB.
    // When n_red exceeds KNN_RED_CAP, subsample via Fisher-Yates and recurse.
    // Recursion is safe: the inner call has n_red == KNN_RED_CAP ≤ cap.
    // The HVRT-reduced set is geometrically dense, so approximate NN quality
    // is preserved: at cap=5k from 70k, partition density ≈ 5k/70k × n_part
    // which still yields good IDW estimates within each geometric region.
    static constexpr int KNN_RED_CAP = 5000;
    if (n_red > KNN_RED_CAP) {
        uint64_t lcg = static_cast<uint64_t>(n_red) | 1u;
        auto lcg_next = [&](int range) -> int {
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<int>((lcg >> 33) % static_cast<uint64_t>(range));
        };
        std::vector<int> sub(n_red);
        std::iota(sub.begin(), sub.end(), 0);
        for (int i = 0; i < KNN_RED_CAP; ++i) {
            int j = i + lcg_next(n_red - i);
            std::swap(sub[i], sub[j]);
        }
        Eigen::MatrixXd X_red_sub(KNN_RED_CAP, X_red_z.cols());
        Eigen::VectorXd y_red_sub(KNN_RED_CAP);
        for (int i = 0; i < KNN_RED_CAP; ++i) {
            X_red_sub.row(i) = X_red_z.row(sub[i]);
            y_red_sub[i]     = y_red[sub[i]];
        }
        return knn_assign_y(X_syn_z, X_red_sub, y_red_sub);
    }

    // ── BLAS pairwise squared-distance matrix ────────────────────────────────
    // D[i,j] = ||X_syn_z[i] - X_red_z[j]||²
    //        = ||X_syn_z[i]||² - 2·X_syn_z[i]·X_red_z[j]ᵀ + ||X_red_z[j]||²
    //
    // Single GEMM (n_syn×d)×(d×n_red) replaces the O(n_syn·n_red·d) scalar
    // double loop, and eliminates n_syn per-iteration heap allocations.
    const Eigen::VectorXd norm_syn = X_syn_z.rowwise().squaredNorm();  // (n_syn,)
    const Eigen::VectorXd norm_red = X_red_z.rowwise().squaredNorm();  // (n_red,)

    // RowMajor so D.row(i) is contiguous — critical for cache-friendly partial sort.
    // Persistent buffer: resize is a no-op if dimensions haven't changed (saves ~14 MB
    // of VirtualAlloc/free per call on Windows when n_syn=500, n_red=3500).
    knn_D_.resize(n_syn, n_red);
    auto& D = knn_D_;
    D.noalias() = X_syn_z * X_red_z.transpose();
    D *= -2.0;
    D.colwise() += norm_syn;            // D[i,j] += ||X_syn_z[i]||²
    D.rowwise() += norm_red.transpose();// D[i,j] += ||X_red_z[j]||²
    D = D.cwiseMax(0.0);                // clamp fp rounding negatives

    // ── IDW via top-k linear scan on contiguous D rows ───────────────────────
    // D is RowMajor → D.row(i).data() is contiguous, enabling sequential
    // streaming reads and compiler auto-vectorisation of the inner loop.
    // Replaces iota+partial_sort: saves ~7 MB of index writes and O(n_red·k)
    // heap comparisons per call.  k=3 so insertion-sort is 2 compares at most.
    constexpr int kmax = 3;
    constexpr double BIG = 1e300;
    for (int i = 0; i < n_syn; ++i) {
        const double* row = D.row(i).data();  // contiguous pointer into RowMajor D
        double best_d[kmax] = {BIG, BIG, BIG};
        int    best_j[kmax] = {-1,  -1,  -1};

        for (int j = 0; j < n_red; ++j) {
            const double dj = row[j];
            if (dj < best_d[k - 1]) {
                best_d[k - 1] = dj;
                best_j[k - 1] = j;
                // Insertion-sort the k=3 slots (at most 2 swaps).
                for (int s = k - 2; s >= 0; --s) {
                    if (best_d[s] <= best_d[s + 1]) break;
                    double td = best_d[s]; best_d[s] = best_d[s+1]; best_d[s+1] = td;
                    int    ti = best_j[s]; best_j[s] = best_j[s+1]; best_j[s+1] = ti;
                }
            }
        }

        double w_sum = 0.0, y_sum = 0.0;
        for (int s = 0; s < k; ++s) {
            if (best_j[s] < 0) continue;
            const double w = 1.0 / (std::sqrt(best_d[s]) + 1e-10);
            w_sum += w;
            y_sum += w * y_red[best_j[s]];
        }
        y_syn[i] = y_sum / w_sum;
    }
    return y_syn;
}

// ── Per-partition z-space sub-split residual correction ───────────────────────
// For each HVRT partition p, finds the best variance-reducing binary split of
// y_signal (current residuals) in z-space.  Applies the shrunk per-child mean
// correction to synthetic y targets:
//   delta_shrunk = mean(resid[child]) * n_child / (n_child + lambda)
// Only called during refit (y_signal = gradients), never on initial fit.

static void apply_sub_split_correction(
        const Eigen::MatrixXd&  X_z,       // n_train × d  (training z-coords)
        const Eigen::VectorXd&  y_signal,  // n_train      (current residuals)
        const Eigen::VectorXi&  tr_pids,   // n_train      (training partition IDs)
        const Eigen::MatrixXd&  X_syn_z,   // n_syn × d    (synthetic z-coords)
        const Eigen::VectorXi&  syn_pids,  // n_syn        (synthetic partition IDs)
        Eigen::VectorXd&        y_syn,     // [in/out]     (synthetic targets)
        double lambda,
        int min_leaf = 3)
{
    const int n_train = static_cast<int>(X_z.rows());
    const int d       = static_cast<int>(X_z.cols());
    const int n_syn   = static_cast<int>(y_syn.size());
    if (n_syn == 0 || lambda <= 0.0) return;

    // Group training indices by partition
    std::unordered_map<int, std::vector<int>> part_idx;
    part_idx.reserve(64);
    for (int i = 0; i < n_train; ++i)
        part_idx[tr_pids[i]].push_back(i);

    // For each partition: find best variance-reducing z-space split of residuals
    struct SplitInfo { int feat = -1; double thresh = 0.0, dl = 0.0, dr = 0.0; };
    std::unordered_map<int, SplitInfo> splits;
    splits.reserve(static_cast<int>(part_idx.size()));

    for (auto& [pid, idx] : part_idx) {
        const int n_p = static_cast<int>(idx.size());
        if (n_p < 2 * min_leaf) continue;

        SplitInfo best;
        double best_score = -1.0;

        for (int k = 0; k < d; ++k) {
            // Gather (z_k, resid) pairs and sort by z_k
            std::vector<std::pair<double,double>> kv(n_p);
            for (int i = 0; i < n_p; ++i)
                kv[i] = { X_z(idx[i], k), y_signal[idx[i]] };
            std::sort(kv.begin(), kv.end());

            // Prefix sum of residuals
            double total = 0.0;
            for (auto& pv : kv) total += pv.second;

            // Sweep: sum_l accumulates left, sum_r = total - sum_l
            double sum_l = 0.0;
            for (int s = 0; s < min_leaf - 1; ++s) sum_l += kv[s].second;

            for (int sp = min_leaf - 1; sp < n_p - min_leaf; ++sp) {
                sum_l += kv[sp].second;
                const int    n_l    = sp + 1;
                const int    n_r    = n_p - n_l;
                const double mean_l = sum_l / n_l;
                const double mean_r = (total - sum_l) / n_r;
                // Variance-reduction proxy: maximise between-group SS of means
                const double score = (double)n_l * mean_l * mean_l
                                   + (double)n_r * mean_r * mean_r;
                if (score > best_score) {
                    best_score   = score;
                    best.feat    = k;
                    best.thresh  = kv[sp].first;
                    best.dl      = mean_l * n_l / (n_l + lambda);
                    best.dr      = mean_r * n_r / (n_r + lambda);
                }
            }
        }
        if (best.feat >= 0) splits[pid] = best;
    }

    // Apply correction to each synthetic sample
    for (int i = 0; i < n_syn; ++i) {
        auto it = splits.find(syn_pids[i]);
        if (it == splits.end()) continue;
        const SplitInfo& si = it->second;
        y_syn[i] += (X_syn_z(i, si.feat) <= si.thresh) ? si.dl : si.dr;
    }
}

// ── HVRT resampling ───────────────────────────────────────────────────────────

GeoXGBBase::ResampleResult GeoXGBBase::do_resample(
        const Eigen::MatrixXd& X_full,
        const Eigen::VectorXd& y_signal,
        int seed_offset,
        std::shared_ptr<hvrt::HVRT>& hvrt_out)
{
    const int n = static_cast<int>(X_full.rows());

    // Track whether this is an initial fit (y_signal = raw targets) vs a refit
    // (y_signal = gradients/residuals).  The sub-split correction is meaningful
    // only during refit, when y_signal is centred near zero.
    const bool is_initial_fit = !(hvrt_out && hvrt_out->fitted());

    // Build HVRT config
    hvrt::HVRTConfig hcfg;
    hcfg.y_weight     = static_cast<float>(cfg_.y_weight);
    hcfg.n_bins       = 16;  // HVRT partition tree: 16 bins sufficient (halves BFS scan vs 32)
    hcfg.random_state = cfg_.random_state + seed_offset;
    hcfg.auto_tune    = true;

    if (cfg_.hvrt_min_samples_leaf > 0)
        hcfg.min_samples_leaf = cfg_.hvrt_min_samples_leaf;
    if (cfg_.hvrt_n_partitions > 0) {
        hcfg.n_partitions = cfg_.hvrt_n_partitions;
        hcfg.auto_tune    = false;
    }

    // Forward S1 coupling flag to HVRT
    hcfg.blend_cross_term = cfg_.blend_cross_term;

    // Bandwidth: auto → HVRT handles it; numeric → pass as string
    if (cfg_.bandwidth > 0.0) {
        hcfg.bandwidth = std::to_string(cfg_.bandwidth);
    } else {
        hcfg.bandwidth = "auto";
    }

    // Partitioner type
    if      (cfg_.partitioner == "hart")
        hcfg.partitioner_type = hvrt::PartitionerType::HART;
    else if (cfg_.partitioner == "fasthvrt" || cfg_.partitioner == "fasthart")
        hcfg.partitioner_type = hvrt::PartitionerType::FastHART;
    else if (cfg_.partitioner == "pyramid_hart")
        hcfg.partitioner_type = hvrt::PartitionerType::PyramidHART;
    else
        hcfg.partitioner_type = hvrt::PartitionerType::HVRT;

    // Generation strategy enum (used by expander_.prepare() inside HVRT::fit/refit)
    {
        const std::string& gs = cfg_.generation_strategy;
        if      (gs == "simplex_mixup"    || gs == "SimplexMixup")    hcfg.gen_strategy = hvrt::GenerationStrategy::SimplexMixup;
        else if (gs == "laplace"          || gs == "Laplace")         hcfg.gen_strategy = hvrt::GenerationStrategy::Laplace;
        else if (gs == "multivariate_kde" || gs == "MultivariateKDE") hcfg.gen_strategy = hvrt::GenerationStrategy::MultivariateKDE;
        else if (gs == "bootstrap"        || gs == "BootstrapNoise")  hcfg.gen_strategy = hvrt::GenerationStrategy::BootstrapNoise;
        else if (gs == "copula"           || gs == "UnivariateCopula") hcfg.gen_strategy = hvrt::GenerationStrategy::UnivariateCopula;
        else if (gs == "auto"             || gs == "Auto")             hcfg.gen_strategy = hvrt::GenerationStrategy::Auto;
        else                                                            hcfg.gen_strategy = hvrt::GenerationStrategy::Epanechnikov;
    }

    std::shared_ptr<hvrt::HVRT> h;
    if (hvrt_out && hvrt_out->fitted()) {
        // Fast refit: skip whitening, binning, geometry target — only re-run tree + expander.
        // X_z_, X_binned_cache_, and geom_target_cache_ are already valid from the
        // previous fit() call.  bin_edges_ will also be reused inside tree_.build().
        h = hvrt_out;

        // ── Adaptive y_weight ────────────────────────────────────────────────
        // Compute Pearson ρ between the fixed geometry target (geom_target_cache_,
        // already zscored ~N(0,1)) and the current residuals (y_signal).
        // Scale y_weight by |ρ|: when residuals are noise (ρ≈0) → y_weight→0,
        // letting the partition tree be driven by pure geometry.
        double yw_eff = static_cast<double>(cfg_.y_weight);
        if (cfg_.adaptive_y_weight) {
            const Eigen::VectorXd& geom = h->geom_target();
            const int n = static_cast<int>(y_signal.size());
            // Compute Pearson ρ(geom, y_signal) in one pass — no heap allocation.
            // geom is already ~N(0,1) (output of compute_pairwise_target zscore).
            // ρ = Σ geom_i*(y_i - ȳ) / (σ_y * (n-1))
            double y_mean = y_signal.mean();
            double y_var  = (y_signal.array() - y_mean).square().mean();
            if (y_var > 1e-20 && n > 1) {
                double y_std = std::sqrt(y_var);
                const double* gp = geom.data();
                const double* yp = y_signal.data();
                double sum_xy = 0.0;
                for (int i = 0; i < n; ++i)
                    sum_xy += gp[i] * (yp[i] - y_mean);
                double rho = sum_xy / (y_std * (n - 1.0));
                rho = std::max(-1.0, std::min(1.0, rho));
                rho_trace_.push_back(std::abs(rho));
                yw_eff = cfg_.y_weight * std::abs(rho);
                yw_eff_trace_.push_back(yw_eff);
            }
        }
        // ── Approach 1: selective pairwise target ────────────────────────────
        // Replace the cached pure-geometry target with the top-k pair products
        // most correlated with current residuals (Pearson ranking).  High-|r|
        // pairs concentrate HVRT splits on directions that actually predict
        // the gradient — especially important when d is small and most pairs
        // are orthogonal to the residuals.
        if (cfg_.selective_target) {
            Eigen::VectorXd sel = hvrt::compute_selective_target(
                h->X_z(), y_signal, cfg_.selective_k_pairs);
            h->set_geom_target(sel);
        }

        // ── Strategy 3: y_signal coupling ───────────────────────────────────
        // Blend residuals with the cached pure-geometry target so HVRT always
        // sees some geometric structure even when residuals are noisy.
        // y_for_refit = (1-α)*z(y_signal) + α*geom_target, re-scaled to
        // y_signal's original range so blend_target's normalisation is unaffected.
        // Only active at refit (initial fit has no geom_target yet).
        Eigen::VectorXd y_for_refit = y_signal;
        if (cfg_.y_geom_coupling > 0.0) {
            const Eigen::VectorXd& geom = h->geom_target();
            const int n_s = static_cast<int>(y_signal.size());
            double y_mean = y_signal.mean();
            double y_var  = (y_signal.array() - y_mean).square().mean();
            if (y_var > 1e-20 && n_s > 1) {
                double y_std = std::sqrt(y_var);
                Eigen::VectorXd y_z = (y_signal.array() - y_mean) / y_std;
                // geom is already ~N(0,1) (output of compute_pairwise_target)
                Eigen::VectorXd combined = (1.0 - cfg_.y_geom_coupling) * y_z
                                         + cfg_.y_geom_coupling * geom;
                // Restore original scale so blend_target's [0,1] normalisation
                // and y_weight blending remain semantically consistent.
                y_for_refit = (combined.array() * y_std + y_mean).matrix();
            }
        }
        h->refit(y_for_refit, yw_eff);
    } else {
        h = std::make_shared<hvrt::HVRT>(hcfg);
        h->fit(X_full, y_signal);
        hvrt_out = h;
    }

    // Noise modulation
    double noise_mod = 1.0;
    if (cfg_.auto_noise) {
        noise_mod = estimate_noise_modulation(
            h->X_z(), y_signal, 10, cfg_.random_state + seed_offset);
    }

    // Effective reduce ratio: noisier data → keep more
    double eff_reduce = cfg_.reduce_ratio + (1.0 - noise_mod) * (1.0 - cfg_.reduce_ratio);
    eff_reduce = std::min(eff_reduce, 1.0);

    int n_keep = std::max(10, static_cast<int>(n * eff_reduce));

    // Adaptive reduce ratio: heavy-tailed gradients → increase reduce_ratio
    if (cfg_.adaptive_reduce_ratio && n > 10) {
        std::vector<double> sv(y_signal.data(), y_signal.data() + n);
        std::transform(sv.begin(), sv.end(), sv.begin(),
                       [](double v) { return std::abs(v); });
        int med_pos = n / 2;
        int p90_pos = static_cast<int>(n * 0.9);
        std::nth_element(sv.begin(), sv.begin() + med_pos, sv.end());
        double med_y = sv[med_pos];
        std::nth_element(sv.begin(), sv.begin() + p90_pos, sv.end());
        double p90_y   = sv[p90_pos];
        double tail_ratio  = p90_y / (med_y + 1e-12);
        double adapt_delta = std::clamp((tail_ratio - 1.5) / 20.0, 0.0, 0.15);
        eff_reduce = std::min(eff_reduce + adapt_delta, 1.0);
        n_keep = std::max(10, static_cast<int>(n * eff_reduce));
    }

    // Reduce: dispatch on reduce_method
    std::vector<int> red_idx_vec;
    if (cfg_.reduce_method == "orthant_stratified") {
        red_idx_vec = hvrt::orthant_stratified(
            h->X_z(), y_signal, n_keep, cfg_.random_state + seed_offset);
    } else {
        const std::string& rm = cfg_.reduce_method.empty() ? "variance" : cfg_.reduce_method;
        red_idx_vec = h->reduce_indices(n_keep, std::nullopt, rm, cfg_.variance_weighted);
    }
    Eigen::VectorXi red_idx = Eigen::Map<Eigen::VectorXi>(
        red_idx_vec.data(), static_cast<int>(red_idx_vec.size()));

    // Subset X and y
    const int n_red = static_cast<int>(red_idx.size());
    Eigen::MatrixXd X_red(n_red, X_full.cols());
    Eigen::VectorXd y_red(n_red);
    for (int i = 0; i < n_red; ++i) {
        X_red.row(i) = X_full.row(red_idx[i]);
        y_red[i]     = y_signal[red_idx[i]];
    }

    int n_expanded = 0;

    // ── Strategy 2: per-partition Y mean correction helper ───────────────────
    // After knn_assign_y() assigns y_syn via IDW, the per-partition synthetic
    // mean may drift from the real reduced mean (boundary effects: IDW averages
    // across partitions when the nearest neighbours straddle a boundary).
    // This lambda shifts each y_syn[i] by delta_p = mean(y_red_in_p) - mean(y_syn_in_p)
    // for the partition p that X_syn[i] was routed to.
    // Requirements: h fitted, red_idx valid, X_syn in original feature space.
    auto partition_correct = [&](Eigen::VectorXd& y_syn,
                                  const Eigen::MatrixXd& X_syn,
                                  const Eigen::VectorXi& r_idx,
                                  const Eigen::VectorXd& y_r) {
        if (!cfg_.syn_partition_correct) return;
        const Eigen::VectorXi& tr_pids = h->partition_ids(); // size n
        Eigen::VectorXi syn_pids = h->apply(X_syn);          // route X_syn → partition ids

        const int n_syn_loc = static_cast<int>(y_syn.size());
        const int n_red_loc = static_cast<int>(r_idx.size());

        // Accumulate real-sample per-partition sum/count
        std::unordered_map<int, std::pair<double,int>> real_stat;
        real_stat.reserve(n_red_loc);
        for (int i = 0; i < n_red_loc; ++i) {
            int pid = tr_pids[r_idx[i]];
            auto& s = real_stat[pid];
            s.first  += y_r[i];
            s.second += 1;
        }

        // Accumulate synthetic per-partition sum/count
        std::unordered_map<int, std::pair<double,int>> syn_stat;
        syn_stat.reserve(n_syn_loc);
        for (int i = 0; i < n_syn_loc; ++i) {
            auto& s = syn_stat[syn_pids[i]];
            s.first  += y_syn[i];
            s.second += 1;
        }

        // Apply delta correction per synthetic sample
        for (int i = 0; i < n_syn_loc; ++i) {
            int pid = syn_pids[i];
            auto ir = real_stat.find(pid);
            auto is = syn_stat.find(pid);
            if (ir != real_stat.end() && is != syn_stat.end() &&
                ir->second.second > 0  && is->second.second > 0) {
                double mean_real = ir->second.first / ir->second.second;
                double mean_syn  = is->second.first / is->second.second;
                y_syn[i] += (mean_real - mean_syn);
            }
        }
    };

    // Active generation strategy string (used in h->expand() calls below)
    const std::string& gs = cfg_.generation_strategy.empty()
                            ? "epanechnikov" : cfg_.generation_strategy;

    // Auto-expand: fill up to min_train_samples
    bool expansion_risk = cfg_.auto_expand && (n < cfg_.min_train_samples);
    if (expansion_risk && cfg_.expand_ratio == 0.0 && n_red < cfg_.min_train_samples) {
        int _eff_min = std::min(cfg_.min_train_samples,
                                std::max(n * 5, 1000));
        double eff_noise = std::max(noise_mod, 0.1);
        int n_expand = std::max(0, static_cast<int>((_eff_min - n_red) * eff_noise));

        if (n_expand > 0) {
            Eigen::MatrixXd X_syn = h->expand(n_expand, cfg_.variance_weighted,
                                               std::nullopt, gs);
            // Actual generated count may differ from requested (e.g. tiny partitions).
            n_expand = static_cast<int>(X_syn.rows());

            if (n_expand > 0) {
            // y-assignment: k-NN in z-space
            Eigen::MatrixXd X_red_z(n_red, h->X_z().cols());
            for (int i = 0; i < n_red; ++i)
                X_red_z.row(i) = h->X_z().row(red_idx[i]);
            Eigen::MatrixXd X_syn_z = h->to_z(X_syn);

            Eigen::VectorXd y_syn = knn_assign_y(X_syn_z, X_red_z, y_red);
            partition_correct(y_syn, X_syn, red_idx, y_red);  // S2
            // A3: per-partition z-space sub-split residual correction (refit only)
            if (!is_initial_fit && cfg_.residual_correct_lambda > 0.0) {
                Eigen::VectorXi syn_pids = h->apply(X_syn);
                apply_sub_split_correction(
                    h->X_z(), y_signal, h->partition_ids(),
                    X_syn_z, syn_pids, y_syn,
                    cfg_.residual_correct_lambda);
            }

            // Concatenate real + synthetic
            int n_out = n_red + n_expand;
            Eigen::MatrixXd X_out(n_out, X_full.cols());
            Eigen::VectorXd y_out(n_out);
            X_out.topRows(n_red)       = X_red;
            X_out.bottomRows(n_expand) = X_syn;
            y_out.head(n_red)          = y_red;
            y_out.tail(n_expand)       = y_syn;

            n_expanded = n_expand;

            ResampleResult r;
            r.X          = std::move(X_out);
            r.y          = std::move(y_out);
            r.red_idx    = red_idx;
            r.noise_mod  = noise_mod;
            r.n_expanded = n_expanded;
            return r;
            } // if (n_expand > 0) after actual rows check
        }
    }

    // Manual expand_ratio
    if (cfg_.expand_ratio > 0.0) {
        double eff_noise = std::max(noise_mod, 0.1);
        int n_expand = std::max(0, static_cast<int>(n * cfg_.expand_ratio * eff_noise));
        if (n_expand > 0) {
            Eigen::MatrixXd X_syn = h->expand(n_expand, cfg_.variance_weighted,
                                               std::nullopt, gs);
            n_expand = static_cast<int>(X_syn.rows()); // actual generated count

            if (n_expand > 0) {
            Eigen::MatrixXd X_red_z(n_red, h->X_z().cols());
            for (int i = 0; i < n_red; ++i)
                X_red_z.row(i) = h->X_z().row(red_idx[i]);
            Eigen::MatrixXd X_syn_z = h->to_z(X_syn);
            Eigen::VectorXd y_syn = knn_assign_y(X_syn_z, X_red_z, y_red);
            partition_correct(y_syn, X_syn, red_idx, y_red);  // S2
            // A3: per-partition z-space sub-split residual correction (refit only)
            if (!is_initial_fit && cfg_.residual_correct_lambda > 0.0) {
                Eigen::VectorXi syn_pids = h->apply(X_syn);
                apply_sub_split_correction(
                    h->X_z(), y_signal, h->partition_ids(),
                    X_syn_z, syn_pids, y_syn,
                    cfg_.residual_correct_lambda);
            }

            int n_out = n_red + n_expand;
            Eigen::MatrixXd X_out(n_out, X_full.cols());
            Eigen::VectorXd y_out(n_out);
            X_out.topRows(n_red)       = X_red;
            X_out.bottomRows(n_expand) = X_syn;
            y_out.head(n_red)          = y_red;
            y_out.tail(n_expand)       = y_syn;
            n_expanded = n_expand;

            ResampleResult r;
            r.X = std::move(X_out); r.y = std::move(y_out);
            r.red_idx = red_idx; r.noise_mod = noise_mod; r.n_expanded = n_expanded;
            return r;
            } // if (n_expand > 0) after actual rows check
        }
    }

    ResampleResult r;
    r.X          = std::move(X_red);
    r.y          = std::move(y_red);
    r.red_idx    = red_idx;
    r.noise_mod  = noise_mod;
    r.n_expanded = 0;
    return r;
}

// ── predict_from_trees ────────────────────────────────────────────────────────
// Each tree's contribution is independent → embarrassingly parallel.
// Pattern: thread-local accumulator + single critical-section reduction.
// predict_into() reuses a pre-allocated buffer per thread, eliminating the
// per-tree heap allocation that predict() would otherwise incur.

Eigen::VectorXd GeoXGBBase::predict_from_trees(
        const Eigen::MatrixXd& X, int up_to_tree) const
{
    const int n   = static_cast<int>(X.rows());
    const int lim = (up_to_tree < 0) ? static_cast<int>(trees_.size()) : up_to_tree;

    Eigen::VectorXd p = Eigen::VectorXd::Constant(n, init_pred_);
    if (lim == 0) return p;

#ifdef _OPENMP
    #pragma omp parallel
    {
        Eigen::VectorXd local = Eigen::VectorXd::Zero(n); // thread-local accumulator
        Eigen::VectorXd tmp(n);                            // reused buffer per tree

        #pragma omp for schedule(static)
        for (int t = 0; t < lim; ++t) {
            trees_[t].tree.predict_into(X, tmp);
            local.noalias() += trees_[t].lr * tmp;
        }

        #pragma omp critical
        p += local;  // one reduction per thread, not per tree
    }
#else
    Eigen::VectorXd tmp(n);
    for (int t = 0; t < lim; ++t) {
        trees_[t].tree.predict_into(X, tmp);
        p.noalias() += trees_[t].lr * tmp;
    }
#endif
    return p;
}

// ── predict_raw ───────────────────────────────────────────────────────────────

Eigen::VectorXd GeoXGBBase::predict_raw(const Eigen::MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("Model not fitted");
    return predict_from_trees(X, -1);
}

// ── fit_boosting ──────────────────────────────────────────────────────────────

void GeoXGBBase::fit_boosting(
        const Eigen::MatrixXd& X_arg,
        const Eigen::VectorXd& y_arg)
{
    // ── Block cycling setup ───────────────────────────────────────────────────
    // When sample_block_n > 0 and n_arg > sample_block_n, the full dataset is
    // divided into non-overlapping blocks via a deterministic LCG permutation.
    // At each refit_interval the boosting loop advances to the next block; all
    // per-round costs scale with sample_block_n rather than n_arg.  Different
    // blocks cover different geometric regions, giving progressive data coverage.
    // Future work: pre-fetch next block asynchronously via std::future (Phase 2).
    const int  n_arg       = static_cast<int>(X_arg.rows());
    const int  d           = static_cast<int>(X_arg.cols());
    const bool block_cycle = (cfg_.sample_block_n > 0 && n_arg > cfg_.sample_block_n);
    const int  blk_sz      = block_cycle ? cfg_.sample_block_n : n_arg;

    // Mutable working matrices — updated at each block switch (or assigned once).
    Eigen::MatrixXd X_cur;
    Eigen::VectorXd y_cur;

    // Block permutation state
    std::vector<int> blk_perm;
    int blk_ctr = 0, blk_epoch = 0, n_usable_blocks = 1;

    // LCG-based Fisher-Yates in-place shuffle of blk_perm.
    auto lcg_permute = [&](int epoch_seed) {
        std::iota(blk_perm.begin(), blk_perm.end(), 0);
        uint64_t lcg = static_cast<uint64_t>(cfg_.random_state + epoch_seed)
                       * 6364136223846793005ULL + 1442695040888963407ULL;
        for (int i = 0; i < n_arg - 1; ++i) {
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = i + static_cast<int>((lcg >> 33)
                        % static_cast<uint64_t>(n_arg - i));
            std::swap(blk_perm[i], blk_perm[j]);
        }
    };

    // Slice the current block from X_arg / y_arg into X_cur / y_cur.
    auto slice_block = [&]() {
        const int start = blk_ctr * blk_sz;
        X_cur.resize(blk_sz, d);
        y_cur.resize(blk_sz);
        for (int i = 0; i < blk_sz; ++i) {
            X_cur.row(i) = X_arg.row(blk_perm[start + i]);
            y_cur[i]     = y_arg[blk_perm[start + i]];
        }
    };

    // Advance to next block, reshuffling permutation at epoch boundary.
    auto advance_block = [&]() {
        blk_ctr++;
        if (blk_ctr >= n_usable_blocks) {
            blk_ctr = 0;
            blk_epoch++;
            lcg_permute(blk_epoch);
        }
        slice_block();
    };

    if (block_cycle) {
        const int n_total = n_arg / blk_sz;
        n_usable_blocks = std::max(1, n_total - (cfg_.leave_last_block_out ? 1 : 0));
        blk_perm.resize(n_arg);
        lcg_permute(0);
        slice_block();  // X_cur = block 0, y_cur = block 0 targets
    } else {
        X_cur = X_arg;  // one-time O(n) copy for the non-cycling path
        y_cur = y_arg;
    }

    // n is the current block size; constant when !block_cycle.
    int n = blk_sz;

    trees_.clear();
    trees_.reserve(cfg_.n_rounds);
    convergence_round_ = -1;
    rho_trace_.clear();
    yw_eff_trace_.clear();
    convergence_losses_.clear();
    n_train_arg_ = n_arg;

    // ── Bin for GBT weak learner trees ───────────────────────────────────────
    // GBT trees work on the original feature space (no whitening).
    // When block cycling, fit bin edges on X_arg (full data) for consistent
    // thresholds across blocks; then transform each block on demand.
    cont_cols_full_.resize(d);
    std::iota(cont_cols_full_.begin(), cont_cols_full_.end(), 0);

    full_binner_.fit(block_cycle ? X_arg : X_cur, cfg_.n_bins);
    X_bin_full_ = full_binner_.transform(X_cur);
    // Cache bin edges once; injected into every GBT weak learner to skip
    // the per-round O(n·d·log n) re-sort inside PartitionTree::build().
    // Using the same edges that produced X_bin_full_ also ensures consistency:
    // split thresholds are aligned with the pre-computed bin assignments.
    gbt_bin_edges_ = full_binner_.edges();

    // ── Approach 2: pure GBT fast path (skip HVRT for low-d) ─────────────────
    // When d <= d_geom_threshold, HVRT's pairwise cooperation target has too
    // few feature pairs to align with the gradient direction.  Running full
    // GBT on the entire training set (no reduce/expand/refit) avoids discarding
    // informative points and eliminates the HVRT overhead entirely.
    // Threshold <= 0 disables this path (HVRT always active — default).
    const bool use_hvrt = (cfg_.d_geom_threshold <= 0) || (d > cfg_.d_geom_threshold);
    if (!use_hvrt) {
        init_pred_ = init_prediction(y_cur);
        Eigen::VectorXd preds_pg = Eigen::VectorXd::Constant(n, init_pred_);

        for (int i = 0; i < cfg_.n_rounds; ++i) {
            Eigen::VectorXd grads = gradients(y_cur, preds_pg);
            if (!grads.allFinite()) grads.setZero();

            hvrt::HVRTConfig tcfg;
            tcfg.n_partitions     = 2 * (1 << cfg_.max_depth);
            tcfg.min_samples_leaf = cfg_.min_samples_leaf;
            tcfg.max_depth        = cfg_.max_depth;
            tcfg.n_bins           = cfg_.n_bins;
            tcfg.auto_tune        = false;
            tcfg.random_state     = cfg_.random_state + i;
            tcfg.split_strategy   = hvrt::SplitStrategy::Random;

            WeakLearner wl;
            wl.cont_cols = cont_cols_full_;
            wl.lr        = cfg_.learning_rate;
            wl.tree.inject_bin_edges(gbt_bin_edges_, cont_cols_full_, cfg_.n_bins);
            wl.tree.build(X_cur, X_bin_full_, cont_cols_full_, {}, grads, tcfg);

            Eigen::VectorXd lp = wl.tree.predict(X_cur);
            preds_pg.noalias() += cfg_.learning_rate * lp;
            trees_.push_back(std::move(wl));
        }
        fitted_ = true;
        return;
    }

    // ── Initial resample ──────────────────────────────────────────────────────
    std::shared_ptr<hvrt::HVRT> last_hvrt;
    ResampleResult res = do_resample(X_cur, y_cur, 0, last_hvrt);

    Eigen::MatrixXd Xr      = res.X;
    Eigen::VectorXd yr      = res.y;
    Eigen::VectorXi red_idx = res.red_idx;
    double last_noise_mod   = res.noise_mod;
    last_noise_mod_         = last_noise_mod;
    init_noise_mod_         = last_noise_mod;
    n_init_reduced_         = static_cast<int>(res.red_idx.size());
    int    last_n_expanded  = res.n_expanded;

    init_pred_ = init_prediction(yr);

    Eigen::VectorXd preds       = Eigen::VectorXd::Constant(Xr.rows(), init_pred_);
    Eigen::VectorXd preds_on_X  = Eigen::VectorXd::Constant(n, init_pred_);

    bool expansion_risk = cfg_.auto_expand && (n < cfg_.min_train_samples);

    // ── X_bin_r cache ─────────────────────────────────────────────────────────
    // The binned reduced matrix only changes when Xr changes (at resample
    // boundaries). Recomputing it every round — via a subset copy or a full
    // full_binner_.transform() call — is redundant and measurably expensive.
    // xr_changed tracks whether Xr has been updated since X_bin_r was last built.
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X_bin_r;
    bool xr_changed = true;   // force build on round 0

    // ── Preds sync helper (lambda) ────────────────────────────────────────────
    // When Xr = [X_real | X_syn], real-sample predictions are already
    // accumulated in preds_on_X (updated every round via lp_X = tree.predict(X)).
    // Only the synthetic tail lacks precomputed predictions and must be obtained
    // via predict_from_trees.  At late rounds (many accumulated trees), this cuts
    // the predict_from_trees workload from n_out=~4000 to n_syn=~500.
    auto sync_preds = [&](int n_expanded) {
        const int n_real = static_cast<int>(red_idx.size());
        if (n_expanded == 0) {
            preds.resize(n_real);
            for (int s = 0; s < n_real; ++s)
                preds[s] = preds_on_X[red_idx[s]];
        } else {
            const int n_syn = static_cast<int>(Xr.rows()) - n_real;
            preds.resize(n_real + n_syn);
            for (int s = 0; s < n_real; ++s)
                preds[s] = preds_on_X[red_idx[s]];
            if (n_syn > 0) {
                Eigen::VectorXd syn_p = predict_from_trees(Xr.bottomRows(n_syn), -1);
                preds.tail(n_syn) = syn_p;
            }
        }
    };


    // ── Boosting loop ─────────────────────────────────────────────────────────
    bool converged_early = false;
    for (int i = 0; i < cfg_.n_rounds; ++i) {

        // ── Refit ─────────────────────────────────────────────────────────────
        if (cfg_.refit_interval > 0 && i > 0 && (i % cfg_.refit_interval == 0)) {
            // Block cycling: advance to next non-overlapping block, resync state.
            bool block_just_advanced = false;
            if (block_cycle) {
                advance_block();
                n = blk_sz;
                X_bin_full_ = full_binner_.transform(X_cur);
                xr_changed  = true;
                last_hvrt   = nullptr;  // force fresh HVRT fit on new block
                preds_on_X  = predict_from_trees(X_cur, -1);
                expansion_risk = cfg_.auto_expand && (n < cfg_.min_train_samples);
                block_just_advanced = true;
                // Reset noise estimate: on a fresh block red_idx is stale so
                // skip_refit must NOT fire (would use wrong indices for preds sync).
                last_noise_mod = 1.0;
            }
            Eigen::VectorXd grads_orig = gradients(y_cur, preds_on_X);

            // ── Convergence check ──────────────────────────────────────────
            if (cfg_.convergence_tol > 0.0) {
                double loss_now = grads_orig.array().abs().mean();
                convergence_losses_.push_back(loss_now);
                if (convergence_losses_.size() >= 3) {
                    double prev = convergence_losses_[convergence_losses_.size() - 3];
                    double rel  = (prev - loss_now) / (prev + 1e-12);
                    if (rel < cfg_.convergence_tol) {
                        convergence_round_ = i;
                        converged_early    = true;
                    }
                }
            }
            if (converged_early) break;

            bool skip_refit = !block_just_advanced
                              && cfg_.noise_guard && cfg_.auto_noise
                              && expansion_risk
                              && (last_noise_mod < cfg_.refit_noise_floor);

            if (!skip_refit) {
                // Refit: pass last_hvrt so do_resample() can reuse cached whitener/binner/
                // geometry-target via HVRT::refit(), saving O(n·d²) + O(n²·d) per refit.
                ResampleResult new_res = do_resample(X_cur, grads_orig, i, last_hvrt);
                last_noise_mod  = new_res.noise_mod;
                last_noise_mod_ = last_noise_mod;

                // Safety: discard if the resampled data contains NaN/Inf
                bool finite_ok = new_res.X.allFinite() && new_res.y.allFinite();

                bool discard = !finite_ok
                               || (cfg_.noise_guard && cfg_.auto_noise
                                   && expansion_risk
                                   && (new_res.noise_mod < cfg_.refit_noise_floor));

                if (!discard) {
                    Xr              = new_res.X;
                    yr              = new_res.y;
                    red_idx         = new_res.red_idx;
                    last_n_expanded = new_res.n_expanded;
                    xr_changed      = true;
                    // last_hvrt already points to the refitted HVRT (updated in-place)

                    sync_preds(new_res.n_expanded);
                    // Reconstruct yr: regression adds, classifier subclass overrides
                    yr = targets_from_gradients(new_res.y, preds);
                } else if (block_just_advanced) {
                    // Discard on a fresh block: red_idx is stale — recompute preds
                    // on the existing Xr (from old block) via full tree traversal.
                    // yr stays as-is (old block targets) since Xr is unchanged.
                    preds = predict_from_trees(Xr, -1);
                } else {
                    // Discard: Xr unchanged; sync preds from preds_on_X
                    sync_preds(last_n_expanded);
                }
            } else {
                // Pure skip: Xr unchanged; sync preds from preds_on_X
                sync_preds(last_n_expanded);
            }
        }

        // ── Fit GBT weak learner ─────────────────────────────────────────────
        Eigen::VectorXd grads = gradients(yr, preds);

        // Guard: NaN/Inf in grads (degenerate refit state) — zero them out.
        // A tree built on all-zero grads adds nothing and avoids corrupting predictions.
        if (!grads.allFinite()) grads.setZero();

        // Subset X_bin_full_ to rows in Xr.
        // When n_expanded == 0, Xr rows = X[red_idx]; we can subset X_bin_full_.
        // When synthetic samples present, they don't have precomputed bins —
        // we bin Xr on-the-fly using the full_binner_.
        // Recompute only when Xr has changed (xr_changed flag).
        if (xr_changed) {
            if (last_n_expanded == 0) {
                const int nr = static_cast<int>(red_idx.size());
                X_bin_r.resize(nr, d);
                for (int s = 0; s < nr; ++s)
                    X_bin_r.row(s) = X_bin_full_.row(red_idx[s]);
            } else {
                X_bin_r = full_binner_.transform(Xr);
            }
            xr_changed = false;
        }

        // Build tree config for the GBT weak learner.
        // n_partitions must be large enough to never constrain a depth-4 tree;
        // 2*2^max_depth is sufficient since BFS produces at most 2^max_depth leaves.
        // Use Random splits (ExtraTrees style) for the GBT weak learners:
        // each feature's threshold is chosen at random, matching sklearn's
        // DecisionTreeRegressor(splitter="random") used by the Python backend.
        hvrt::HVRTConfig tcfg;
        tcfg.n_partitions     = 2 * (1 << cfg_.max_depth);  // e.g. 32 for depth=4
        tcfg.min_samples_leaf = cfg_.min_samples_leaf;
        tcfg.max_depth        = cfg_.max_depth;
        tcfg.n_bins           = cfg_.n_bins;
        tcfg.auto_tune        = false;
        tcfg.random_state     = cfg_.random_state + i;
        tcfg.split_strategy   = hvrt::SplitStrategy::Random;

        WeakLearner wl;
        wl.cont_cols = cont_cols_full_;
        wl.lr        = cfg_.learning_rate;

        // Inject pre-computed bin edges so build() skips the O(n·d·log n) sort.
        wl.tree.inject_bin_edges(gbt_bin_edges_, cont_cols_full_, cfg_.n_bins);
        wl.tree.build(Xr,           // raw X (no whitening for GBT trees)
                      X_bin_r,      // pre-binned X subset
                      cont_cols_full_,
                      {},           // no binary columns for GBT
                      grads,
                      tcfg);

        // Accumulate predictions
        Eigen::VectorXd lp_Xr = wl.tree.predict(Xr);
        Eigen::VectorXd lp_X  = wl.tree.predict(X_cur);
        preds       += cfg_.learning_rate * lp_Xr;
        preds_on_X  += cfg_.learning_rate * lp_X;

        trees_.push_back(std::move(wl));
    }

    fitted_ = true;

    // Persist HVRT geometry for interpretability APIs.
    // X_z_ and partition_ids_ come from the last HVRT (geometry for X_cur / last block).
    // train_predictions_ covers the full X_arg so interpretability callers always get
    // predictions for all training samples.
    if (last_hvrt && last_hvrt->fitted()) {
        last_hvrt_         = last_hvrt;
        X_z_               = last_hvrt->X_z();
        partition_ids_     = last_hvrt->partition_ids();
        train_predictions_ = predict_from_trees(X_arg, -1);
    }
}

} // namespace geoxgb
