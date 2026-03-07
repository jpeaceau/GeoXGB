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
#include <future>

// Flush denormals to zero: tiny floating-point values (< ~1e-308) trigger
// microcode traps on x86, ~100x slower than normal FP ops.  Gradient residuals
// can decay to denormal range in late boosting rounds.  FTZ/DAZ avoids this.
#if defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define GEOXGB_FTZ_GUARD() \
    _mm_setcsr(_mm_getcsr() | 0x8040) /* FTZ (bit 15) + DAZ (bit 6) */
#else
#define GEOXGB_FTZ_GUARD() ((void)0)
#endif

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
    const int d_out = static_cast<int>(cont_cols_full_.size());
    std::vector<double> total(d_out, 0.0);
    for (const auto& wl : trees_) {
        const auto& fi = wl.tree.feature_importances();
        const int nd = static_cast<int>(std::min(fi.size(),
                       static_cast<size_t>(d_out)));
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
    hcfg.n_bins       = 32;  // HVRT partition tree bins
    hcfg.random_state = cfg_.random_state + seed_offset;
    hcfg.skip_expander = cfg_.fast_refit;  // no expand → no need to fit KDE

    // GeoXGB-specific auto-tune: HVRT's standalone for_reduction formula
    // (msl = d*80/3) is designed for density estimation and is far too
    // conservative inside a boosting loop — e.g. d=19 → msl=506, yielding
    // only 4 partitions on a 9k block.  GeoXGB needs finer partitions to
    // capture local structure for the GBT weak learners.
    //
    // Formula: msl = max(10, ceil(sqrt(n / d)))
    //   d=19, n=9400  → msl=22, ~427 max partitions
    //   d=8,  n=5000  → msl=25, ~200 max partitions
    //   d=50, n=10000 → msl=15, ~666 max partitions
    // The sqrt(n/d) scaling ensures msl grows slowly with n (more data →
    // slightly larger leaves) and shrinks with d (more features → need
    // more partitions to capture interactions).
    if (cfg_.hvrt_min_samples_leaf > 0) {
        hcfg.min_samples_leaf = cfg_.hvrt_min_samples_leaf;
        hcfg.auto_tune        = false;
    } else if (cfg_.hvrt_n_partitions > 0) {
        hcfg.n_partitions = cfg_.hvrt_n_partitions;
        hcfg.auto_tune    = false;
    } else {
        // GeoXGB auto: override HVRT's for_reduction formula.
        // HVRT standalone uses msl = d*80/3 (for_reduction) which is far too
        // conservative inside a boosting loop.  GeoXGB needs finer partitions.
        // Use msl=10 unconditionally: the HVRT tree's variance-reduction
        // splitting naturally limits the actual partition count based on the
        // data's structure.  On categorical-heavy data (few unique values per
        // feature), the tree self-limits even at msl=10.  On continuous data,
        // msl=10 allows rich partitioning that captures local geometry.
        const int geoxgb_msl = 10;
        const int geoxgb_max_leaf = std::max(30,
            std::min(1500, 3 * n / (geoxgb_msl * 2)));
        hcfg.min_samples_leaf = geoxgb_msl;
        hcfg.n_partitions     = geoxgb_max_leaf;
        hcfg.auto_tune        = false;
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
        // Test B: Fixed geometry — skip partition tree refit, keep frozen geometry
        if (!cfg_.fixed_geometry) {
            h->refit(y_for_refit, yw_eff);
        }
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
    // fast_refit on refits: use stratified (random) selection — O(n) instead of
    // O(n²/partition) for variance_ordered.  Initial fit still uses full method.
    std::vector<int> red_idx_vec;
    const bool use_fast_reduce = cfg_.fast_refit && !is_initial_fit;

    // ── Gradient-aware budget allocation (deterministic GOSS via HVRT) ───────
    // At refit, weight each partition's reduction budget by its gradient mass:
    //   budget_p ∝ (1−α)·(n_p/n) + α·(Σ_{i∈p}|g_i| / Σ|g|)
    // High-gradient partitions (near decision boundary) retain more samples;
    // low-gradient partitions (well-predicted) get reduced more aggressively.
    // Within each partition, variance_ordered preserves geometric diversity.
    const bool use_grad_budgets = !is_initial_fit
                                  && cfg_.grad_budget_weight > 0.0
                                  && !use_fast_reduce;

    if (use_grad_budgets) {
        const double alpha = cfg_.grad_budget_weight;
        const Eigen::VectorXi& part_ids = h->partition_ids();
        const Eigen::MatrixXd& X_z = h->X_z();
        const int n_parts = part_ids.maxCoeff() + 1;

        // Accumulate per-partition |gradient| mass and sample counts
        std::vector<double> grad_mass(n_parts, 0.0);
        std::vector<int>    part_sizes(n_parts, 0);
        std::vector<std::vector<int>> part_indices(n_parts);
        for (int i = 0; i < n; ++i) {
            const int p = part_ids[i];
            grad_mass[p] += std::abs(y_signal[i]);
            part_sizes[p]++;
            part_indices[p].push_back(i);
        }

        double total_grad = 0.0;
        for (int p = 0; p < n_parts; ++p) total_grad += grad_mass[p];

        // Compute blended budgets: (1-α)·size + α·gradient_mass
        std::vector<int> budgets(n_parts, 0);
        std::vector<double> frac(n_parts);
        int allocated = 0;
        for (int p = 0; p < n_parts; ++p) {
            if (part_sizes[p] == 0) continue;
            double w_size = static_cast<double>(part_sizes[p]) / n;
            double w_grad = (total_grad > 1e-12)
                          ? grad_mass[p] / total_grad
                          : w_size;  // fallback if all gradients zero
            double w = (1.0 - alpha) * w_size + alpha * w_grad;
            double exact = w * n_keep / ((1.0 - alpha) + alpha);  // normalise
            // Simpler: w already sums to 1.0 by construction, so:
            exact = w * n_keep;
            int floor_val = std::max(1, static_cast<int>(std::floor(exact)));
            floor_val = std::min(floor_val, part_sizes[p]);
            budgets[p] = floor_val;
            frac[p] = exact - floor_val;
            allocated += budgets[p];
        }

        // Greedy ±1 correction to hit n_keep exactly
        int diff = n_keep - allocated;
        if (diff > 0) {
            std::vector<int> order(n_parts);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return frac[a] > frac[b]; });
            for (int k = 0; k < diff && k < n_parts; ++k) {
                int p = order[k];
                if (budgets[p] < part_sizes[p]) { ++budgets[p]; }
            }
        } else if (diff < 0) {
            std::vector<int> order(n_parts);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return frac[a] < frac[b]; });
            for (int k = 0; k < -diff && k < n_parts; ++k) {
                int p = order[k];
                if (budgets[p] > 1) { --budgets[p]; }
            }
        }

        // Per-partition selection: variance_ordered for geometric diversity
        red_idx_vec.reserve(n_keep);
        for (int p = 0; p < n_parts; ++p) {
            if (budgets[p] <= 0 || part_indices[p].empty()) continue;
            const auto& pidx = part_indices[p];
            const int ps = static_cast<int>(pidx.size());

            if (budgets[p] >= ps) {
                // Keep all samples in this partition
                for (int idx : pidx) red_idx_vec.push_back(idx);
            } else {
                // Build sub-matrix for variance_ordered
                Eigen::MatrixXd X_part(ps, X_z.cols());
                for (int k = 0; k < ps; ++k)
                    X_part.row(k) = X_z.row(pidx[k]);

                std::vector<int> local_sel = hvrt::variance_ordered(
                    X_part, budgets[p]);

                for (int li : local_sel)
                    red_idx_vec.push_back(pidx[li]);
            }
        }

    } else if (use_fast_reduce) {
        red_idx_vec = h->reduce_indices(n_keep, std::nullopt, "stratified", cfg_.variance_weighted);
    } else if (cfg_.reduce_method == "orthant_stratified") {
        red_idx_vec = hvrt::orthant_stratified(
            h->X_z(), y_signal, n_keep, cfg_.random_state + seed_offset);
    } else {
        const std::string& rm = cfg_.reduce_method.empty() ? "variance" : cfg_.reduce_method;
        red_idx_vec = h->reduce_indices(n_keep, std::nullopt, rm, cfg_.variance_weighted);
    }
    Eigen::VectorXi red_idx = Eigen::Map<Eigen::VectorXi>(
        red_idx_vec.data(), static_cast<int>(red_idx_vec.size()));

    // Subset y (always needed). X_red is built lazily: only when expansion is
    // active.  When no expansion, the boosting loop uses X_bin_full_[red_idx]
    // directly, avoiding an O(n_red × d × 8) byte copy per refit.
    const int n_red = static_cast<int>(red_idx.size());
    const int d_full = static_cast<int>(X_full.cols());
    Eigen::VectorXd y_red(n_red);
    for (int i = 0; i < n_red; ++i)
        y_red[i] = y_signal[red_idx[i]];

    // Lazy X_red builder: only materializes the double matrix when needed
    // (synthetic expansion requires concatenation with X_syn).
    Eigen::MatrixXd X_red;
    auto ensure_X_red = [&]() {
        if (X_red.rows() == n_red) return;  // already built
        X_red.resize(n_red, d_full);
        for (int i = 0; i < n_red; ++i)
            X_red.row(i) = X_full.row(red_idx[i]);
    };

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

    // fast_refit: skip expand entirely — use only reduced real data.
    // Eliminates KDE generation + knn_assign_y GEMM + to_z whitening.
    // Applied on ALL fits (initial + refits) since at 475k the initial expand
    // with knn_assign_y is itself the primary bottleneck.
    const bool skip_expand = cfg_.fast_refit;

    // Auto-expand: fill up to min_train_samples
    bool expansion_risk = cfg_.auto_expand && (n < cfg_.min_train_samples);
    if (!skip_expand && expansion_risk && cfg_.expand_ratio == 0.0 && n_red < cfg_.min_train_samples) {
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
            ensure_X_red();  // lazy build for concatenation
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
    if (!skip_expand && cfg_.expand_ratio > 0.0) {
        double eff_noise = std::max(noise_mod, 0.1);
        // Test C: Progressive expand — scale expand_ratio by training progress
        double eff_expand_ratio = cfg_.expand_ratio;
        if (cfg_.progressive_expand && cfg_.n_rounds > 1 && seed_offset > 0) {
            eff_expand_ratio *= static_cast<double>(seed_offset) / cfg_.n_rounds;
        }
        int n_expand = std::max(0, static_cast<int>(n * eff_expand_ratio * eff_noise));
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

            ensure_X_red();  // lazy build for concatenation
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

    // No expansion: still build X_red for tree.build() which needs n=X_z.rows().
    ensure_X_red();
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
    GEOXGB_FTZ_GUARD();  // flush denormals to zero for the entire fit

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

    // ── Pre-allocated per-round buffers (hoisted out of loop) ────────────────
    // Avoids heap allocation/deallocation every round.  resize() is a no-op
    // when the size doesn't change, which is the common case between refits.
    Eigen::VectorXd lp_X_buf(n);          // full-dataset predict buffer
    Eigen::VectorXd lp_Xr_buf;            // reduced-set predict buffer
    Eigen::VectorXd grads_eff_buf;        // GOSS gradient subset
    Eigen::MatrixXd Xr_eff_buf;           // GOSS X subset
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X_bin_eff_buf;
    std::vector<int> sorted_idx_buf;      // GOSS sort indices

    // ── Boosting loop ─────────────────────────────────────────────────────────
    bool converged_early = false;
    double last_refit_grad_mag = -1.0;  // Test A: lazy refit tracking
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
                last_hvrt = nullptr;  // force fresh HVRT fit on new block
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

            // Test A: Lazy refit — skip when gradient magnitude hasn't changed
            if (!skip_refit && cfg_.lazy_refit_tol > 0.0 && !block_just_advanced) {
                double grad_mag = grads_orig.array().abs().mean();
                if (last_refit_grad_mag > 0.0) {
                    double rel_change = std::abs(grad_mag - last_refit_grad_mag)
                                        / (last_refit_grad_mag + 1e-12);
                    if (rel_change < cfg_.lazy_refit_tol) {
                        skip_refit = true;
                    }
                }
                if (!skip_refit) last_refit_grad_mag = grad_mag;
            }

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

        // ── Gradient amplification via HVRT partition structure ───────────
        // Deterministic alternative to GOSS.  For each HVRT partition,
        // compute per-partition gradient statistics on X_cur (full block).
        // Partitions that straddle the decision boundary have high gradient
        // magnitude.  Amplify those gradients so the tree focuses splits
        // on boundary-relevant regions.
        //
        // The amplification is applied to the *full block* gradient array
        // (grads_orig computed at refit) and cached as per-sample weights
        // for the reduced set.  Between refits, the weights are reused.
        //
        // weight_p = (mean|grad|_p / mean|grad|_overall)^alpha
        // where alpha = grad_budget_weight.  This is a power transform:
        //   alpha=0 → all weights 1 (no effect)
        //   alpha=1 → weight proportional to partition gradient magnitude
        //
        // Xr samples are mapped back to partitions via red_idx → partition_ids_.
        if (cfg_.grad_budget_weight > 0.0 && last_hvrt && last_hvrt->fitted()) {
            const double alpha = cfg_.grad_budget_weight;
            const Eigen::VectorXi& full_pids = last_hvrt->partition_ids();
            const int n_parts = full_pids.maxCoeff() + 1;
            const int nr = static_cast<int>(grads.size());

            // Compute per-partition mean |gradient| from the full block
            // (only needs recomputing at refit; cache between refits)
            if (i == 0 || (cfg_.refit_interval > 0 && i % cfg_.refit_interval == 0)) {
                Eigen::VectorXd grads_full = gradients(y_cur, preds_on_X);
                grad_amp_weights_.resize(n_parts);
                std::vector<double> part_grad_sum(n_parts, 0.0);
                std::vector<int> part_cnt(n_parts, 0);
                for (int j = 0; j < n; ++j) {
                    const int p = full_pids[j];
                    part_grad_sum[p] += std::abs(grads_full[j]);
                    part_cnt[p]++;
                }
                // Global mean |gradient|
                double global_mean = 0.0;
                int global_cnt = 0;
                for (int p = 0; p < n_parts; ++p) {
                    if (part_cnt[p] > 0) {
                        grad_amp_weights_[p] = part_grad_sum[p] / part_cnt[p];
                        global_mean += part_grad_sum[p];
                        global_cnt += part_cnt[p];
                    } else {
                        grad_amp_weights_[p] = 0.0;
                    }
                }
                global_mean /= (global_cnt > 0 ? global_cnt : 1);
                // Convert to power-scaled amplification factors
                for (int p = 0; p < n_parts; ++p) {
                    double ratio = (global_mean > 1e-12)
                                 ? grad_amp_weights_[p] / global_mean
                                 : 1.0;
                    grad_amp_weights_[p] = std::pow(ratio, alpha);
                }
            }

            // Apply cached partition weights to Xr gradients.
            // Real samples: use red_idx → full_pids mapping.
            // Synthetic samples: use last_hvrt->apply() to route.
            const int n_real = static_cast<int>(red_idx.size());
            if (!grad_amp_weights_.empty()) {
                for (int j = 0; j < std::min(n_real, nr); ++j) {
                    const int p = full_pids[red_idx[j]];
                    if (p < static_cast<int>(grad_amp_weights_.size()))
                        grads[j] *= grad_amp_weights_[p];
                }
                // Synthetic samples (tail of Xr): route through HVRT apply
                if (nr > n_real && last_n_expanded > 0) {
                    Eigen::VectorXi syn_pids = last_hvrt->apply(
                        Xr.bottomRows(nr - n_real));
                    for (int j = n_real; j < nr; ++j) {
                        const int p = syn_pids[j - n_real];
                        if (p < static_cast<int>(grad_amp_weights_.size()))
                            grads[j] *= grad_amp_weights_[p];
                    }
                }
            }
        }

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

        // ── GOSS: Gradient-based One-Side Sampling ─────────────────────────
        // Keep top goss_alpha large-gradient samples + randomly sample goss_beta
        // of the remainder.  Upweight the randomly sampled ones.
        // Operates on the reduced set (grads, X_bin_r, Xr).
        // Buffers are pre-allocated above the loop; resize() is a no-op when
        // sizes are unchanged (common case between refits).
        const int nr_orig = static_cast<int>(grads.size());
        bool using_goss = false;

        if (cfg_.goss_alpha > 0.0 && cfg_.goss_beta > 0.0 && nr_orig > 20) {
            const int n_top = std::max(1, static_cast<int>(nr_orig * cfg_.goss_alpha));
            const int n_rand = std::max(1, static_cast<int>((nr_orig - n_top) * cfg_.goss_beta));

            // Sort by |gradient| descending (reuse pre-allocated buffer)
            sorted_idx_buf.resize(nr_orig);
            std::iota(sorted_idx_buf.begin(), sorted_idx_buf.end(), 0);
            std::partial_sort(sorted_idx_buf.begin(), sorted_idx_buf.begin() + n_top,
                              sorted_idx_buf.end(), [&](int a, int b) {
                return std::abs(grads[a]) > std::abs(grads[b]);
            });

            // Random sample from the remainder
            uint64_t goss_lcg = static_cast<uint64_t>(cfg_.random_state + i) * 2654435761ULL + 1;
            for (int j = n_top; j < n_top + n_rand && j < nr_orig; ++j) {
                goss_lcg = goss_lcg * 6364136223846793005ULL + 1442695040888963407ULL;
                int k = n_top + static_cast<int>((goss_lcg >> 33)
                        % static_cast<uint64_t>(nr_orig - n_top));
                std::swap(sorted_idx_buf[j], sorted_idx_buf[k]);
            }

            const int n_goss = n_top + n_rand;
            const double upweight = static_cast<double>(nr_orig - n_top) / n_rand;

            grads_eff_buf.resize(n_goss);
            Xr_eff_buf.resize(n_goss, Xr.cols());
            X_bin_eff_buf.resize(n_goss, X_bin_r.cols());
            for (int j = 0; j < n_goss; ++j) {
                const int si = sorted_idx_buf[j];
                Xr_eff_buf.row(j) = Xr.row(si);
                X_bin_eff_buf.row(j) = X_bin_r.row(si);
                grads_eff_buf[j] = (j >= n_top) ? grads[si] * upweight : grads[si];
            }
            using_goss = true;
        }

        const auto& Xr_build    = using_goss ? Xr_eff_buf : Xr;
        const auto& grads_build = using_goss ? grads_eff_buf : grads;
        const auto& Xbin_build  = using_goss ? X_bin_eff_buf : X_bin_r;

        // Build tree config for the GBT weak learner.
        hvrt::HVRTConfig tcfg;
        tcfg.n_partitions     = 2 * (1 << cfg_.max_depth);
        tcfg.min_samples_leaf = cfg_.min_samples_leaf;
        tcfg.max_depth        = cfg_.max_depth;
        tcfg.n_bins           = cfg_.n_bins;
        tcfg.auto_tune        = false;
        tcfg.random_state     = cfg_.random_state + i;
        tcfg.split_strategy   = hvrt::SplitStrategy::Random;
        tcfg.colsample_bytree = cfg_.colsample_bytree;

        WeakLearner wl;
        wl.cont_cols = cont_cols_full_;
        wl.lr        = cfg_.learning_rate;

        // Inject pre-computed bin edges so build() skips the O(n·d·log n) sort.
        wl.tree.inject_bin_edges(gbt_bin_edges_, cont_cols_full_, cfg_.n_bins);
        wl.tree.build(Xr_build, Xbin_build, cont_cols_full_, {}, grads_build, tcfg);

        // ── Predict stride: defer full-dataset predict ─────────────────────
        // Full-dataset predict (K08) is only needed at refit boundaries for
        // gradient computation on X_cur. Between refits, accumulate on the
        // reduced set only. Every predict_stride rounds, sync preds_on_X.
        const bool do_full_predict = (cfg_.predict_stride <= 1)
            || (i % cfg_.predict_stride == 0)
            || (cfg_.refit_interval > 0 && (i + 1) % cfg_.refit_interval == 0);

        if (do_full_predict) {
            // Full predict on X_cur (reuse pre-allocated buffer)
            lp_X_buf.resize(n);
            wl.tree.predict_binned_into(X_bin_full_, lp_X_buf);
            preds_on_X += cfg_.learning_rate * lp_X_buf;
            if (last_n_expanded == 0) {
                const int nr = static_cast<int>(red_idx.size());
                for (int s = 0; s < nr; ++s)
                    preds[s] += cfg_.learning_rate * lp_X_buf[red_idx[s]];
            } else {
                lp_Xr_buf.resize(Xr.rows());
                wl.tree.predict_into(Xr, lp_Xr_buf);
                preds += cfg_.learning_rate * lp_Xr_buf;
            }
            pending_trees_for_preds_ = 0;
        } else {
            // Deferred: only update reduced-set preds using binned predict
            if (last_n_expanded == 0) {
                lp_Xr_buf.resize(static_cast<int>(red_idx.size()));
                wl.tree.predict_binned_into(X_bin_r, lp_Xr_buf);
                const int nr = static_cast<int>(red_idx.size());
                for (int s = 0; s < nr; ++s)
                    preds[s] += cfg_.learning_rate * lp_Xr_buf[s];
            } else {
                lp_Xr_buf.resize(Xr.rows());
                wl.tree.predict_into(Xr, lp_Xr_buf);
                preds += cfg_.learning_rate * lp_Xr_buf;
            }
            pending_trees_for_preds_++;
        }

        trees_.push_back(std::move(wl));
    }

    // Sync preds_on_X for any deferred trees at end of training
    if (pending_trees_for_preds_ > 0) {
        const int start = static_cast<int>(trees_.size()) - pending_trees_for_preds_;
        Eigen::VectorXd tmp(n);
        for (int t = start; t < static_cast<int>(trees_.size()); ++t) {
            trees_[t].tree.predict_binned_into(X_bin_full_, tmp);
            preds_on_X += trees_[t].lr * tmp;
        }
        pending_trees_for_preds_ = 0;
    }

    fitted_ = true;

    // Persist HVRT geometry for interpretability APIs.
    // train_predictions_ covers the full X_arg so interpretability callers always get
    // predictions for all training samples.
    if (last_hvrt && last_hvrt->fitted()) {
        last_hvrt_     = last_hvrt;
        X_z_           = last_hvrt->X_z();
        partition_ids_ = last_hvrt->partition_ids();
        train_predictions_ = predict_from_trees(X_arg, -1);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Serialization
// ══════════════════════════════════════════════════════════════════════════════

namespace {

class BW {
    std::vector<uint8_t>& b_;
public:
    explicit BW(std::vector<uint8_t>& b) : b_(b) {}
    void i(int v)       { auto p = reinterpret_cast<const uint8_t*>(&v); b_.insert(b_.end(), p, p+4); }
    void d(double v)    { auto p = reinterpret_cast<const uint8_t*>(&v); b_.insert(b_.end(), p, p+8); }
    void bo(bool v)     { b_.push_back(v ? 1 : 0); }
    void s(const std::string& v) { i(static_cast<int>(v.size())); b_.insert(b_.end(), v.begin(), v.end()); }
    void raw(const void* data, size_t n) { auto p = static_cast<const uint8_t*>(data); b_.insert(b_.end(), p, p+n); }
    void vec_d(const std::vector<double>& v) { i(static_cast<int>(v.size())); if(!v.empty()) raw(v.data(), v.size()*8); }
    void vec_i(const std::vector<int>& v)    { i(static_cast<int>(v.size())); if(!v.empty()) raw(v.data(), v.size()*4); }
    void mat(const Eigen::MatrixXd& m) {
        i(static_cast<int>(m.rows())); i(static_cast<int>(m.cols()));
        if(m.size()>0) raw(m.data(), m.size()*8);
    }
    void vd(const Eigen::VectorXd& v) { i(static_cast<int>(v.size())); if(v.size()>0) raw(v.data(), v.size()*8); }
    void vi(const Eigen::VectorXi& v) { i(static_cast<int>(v.size())); if(v.size()>0) raw(v.data(), v.size()*4); }
};

class BR {
    const uint8_t* p_; size_t pos_, sz_;
public:
    BR(const uint8_t* data, size_t size) : p_(data), pos_(0), sz_(size) {}
    int    i()  { int v;    std::memcpy(&v, p_+pos_, 4); pos_+=4; return v; }
    double d()  { double v; std::memcpy(&v, p_+pos_, 8); pos_+=8; return v; }
    bool   bo() { return p_[pos_++] != 0; }
    std::string s() { int n=i(); std::string v(reinterpret_cast<const char*>(p_+pos_), n); pos_+=n; return v; }
    void raw(void* out, size_t n) { std::memcpy(out, p_+pos_, n); pos_+=n; }
    std::vector<double> vec_d() { int n=i(); std::vector<double> v(n); if(n>0) raw(v.data(), n*8); return v; }
    std::vector<int>    vec_i() { int n=i(); std::vector<int> v(n);    if(n>0) raw(v.data(), n*4); return v; }
    Eigen::MatrixXd mat() {
        int r=i(), c=i(); Eigen::MatrixXd m(r,c);
        if(r*c>0) raw(m.data(), static_cast<size_t>(r)*c*8);
        return m;
    }
    Eigen::VectorXd vd() { int n=i(); Eigen::VectorXd v(n); if(n>0) raw(v.data(), n*8); return v; }
    Eigen::VectorXi vi() { int n=i(); Eigen::VectorXi v(n); if(n>0) raw(v.data(), n*4); return v; }
};

} // anon

std::vector<uint8_t> GeoXGBBase::to_bytes() const {
    std::vector<uint8_t> buf;
    buf.reserve(1024 + trees_.size() * 4096);
    BW w(buf);

    w.i(1);  // format version

    // ── Config ───────────────────────────────────────────────────────────────
    w.i(cfg_.n_rounds);         w.d(cfg_.learning_rate);    w.i(cfg_.max_depth);
    w.i(cfg_.min_samples_leaf); w.d(cfg_.reduce_ratio);     w.d(cfg_.expand_ratio);
    w.d(cfg_.y_weight);         w.bo(cfg_.adaptive_y_weight);
    w.i(cfg_.refit_interval);   w.bo(cfg_.auto_noise);      w.bo(cfg_.noise_guard);
    w.d(cfg_.refit_noise_floor);w.bo(cfg_.auto_expand);     w.i(cfg_.min_train_samples);
    w.d(cfg_.bandwidth);        w.bo(cfg_.selective_target); w.i(cfg_.selective_k_pairs);
    w.i(cfg_.d_geom_threshold); w.d(cfg_.residual_correct_lambda);
    w.bo(cfg_.blend_cross_term);w.bo(cfg_.syn_partition_correct);
    w.d(cfg_.y_geom_coupling);  w.i(cfg_.hvrt_min_samples_leaf);
    w.i(cfg_.hvrt_n_partitions);w.i(cfg_.n_bins);
    w.s(cfg_.partitioner);      w.s(cfg_.reduce_method);
    w.s(cfg_.generation_strategy); w.bo(cfg_.adaptive_reduce_ratio);
    w.i(cfg_.sample_block_n);   w.bo(cfg_.leave_last_block_out);
    w.s(cfg_.loss);             w.d(cfg_.convergence_tol);  w.d(cfg_.pos_class_weight);
    w.d(cfg_.e3_target_lambda); w.d(cfg_.lazy_refit_tol);   w.bo(cfg_.fixed_geometry);
    w.bo(cfg_.progressive_expand); w.bo(cfg_.fast_refit);
    w.d(cfg_.colsample_bytree); w.d(cfg_.goss_alpha);       w.d(cfg_.goss_beta);
    w.i(cfg_.predict_stride);   w.d(cfg_.grad_budget_weight);
    w.i(cfg_.random_state);     w.bo(cfg_.variance_weighted);

    // ── Scalars ──────────────────────────────────────────────────────────────
    w.bo(fitted_);              w.i(convergence_round_);
    w.d(last_noise_mod_);       w.d(init_noise_mod_);
    w.i(n_train_arg_);          w.i(n_init_reduced_);
    w.d(init_pred_);

    // ── Traces ───────────────────────────────────────────────────────────────
    w.vec_d(rho_trace_);        w.vec_d(yw_eff_trace_);
    w.vec_d(convergence_losses_);

    // ── Trees ────────────────────────────────────────────────────────────────
    w.i(static_cast<int>(trees_.size()));
    for (const auto& wl : trees_) {
        w.d(wl.lr);
        w.vec_i(wl.cont_cols);
        auto tree_bytes = wl.tree.to_bytes();
        w.i(static_cast<int>(tree_bytes.size()));
        w.raw(tree_bytes.data(), tree_bytes.size());
    }

    // ── Geometry ─────────────────────────────────────────────────────────────
    w.mat(X_z_);
    w.vi(partition_ids_);
    w.vd(train_predictions_);

    return buf;
}

void GeoXGBBase::from_bytes(const std::vector<uint8_t>& data) {
    BR r(data.data(), data.size());

    int ver = r.i();
    (void)ver;

    // ── Config ───────────────────────────────────────────────────────────────
    cfg_.n_rounds           = r.i();  cfg_.learning_rate      = r.d();  cfg_.max_depth          = r.i();
    cfg_.min_samples_leaf   = r.i();  cfg_.reduce_ratio       = r.d();  cfg_.expand_ratio       = r.d();
    cfg_.y_weight           = r.d();  cfg_.adaptive_y_weight  = r.bo();
    cfg_.refit_interval     = r.i();  cfg_.auto_noise         = r.bo(); cfg_.noise_guard        = r.bo();
    cfg_.refit_noise_floor  = r.d();  cfg_.auto_expand        = r.bo(); cfg_.min_train_samples  = r.i();
    cfg_.bandwidth          = r.d();  cfg_.selective_target   = r.bo(); cfg_.selective_k_pairs  = r.i();
    cfg_.d_geom_threshold   = r.i();  cfg_.residual_correct_lambda = r.d();
    cfg_.blend_cross_term   = r.bo(); cfg_.syn_partition_correct   = r.bo();
    cfg_.y_geom_coupling    = r.d();  cfg_.hvrt_min_samples_leaf   = r.i();
    cfg_.hvrt_n_partitions  = r.i();  cfg_.n_bins             = r.i();
    cfg_.partitioner        = r.s();  cfg_.reduce_method      = r.s();
    cfg_.generation_strategy= r.s();  cfg_.adaptive_reduce_ratio   = r.bo();
    cfg_.sample_block_n     = r.i();  cfg_.leave_last_block_out    = r.bo();
    cfg_.loss               = r.s();  cfg_.convergence_tol    = r.d();  cfg_.pos_class_weight   = r.d();
    cfg_.e3_target_lambda   = r.d();  cfg_.lazy_refit_tol     = r.d();  cfg_.fixed_geometry     = r.bo();
    cfg_.progressive_expand = r.bo(); cfg_.fast_refit         = r.bo();
    cfg_.colsample_bytree   = r.d();  cfg_.goss_alpha         = r.d();  cfg_.goss_beta          = r.d();
    cfg_.predict_stride     = r.i();  cfg_.grad_budget_weight = r.d();
    cfg_.random_state       = r.i();  cfg_.variance_weighted  = r.bo();

    // ── Scalars ──────────────────────────────────────────────────────────────
    fitted_            = r.bo(); convergence_round_ = r.i();
    last_noise_mod_    = r.d();  init_noise_mod_    = r.d();
    n_train_arg_       = r.i();  n_init_reduced_    = r.i();
    init_pred_         = r.d();

    // ── Traces ───────────────────────────────────────────────────────────────
    rho_trace_            = r.vec_d();
    yw_eff_trace_         = r.vec_d();
    convergence_losses_   = r.vec_d();

    // ── Trees ────────────────────────────────────────────────────────────────
    int n_trees = r.i();
    trees_.resize(n_trees);
    for (int t = 0; t < n_trees; ++t) {
        trees_[t].lr        = r.d();
        trees_[t].cont_cols = r.vec_i();
        int tree_sz = r.i();
        std::vector<uint8_t> tree_bytes(tree_sz);
        r.raw(tree_bytes.data(), tree_sz);
        trees_[t].tree.from_bytes(tree_bytes);
    }

    // ── Geometry ─────────────────────────────────────────────────────────────
    X_z_               = r.mat();
    partition_ids_     = r.vi();
    train_predictions_ = r.vd();

    // HVRT not restored — to_z()/apply() will throw "No geometry state"
    last_hvrt_ = nullptr;
}

} // namespace geoxgb
