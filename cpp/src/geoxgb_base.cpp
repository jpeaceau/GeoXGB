#include "geoxgb/geoxgb_base.h"
#include "geoxgb/noise.h"
#include "hvrt/types.h"
#include "hvrt/hvrt.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>

namespace geoxgb {

// ── Constructor ────────────────────────────────────────────────────────────────

GeoXGBBase::GeoXGBBase(GeoXGBConfig cfg) : cfg_(std::move(cfg)) {}

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

// ── HVRT resampling ───────────────────────────────────────────────────────────

GeoXGBBase::ResampleResult GeoXGBBase::do_resample(
        const Eigen::MatrixXd& X_full,
        const Eigen::VectorXd& y_signal,
        int seed_offset,
        std::shared_ptr<hvrt::HVRT>& hvrt_out)
{
    const int n = static_cast<int>(X_full.rows());

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

    // Bandwidth: auto → HVRT handles it; numeric → pass as string
    if (cfg_.bandwidth > 0.0) {
        hcfg.bandwidth = std::to_string(cfg_.bandwidth);
    } else {
        hcfg.bandwidth = "auto";
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
        h->refit(y_signal, yw_eff);
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

    // Reduce: variance_ordered FPS
    std::vector<int> red_idx_vec = h->reduce_indices(n_keep, std::nullopt,
                                                      "variance", cfg_.variance_weighted);
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

    // Auto-expand: fill up to min_train_samples using Epanechnikov KDE
    bool expansion_risk = cfg_.auto_expand && (n < cfg_.min_train_samples);
    if (expansion_risk && cfg_.expand_ratio == 0.0 && n_red < cfg_.min_train_samples) {
        int _eff_min = std::min(cfg_.min_train_samples,
                                std::max(n * 5, 1000));
        double eff_noise = std::max(noise_mod, 0.1);
        int n_expand = std::max(0, static_cast<int>((_eff_min - n_red) * eff_noise));

        if (n_expand > 0) {
            Eigen::MatrixXd X_syn = h->expand(n_expand, cfg_.variance_weighted,
                                               std::nullopt, "epanechnikov");
            // Actual generated count may differ from requested (e.g. tiny partitions).
            n_expand = static_cast<int>(X_syn.rows());

            if (n_expand > 0) {
            // y-assignment: k-NN in z-space
            Eigen::MatrixXd X_red_z(n_red, h->X_z().cols());
            for (int i = 0; i < n_red; ++i)
                X_red_z.row(i) = h->X_z().row(red_idx[i]);
            Eigen::MatrixXd X_syn_z = h->to_z(X_syn);

            Eigen::VectorXd y_syn = knn_assign_y(X_syn_z, X_red_z, y_red);

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
                                               std::nullopt, "epanechnikov");
            n_expand = static_cast<int>(X_syn.rows()); // actual generated count

            if (n_expand > 0) {
            Eigen::MatrixXd X_red_z(n_red, h->X_z().cols());
            for (int i = 0; i < n_red; ++i)
                X_red_z.row(i) = h->X_z().row(red_idx[i]);
            Eigen::MatrixXd X_syn_z = h->to_z(X_syn);
            Eigen::VectorXd y_syn = knn_assign_y(X_syn_z, X_red_z, y_red);

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
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y)
{
    const int n = static_cast<int>(X.rows());
    const int d = static_cast<int>(X.cols());

    trees_.clear();
    trees_.reserve(cfg_.n_rounds);
    convergence_round_ = -1;
    rho_trace_.clear();
    yw_eff_trace_.clear();

    // ── Bin full X once for GBT weak learner trees ────────────────────────────
    // GBT trees work on the original feature space (no whitening).
    // We bin X at the start and subset X_bin_full_[red_idx] for each Xr.
    cont_cols_full_.resize(d);
    std::iota(cont_cols_full_.begin(), cont_cols_full_.end(), 0);

    full_binner_.fit(X, cfg_.n_bins);
    X_bin_full_    = full_binner_.transform(X);
    // Cache bin edges once; injected into every GBT weak learner to skip
    // the per-round O(n·d·log n) re-sort inside PartitionTree::build().
    // Using the same edges that produced X_bin_full_ also ensures consistency:
    // split thresholds are aligned with the pre-computed bin assignments.
    gbt_bin_edges_ = full_binner_.edges();

    // ── Initial resample ──────────────────────────────────────────────────────
    std::shared_ptr<hvrt::HVRT> last_hvrt;
    ResampleResult res = do_resample(X, y, 0, last_hvrt);

    Eigen::MatrixXd Xr      = res.X;
    Eigen::VectorXd yr      = res.y;
    Eigen::VectorXi red_idx = res.red_idx;
    double last_noise_mod   = res.noise_mod;
    last_noise_mod_         = last_noise_mod;
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
    for (int i = 0; i < cfg_.n_rounds; ++i) {

        // ── Refit ─────────────────────────────────────────────────────────────
        if (cfg_.refit_interval > 0 && i > 0 && (i % cfg_.refit_interval == 0)) {
            Eigen::VectorXd grads_orig = gradients(y, preds_on_X);

            bool skip_refit = cfg_.noise_guard && cfg_.auto_noise
                              && expansion_risk
                              && (last_noise_mod < cfg_.refit_noise_floor);

            if (!skip_refit) {
                // Refit: pass last_hvrt so do_resample() can reuse cached whitener/binner/
                // geometry-target via HVRT::refit(), saving O(n·d²) + O(n²·d) per refit.
                ResampleResult new_res = do_resample(X, grads_orig, i, last_hvrt);
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
        Eigen::VectorXd lp_X  = wl.tree.predict(X);
        preds       += cfg_.learning_rate * lp_Xr;
        preds_on_X  += cfg_.learning_rate * lp_X;

        trees_.push_back(std::move(wl));
    }

    fitted_ = true;
}

} // namespace geoxgb
