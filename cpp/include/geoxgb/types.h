#pragma once
#include <string>

namespace geoxgb {

// ── GeoXGB configuration ──────────────────────────────────────────────────────
// Only parameters that GeoXGB actually uses are exposed.  Everything
// HVRT-internal (generation_strategy variants, copula, multivariate KDE,
// feature_weights, assignment_strategy details) is hardcoded to the
// GeoXGB-optimal defaults.

struct GeoXGBConfig {
    // Boosting
    int    n_rounds           = 1000;
    double learning_rate      = 0.2;
    int    max_depth          = 3;
    int    min_samples_leaf   = 5;   // for GBT weak learner trees

    // HVRT resampling
    double reduce_ratio       = 0.7;
    double expand_ratio       = 0.0;
    double y_weight           = 0.2;   // upper bound when adaptive_y_weight=true
    bool   adaptive_y_weight  = true;  // scale y_weight by |ρ(geom, residuals)| each refit
    int    refit_interval     = 20;   // 0 = disable
    bool   auto_noise         = true;
    bool   noise_guard        = true;
    double refit_noise_floor  = 0.05;
    bool   auto_expand        = true;
    int    min_train_samples  = 5000;
    double bandwidth          = -1.0; // -1 = auto (Scott's rule per partition)

    // Approach 1: residual-guided selective pairwise target.
    // At each HVRT refit, replaces the static pairwise cooperation target with
    // the top-k feature-pair products ranked by |Pearson(pair_product, residuals)|.
    // Only pairs that actually predict the gradient direction are used.
    // selective_k_pairs <= 0 → auto: max(5, d*(d-1)/4).
    bool selective_target   = false;
    int  selective_k_pairs  = -1;

    // Approach 2: skip HVRT resampling entirely for low-d datasets.
    // When d <= d_geom_threshold (and threshold > 0), fit_boosting runs as a
    // pure GBT on the full training set without any HVRT reduce/expand/refit.
    // On low-d data (d ≤ 8–12), the pairwise cooperation target has too few
    // pairs to align with the gradient direction — HVRT sampling only discards
    // informative points.  Threshold 0 = disabled (HVRT always active).
    int  d_geom_threshold   = 0;

    // Approach 3: per-partition z-space sub-split residual correction.
    // At each HVRT refit, for each partition p, finds the best variance-reducing
    // binary split of the current residuals in z-space (using all n training
    // samples in p).  Applies the shrunk per-child mean correction to the
    // synthetic y targets:
    //   delta_child_shrunk = mean(resid[child]) * n_child / (n_child + lambda)
    // At lambda → ∞ the correction → 0 (safe default).  lambda = 50 showed
    // break-even in dual_strategy.py; good starting range: [25, 200].
    // 0.0 = disabled.
    double residual_correct_lambda = 0.0;

    // Y-coupling strategies (all off by default; enable one at a time)
    // S1: add x_z*y_comp interaction term to blend_target splitting criterion
    bool   blend_cross_term       = false;
    // S2: after knn_assign_y, shift each synthetic y by the within-partition
    //     mean difference (real mean − kNN mean) to remove interpolation bias
    bool   syn_partition_correct  = false;
    // S3: at refit, pass y_coupled = (1-α)*y_std + α*geom to HVRT instead of
    //     raw residuals; α=y_geom_coupling (0 = off, disabled at first fit)
    double y_geom_coupling        = 0.0;

    // HVRT partition tree sizing (both -1 = HVRT auto-tune)
    int    hvrt_min_samples_leaf = -1;
    int    hvrt_n_partitions     = -1;

    // GBT tree binning (separate from HVRT partition tree binning)
    int    n_bins             = 64;

    // Geometry / resampling strategy dispatch
    // "hvrt"|"hart"|"fasthvrt"|"fasthart"|"pyramid_hart"
    std::string partitioner           = "hvrt";
    // "variance"|"orthant_stratified"|"centroid_fps"|"medoid_fps"|"stratified"
    std::string reduce_method         = "variance";
    // "epanechnikov"|"simplex_mixup"|"laplace"|"multivariate_kde"|"copula"|"bootstrap"
    std::string generation_strategy   = "epanechnikov";
    // Adaptive reduce ratio: increase reduce_ratio for heavy-tailed gradient distributions
    bool        adaptive_reduce_ratio = false;

    // Scalability: epoch-based block cycling.
    // -1 = disabled.  > 0 → divide full training set into non-overlapping blocks
    // of sample_block_n rows (epoch-permuted, deterministic via random_state).
    // At each refit_interval, advance to the next block; all downstream costs
    // (HVRT fit, GBT training, predict-on-X tracking) scale with sample_block_n
    // rather than full n.  Recommended for n > 20 000: 5 000–20 000.
    // Next block is pre-fetched asynchronously while the current block trains.
    int    sample_block_n        = -1;
    // When true, the last block of each epoch is held out (never trained on).
    // Provides an implicit held-out validation set for monitoring convergence.
    bool   leave_last_block_out  = false;

    // Loss / convergence / class weighting
    std::string loss             = "squared_error"; // "squared_error" | "absolute_error"
    double      convergence_tol  = 0.0;             // 0.0 = disabled
    double      pos_class_weight = 1.0;             // binary classifier positive-class scale

    // Blended e₃ target: when > 0, the HVRT geometry target is blended with
    // the third elementary symmetric polynomial e₃ (degree-3 interactions).
    // target = zscore(T + e3_lambda * e₃).  e₃ is noise-invariant and captures
    // triple-feature interactions that T (degree-2) is blind to.
    // 0.0 = disabled (default).  Recommended range: 1.0–3.0.
    double e3_target_lambda   = 0.0;

    // ── Scalability experiments (Tests A/B/C) ──────────────────────────────
    // Test A: Lazy refit — skip do_resample when mean |gradient| hasn't
    // changed significantly since last refit.  Threshold = relative change.
    // 0.0 = disabled (always refit).
    double lazy_refit_tol     = 0.0;

    // Test B: Fixed geometry — fit HVRT once on round 0, then at each refit
    // only re-reduce using the existing HVRT (no refit of the partition tree).
    // Reduce/expand still happen, but the whitener + partition tree are frozen.
    bool   fixed_geometry     = false;

    // Test C: Progressive expand — scale expand_ratio linearly from 0 to
    // expand_ratio over the course of training: eff_er = er * (round / n_rounds).
    // false = constant expand_ratio (default).
    bool   progressive_expand = false;

    // ── Determinism & sample management ─────────────────────────────────
    // Sample-without-replacement: each sample is used in at most one refit
    // window before all samples have been cycled (epoch).  At each refit,
    // only unused samples are candidates for reduction.  HVRT still
    // partitions the full dataset for geometry — only the reduce step
    // filters by usage.  When fewer than n_keep unused samples remain,
    // the usage mask resets (new epoch).
    bool   sample_without_replacement = false;

    // ── Performance optimisations ──────────────────────────────────────────
    // Feature subsampling: fraction of features used per tree (1.0 = all).
    // Deterministic round-robin rotation (no RNG) when < 1.0.
    double colsample_bytree   = 1.0;

    // Predict stride: run full-dataset predict every N rounds instead of
    // every round.  Between strides, accumulate on reduced set only.
    // 1 = every round (default, current behaviour).
    int    predict_stride     = 1;

    // Gradient-aware budget allocation (deterministic GOSS via HVRT).
    // At each refit, partition reduction budgets are weighted by gradient mass:
    //   budget_p ∝ (1−α)·(n_p/n) + α·(Σ_{i∈p}|g_i| / Σ|g|)
    // α = grad_budget_weight.  High-gradient partitions (near decision boundary)
    // get more of the reduction budget; low-gradient partitions (well-predicted)
    // are reduced more aggressively.  Fully deterministic — no random sampling.
    // 0.0 = disabled (standard size-proportional budgets).  Range [0, 1].
    double grad_budget_weight = 0.0;

    // Misc
    int    random_state       = 42;
    bool   variance_weighted  = true;
};

} // namespace geoxgb
