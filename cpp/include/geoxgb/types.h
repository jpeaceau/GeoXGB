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

    // Misc
    int    random_state       = 42;
    bool   variance_weighted  = true;
};

} // namespace geoxgb
