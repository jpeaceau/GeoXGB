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

    // Misc
    int    random_state       = 42;
    bool   variance_weighted  = true;
};

} // namespace geoxgb
