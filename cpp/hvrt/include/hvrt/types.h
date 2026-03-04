#pragma once
#include <string>
#include <vector>

namespace hvrt {

// ── Enums ────────────────────────────────────────────────────────────────────

enum class PartitionerType {
    HVRT,        // pairwise cooperation: Σ z_i·z_j
    HART,        // absolute pairwise cooperation: 0.5*(||z||_1² − ||z||_2²)
    FastHART,    // row-sum target (O(n·d)); FastHART reuses compute_sum_target
    PyramidHART  // axis-aligned level sets: |Σz_i| − ||z||_1 ≤ 0
};

enum class SplitStrategy {
    Best,
    Random
};

enum class ReductionMethod {
    CentroidFPS,
    MedoidFPS,
    VarianceOrdered,
    Stratified,
    OrthantStratified  // y-MAD weighted per orthant; needs y — call as free function
};

enum class GenerationStrategy {
    Epanechnikov,
    MultivariateKDE,
    BootstrapNoise,
    UnivariateCopula,
    Auto,
    SimplexMixup,  // convex combination of two random partition rows
    Laplace        // per-feature Laplace KDE centered on partition centroid
};

// ── Configuration ─────────────────────────────────────────────────────────────

struct HVRTConfig {
    int    n_partitions    = 50;
    int    min_samples_leaf = 20;
    int    max_depth       = 10;
    int    n_bins          = 32;
    int    n_threads       = 4;
    int    random_state    = 42;
    float  y_weight        = 0.0f;
    bool   blend_cross_term = false; // add x_z*y_comp interaction to blend_target
    bool   auto_tune       = true;
    std::string bandwidth  = "auto";  // "auto", "scott", or numeric string
    SplitStrategy    split_strategy  = SplitStrategy::Best;
    PartitionerType  partitioner_type = PartitionerType::HVRT;
    GenerationStrategy gen_strategy  = GenerationStrategy::Epanechnikov;
};

// ── Output structs ────────────────────────────────────────────────────────────

struct PartitionInfo {
    int    id;
    int    size;
    double mean_abs_z;
    double variance;
};

struct ParamRecommendation {
    int n_partitions;
    int min_samples_leaf;
};

} // namespace hvrt
