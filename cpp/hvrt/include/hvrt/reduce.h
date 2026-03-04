#pragma once
#include <Eigen/Dense>
#include <vector>
#include <optional>
#include "hvrt/types.h"
#include "hvrt/threadpool.h"

namespace hvrt {

// ── Budget allocation ─────────────────────────────────────────────────────────

// Compute per-partition sample budgets that sum exactly to n_target.
//
// part_ids      : n-vector of partition labels (0-based, dense)
// n_target      : desired total number of selected samples
// min_per_part  : floor for each partition's budget
// var_weighted  : if true, weight by mean |X_z| (variance proxy); else by size
// X_z           : n x d whitened feature matrix (only used when var_weighted=true)
// clamp_to_sizes: if true (default), cap each partition's budget to its size.
//                 Set false for generation, where oversampling is desired.
//
// Returns VectorXi of length max_partition_id+1.
Eigen::VectorXi compute_budgets(
    const Eigen::VectorXi& part_ids,
    int n_target,
    int min_per_part,
    bool var_weighted,
    const Eigen::MatrixXd& X_z,
    bool clamp_to_sizes = true);

// ── Selection strategies ──────────────────────────────────────────────────────

// CentroidFPS: greedy farthest-point sampling seeded from centroid.
// X_part: sub-matrix of rows in partition (local rows).
// Returns local indices (within X_part rows) of selected samples.
std::vector<int> centroid_fps(const Eigen::MatrixXd& X_part, int budget);

// MedoidFPS: FPS seeded from (approximate) medoid.
std::vector<int> medoid_fps(const Eigen::MatrixXd& X_part, int budget);

// VarianceOrdered: select samples with highest local variance (k-NN based).
std::vector<int> variance_ordered(const Eigen::MatrixXd& X_part, int budget, int k_nn = 10);

// Stratified: uniform random selection per partition.
// Uses simple shuffle + take first `budget` (deterministic via rng_seed).
std::vector<int> stratified_select(const Eigen::MatrixXd& X_part, int budget,
                                   uint64_t rng_seed);

// ── Orthant-stratified reduction ─────────────────────────────────────────────
// Groups samples by sign(X_z − median_z) orthant, allocates budget proportional
// to size·MAD(y) per orthant, then selects within each orthant by L1-distance
// from the orthant centroid at linearly-spaced positions (farthest-first order).
// Requires y — call directly from do_resample, not via HVRT::reduce_indices().
std::vector<int> orthant_stratified(
    const Eigen::MatrixXd& X_z,
    const Eigen::VectorXd& y,
    int n_target,
    int random_state);

// ── Top-level reduction ───────────────────────────────────────────────────────

// Reduce X_z to n_target samples using specified method.
// Returns row-indices of selected samples in original X_z.
std::vector<int> reduce(
    const Eigen::MatrixXd& X_z,
    const Eigen::VectorXi& part_ids,
    int n_target,
    ReductionMethod method,
    bool var_weighted,
    std::optional<int> n_parts_override,
    int random_state,
    int n_threads);

// ── Debug timing ─────────────────────────────────────────────────────────────
// Enable per-call timing prints in variance_ordered().
// Not thread-safe — only call from a single thread at startup.
void variance_ordered_enable_timing(bool enable);

} // namespace hvrt
