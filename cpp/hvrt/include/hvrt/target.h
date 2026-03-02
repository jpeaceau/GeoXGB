#pragma once
#include <Eigen/Dense>

namespace hvrt {

// Compute synthetic target vectors from whitened feature matrix X_z (n x d).
// All functions return column vectors of length n.

// Pairwise interaction target: O(n·d²).
// For each pair (i,j), i<j: compute element-wise products, z-score them,
// accumulate into output.  Matches Python _pairwise_target_numpy.
Eigen::VectorXd compute_pairwise_target(const Eigen::MatrixXd& X_z);

// Row z-sum target: O(n·d).
// Computes z-score of each row sum.
Eigen::VectorXd compute_sum_target(const Eigen::MatrixXd& X_z);

// Blend X-derived composite target with external y.
// y_component = zscore(|y_norm - median(y_norm)|)
// result = zscore(x_comp) blended with y_comp via y_weight.
// use_cross: also add a zscore(x_z * y_comp) interaction term scaled by y_weight.
//   The cross term is high where geometric cooperation AND y-extremality co-occur,
//   encouraging partitions that separate both structure and y simultaneously.
Eigen::VectorXd blend_target(const Eigen::VectorXd& x_comp,
                              const Eigen::VectorXd& y,
                              double y_weight,
                              bool use_cross = false);

// Residual-guided selective pairwise target.
// Computes |Pearson(zscore(z_a*z_b), zscore(resid))| for every feature pair (a,b),
// selects the top k_pairs, and accumulates their z-scored products.
// k_pairs <= 0 means auto: max(5, d*(d-1)/4).
// Falls back to full pairwise target when k_pairs >= d*(d-1)/2.
// Used by GeoXGB's Approach 1: at each HVRT refit, the cached geom_target is
// replaced by a target that is biased toward pairs actually correlated with
// the current residuals.
Eigen::VectorXd compute_selective_target(const Eigen::MatrixXd& X_z,
                                          const Eigen::VectorXd& resid,
                                          int k_pairs = -1);

// Internal helper: standardize a vector to zero-mean unit-variance (population).
Eigen::VectorXd zscore(const Eigen::VectorXd& v);

} // namespace hvrt
