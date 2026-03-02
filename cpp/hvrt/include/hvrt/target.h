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
// result = zscore(x_comp * (1-y_weight) + y_comp * y_weight)   [conceptually]
// Actually: result = zscore(x_comp) blended with y_comp via y_weight.
Eigen::VectorXd blend_target(const Eigen::VectorXd& x_comp,
                              const Eigen::VectorXd& y,
                              double y_weight);

// Internal helper: standardize a vector to zero-mean unit-variance (population).
Eigen::VectorXd zscore(const Eigen::VectorXd& v);

} // namespace hvrt
