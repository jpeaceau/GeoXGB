#pragma once
#include <Eigen/Dense>
#include <vector>

namespace hvrt {

// StandardScaler-equivalent: mean-center + unit variance.
// Handles both continuous and categorical columns identically
// (same algorithm, separate cat_mask tracking).
class Whitener {
public:
    // fit: compute per-feature mean + std from X (n x d).
    // cat_mask[j] == true  →  column j is categorical.
    void fit(const Eigen::MatrixXd& X, const std::vector<bool>& cat_mask);

    // transform: X → X_z  (in-place copy; X unchanged)
    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;

    // inverse_transform: X_z → X_orig
    Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& X_z) const;

    // Accessors
    const Eigen::VectorXd& means()    const { return means_; }
    const Eigen::VectorXd& stds()     const { return stds_; }
    const std::vector<bool>& cat_mask() const { return cat_mask_; }

    bool fitted() const { return fitted_; }

private:
    Eigen::VectorXd means_;
    Eigen::VectorXd stds_;
    std::vector<bool> cat_mask_;
    bool fitted_ = false;

    static constexpr double kEps = 1e-8;
};

} // namespace hvrt
