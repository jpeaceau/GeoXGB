#include "hvrt/whitener.h"
#include <stdexcept>

namespace hvrt {

void Whitener::fit(const Eigen::MatrixXd& X, const std::vector<bool>& cat_mask) {
    const int d = static_cast<int>(X.cols());
    if (static_cast<int>(cat_mask.size()) != d) {
        throw std::invalid_argument("cat_mask size must match X columns");
    }

    cat_mask_ = cat_mask;
    means_    = X.colwise().mean();

    // std = sqrt(mean((X - mean)^2))  — population std (ddof=0), matching sklearn default
    Eigen::MatrixXd centered = X.rowwise() - means_.transpose();
    stds_ = (centered.array().square().colwise().mean()).sqrt();

    // Guard against zero-std (constant features)
    for (int j = 0; j < d; ++j) {
        if (stds_[j] < kEps) stds_[j] = 1.0;
    }

    fitted_ = true;
}

Eigen::MatrixXd Whitener::transform(const Eigen::MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("Whitener not fitted");
    Eigen::MatrixXd X_z = (X.rowwise() - means_.transpose());
    X_z = X_z.array().rowwise() / stds_.transpose().array();
    return X_z;
}

Eigen::MatrixXd Whitener::inverse_transform(const Eigen::MatrixXd& X_z) const {
    if (!fitted_) throw std::runtime_error("Whitener not fitted");
    Eigen::MatrixXd X = X_z.array().rowwise() * stds_.transpose().array();
    X = X.rowwise() + means_.transpose();
    return X;
}

} // namespace hvrt
