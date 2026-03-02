#include "hvrt/target.h"
#include <algorithm>
#include <cmath>

namespace hvrt {

static constexpr double kEps = 1e-8;

Eigen::VectorXd zscore(const Eigen::VectorXd& v) {
    double mu  = v.mean();
    double sig = std::sqrt((v.array() - mu).square().mean());
    if (sig < kEps) sig = 1.0;
    return (v.array() - mu) / sig;
}

// Compute median of a vector (modifies a copy).
static double median_val(Eigen::VectorXd v) {
    int n = static_cast<int>(v.size());
    std::nth_element(v.data(), v.data() + n / 2, v.data() + n);
    if (n % 2 == 1) return v[n / 2];
    // even: average two middle elements
    double upper = v[n / 2];
    std::nth_element(v.data(), v.data() + n / 2 - 1, v.data() + n);
    return 0.5 * (v[n / 2 - 1] + upper);
}

Eigen::VectorXd compute_pairwise_target(const Eigen::MatrixXd& X_z) {
    const int n = static_cast<int>(X_z.rows());
    const int d = static_cast<int>(X_z.cols());

    Eigen::VectorXd result = Eigen::VectorXd::Zero(n);

    // Outer loop over feature pairs (i, j), i < j
    // For each pair: product col, zscore it, accumulate
    for (int i = 0; i < d - 1; ++i) {
        // Vectorised: compute products for all j > i at once
        // X_z[:,i+1:] .colwise() * X_z[:,i]
        Eigen::MatrixXd prods = X_z.rightCols(d - i - 1).array().colwise()
                                * X_z.col(i).array();
        // zscore each product column and accumulate
        for (int k = 0; k < prods.cols(); ++k) {
            result += zscore(prods.col(k));
        }
    }

    return zscore(result);
}

Eigen::VectorXd compute_sum_target(const Eigen::MatrixXd& X_z) {
    return zscore(X_z.rowwise().sum());
}

Eigen::VectorXd blend_target(const Eigen::VectorXd& x_comp,
                              const Eigen::VectorXd& y,
                              double y_weight,
                              bool use_cross) {
    // Normalise y to [0,1] range, then compute deviation from median
    double y_min = y.minCoeff();
    double y_max = y.maxCoeff();
    double y_range = y_max - y_min;
    Eigen::VectorXd y_norm;
    if (y_range < kEps) {
        y_norm = Eigen::VectorXd::Zero(y.size());
    } else {
        y_norm = (y.array() - y_min) / y_range;
    }

    double med = median_val(y_norm);
    Eigen::VectorXd y_dev = (y_norm.array() - med).abs();
    Eigen::VectorXd y_comp = zscore(y_dev);

    Eigen::VectorXd x_z = zscore(x_comp);
    Eigen::VectorXd blended = (1.0 - y_weight) * x_z + y_weight * y_comp;

    if (use_cross) {
        // Cross term: zscore(x_z ∘ y_comp) is high where both geometric
        // cooperation AND y-extremality co-occur in the same sample.
        // This biases the partition tree towards regions that exhibit both
        // structural and predictive alignment simultaneously.
        Eigen::VectorXd cross = zscore((x_z.array() * y_comp.array()).matrix());
        blended += y_weight * cross;
    }

    return zscore(blended);
}

} // namespace hvrt
