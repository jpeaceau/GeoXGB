#include "hvrt/target.h"
#include <algorithm>
#include <cmath>
#include <vector>

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

Eigen::VectorXd compute_hart_target(const Eigen::MatrixXd& X_z) {
    // T = 0.5*(||z||_1² − ||z||_2²) = Σ_{i<j} |z_i|·|z_j|  (O(n·d))
    Eigen::VectorXd l1   = X_z.cwiseAbs().rowwise().sum();
    Eigen::VectorXd l2sq = X_z.rowwise().squaredNorm();
    return zscore(0.5 * (l1.array().square() - l2sq.array()).matrix());
}

Eigen::VectorXd compute_pyramid_target(const Eigen::MatrixXd& X_z) {
    // A = |Σ z_i| − ||z||_1  (always ≤ 0; exactly 0 on coordinate hyperplanes)
    // Degree-1 homogeneous → outlier cancellation: one 50σ spike → A shifts minimally
    Eigen::VectorXd S  = X_z.rowwise().sum();
    Eigen::VectorXd l1 = X_z.cwiseAbs().rowwise().sum();
    return zscore((S.array().abs() - l1.array()).matrix());
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

Eigen::VectorXd compute_selective_target(const Eigen::MatrixXd& X_z,
                                          const Eigen::VectorXd& resid,
                                          int k_pairs) {
    const int n     = static_cast<int>(X_z.rows());
    const int d     = static_cast<int>(X_z.cols());
    const int n_all = d * (d - 1) / 2;

    // Edge case: no pairs or too few samples → return residual zscore as proxy
    if (n_all <= 0 || n < 3) return zscore(resid);

    if (k_pairs <= 0) k_pairs = std::max(5, n_all / 4);
    k_pairs = std::min(k_pairs, n_all);

    // If all pairs are selected, fall through to full pairwise target.
    if (k_pairs >= n_all) return compute_pairwise_target(X_z);

    // Compute |Pearson(zscore(z_a*z_b), zscore(resid))| for every pair.
    // Using zscored reference avoids recomputing σ_resid per pair.
    // Pearson = (1/(n-1)) * zscore(pair).dot(zscore(resid))
    Eigen::VectorXd resid_z   = zscore(resid);
    const double    inv_n1    = 1.0 / static_cast<double>(n - 1);

    struct PairScore { int a, b; double score; };
    std::vector<PairScore> scores;
    scores.reserve(n_all);

    for (int a = 0; a < d - 1; ++a) {
        for (int b = a + 1; b < d; ++b) {
            Eigen::VectorXd prod = (X_z.col(a).array() * X_z.col(b).array()).matrix();
            // Pearson correlation of zscore(pair product) with zscore(resid)
            double r = zscore(prod).dot(resid_z) * inv_n1;
            r = std::max(-1.0, std::min(1.0, r));
            scores.push_back({a, b, std::abs(r)});
        }
    }

    // Partial sort: k_pairs highest |r| pairs move to front
    std::partial_sort(scores.begin(), scores.begin() + k_pairs, scores.end(),
                      [](const PairScore& x, const PairScore& y) {
                          return x.score > y.score;
                      });

    // Accumulate z-scored products of the top-k pairs
    Eigen::VectorXd result = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < k_pairs; ++i) {
        Eigen::VectorXd prod = (X_z.col(scores[i].a).array() *
                                X_z.col(scores[i].b).array()).matrix();
        result += zscore(prod);
    }

    return zscore(result);
}

} // namespace hvrt
