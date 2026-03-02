#include "geoxgb/noise.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace geoxgb {

double estimate_noise_modulation(
    const Eigen::MatrixXd& X_z,
    const Eigen::VectorXd& y,
    int   k,
    int   random_state)
{
    const int n = static_cast<int>(X_z.rows());
    const int d = static_cast<int>(X_z.cols());

    double y_var = (y.array() - y.mean()).square().mean();
    // Near-zero variance means gradients are converged — no signal left to learn from.
    // Return 0.0 so the caller treats this as "all noise" and discards the refit.
    if (y_var < 1e-12) return 0.0;

    // Clamp k to valid range
    k = std::min(k, std::max(3, n / 30));
    k = std::min(k, n - 1);
    if (k <= 0) return 1.0;

    // Subsample probe points to keep cost O(m * n * d)
    const int m = std::min(500, n);

    std::vector<int> probe_idx(n);
    std::iota(probe_idx.begin(), probe_idx.end(), 0);
    if (m < n) {
        std::mt19937 rng(static_cast<uint32_t>(random_state));
        std::shuffle(probe_idx.begin(), probe_idx.end(), rng);
        probe_idx.resize(m);
    }

    // Compute local means via BLAS pairwise distance matrix.
    //
    // D[i,j] = ||X_probe[i] - X_z[j]||²
    //        = ||X_probe[i]||² - 2·X_probe[i]·X_z[j]ᵀ + ||X_z[j]||²
    //
    // The cross term is a single BLAS GEMM (m×d)×(d×n) = (m×n), replacing
    // m per-probe O(n·d) scalar loops with one vectorised kernel.
    // Also eliminates m per-probe heap allocations for sq_dists and idx_buf.

    // 1. Extract probe rows into a contiguous (m × d) matrix.
    Eigen::MatrixXd X_probe(m, d);
    for (int pi = 0; pi < m; ++pi)
        X_probe.row(pi) = X_z.row(probe_idx[pi]);

    // 2. Squared norms — cheap: O(m·d) + O(n·d).
    const Eigen::VectorXd norm_probe = X_probe.rowwise().squaredNorm();  // (m,)
    const Eigen::VectorXd norm_all   = X_z.rowwise().squaredNorm();      // (n,)

    // 3. GEMM: D = -2 · X_probe · X_zᵀ,  shape (m × n), stored row-major so
    //    that D.row(pi) is contiguous — critical for cache-friendly partial sort.
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> D(m, n);
    D.noalias() = X_probe * X_z.transpose();
    D *= -2.0;
    D.colwise() += norm_probe;            // D[i,j] += ||X_probe[i]||²
    D.rowwise() += norm_all.transpose();  // D[i,j] += ||X_z[j]||²
    D = D.cwiseMax(0.0);                  // clamp fp rounding negatives

    // 4. For each probe, partial-sort D row to find k nearest; accumulate means.
    Eigen::VectorXd local_means(m);
    std::vector<int> idx_buf(n);          // single allocation, reused across probes
    for (int pi = 0; pi < m; ++pi) {
        int qi = probe_idx[pi];
        std::iota(idx_buf.begin(), idx_buf.end(), 0);
        std::partial_sort(idx_buf.begin(), idx_buf.begin() + k + 1, idx_buf.end(),
                          [&](int a, int b){ return D(pi, a) < D(pi, b); });

        double sum = 0.0;
        int    cnt = 0;
        for (int s = 0; s <= k && cnt < k; ++s) {
            if (idx_buf[s] != qi) {
                sum += y[idx_buf[s]];
                ++cnt;
            }
        }
        local_means[pi] = (cnt > 0) ? sum / cnt : y[qi];
    }

    double lm_var = (local_means.array() - local_means.mean()).square().mean();
    double snr    = lm_var / y_var;

    return static_cast<double>(
        std::max(0.0, std::min(1.0, (snr - 0.15) / 0.45)));
}

} // namespace geoxgb
