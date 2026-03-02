#pragma once
#include <Eigen/Dense>

namespace geoxgb {

// ── Noise / SNR estimation ────────────────────────────────────────────────────
//
// Port of Python GeoXGB's estimate_noise_modulation():
//
//   local_mean_i = mean(y[k nearest neighbours of i in X_z, excluding self])
//   snr          = Var(local_means) / Var(y)
//   modulation   = clip((snr - 0.15) / 0.45, 0, 1)
//
// Maps: snr > 0.60 → 1.0 (clean), snr < 0.15 → 0.0 (pure noise).
//
// Implementation uses a random subsample of probe points (m = min(500, n))
// to keep cost at O(m * n * d) rather than O(n^2 * d), which is fast
// enough for any n encountered during GeoXGB's refit windows.

double estimate_noise_modulation(
    const Eigen::MatrixXd& X_z,   // whitened features, shape (n, d)
    const Eigen::VectorXd& y,     // gradient signal, shape (n,)
    int   k             = 10,     // neighbours to average
    int   random_state  = 42);

} // namespace geoxgb
