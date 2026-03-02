#include "hvrt/binner.h"
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
#include <cmath>

namespace hvrt {

std::vector<bool> Binner::fit(const Eigen::MatrixXd& X_z_cont, int n_bins) {
    if (n_bins < 2) throw std::invalid_argument("n_bins must be >= 2");
    n_bins_    = n_bins;
    const int n = static_cast<int>(X_z_cont.rows());
    const int d = static_cast<int>(X_z_cont.cols());

    edges_.resize(d);
    binary_mask_.assign(d, false);

    for (int f = 0; f < d; ++f) {
        // Count unique values
        std::vector<double> col(n);
        for (int i = 0; i < n; ++i) col[i] = X_z_cont(i, f);
        std::sort(col.begin(), col.end());
        int n_unique = static_cast<int>(
            std::unique(col.begin(), col.end()) - col.begin());

        if (n_unique <= 2) {
            binary_mask_[f] = true;
            // Binary: edges are just min and max
            edges_[f].resize(2);
            edges_[f][0] = col[0];
            edges_[f][1] = col[n_unique - 1];
            continue;
        }

        int b = std::min(n_bins, n_unique);
        edges_[f].resize(b + 1);

        // Compute b+1 quantile positions
        // Use std::nth_element for O(n) per quantile
        std::vector<double> work(X_z_cont.col(f).data(),
                                 X_z_cont.col(f).data() + n);

        for (int q = 0; q <= b; ++q) {
            // quantile position: q / b of [0, n-1]
            int pos = static_cast<int>(std::round(
                static_cast<double>(q) / b * (n - 1)));
            pos = std::clamp(pos, 0, n - 1);
            std::nth_element(work.begin(), work.begin() + pos, work.end());
            edges_[f][q] = work[pos];
        }

        // Ensure strict monotonicity (collapse duplicates)
        // Keep first and last, deduplicate intermediate
        std::vector<double> uniq_edges;
        uniq_edges.push_back(edges_[f][0]);
        for (int q = 1; q <= b; ++q) {
            if (edges_[f][q] > uniq_edges.back() + 1e-15) {
                uniq_edges.push_back(edges_[f][q]);
            }
        }
        // Always ensure at least 2 edges
        if (uniq_edges.size() < 2) {
            uniq_edges = {col[0], col[n_unique - 1]};
        }
        edges_[f] = Eigen::Map<Eigen::VectorXd>(
            uniq_edges.data(), static_cast<int>(uniq_edges.size()));
    }

    fitted_ = true;
    return binary_mask_;
}

Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
Binner::transform(const Eigen::MatrixXd& X_z_cont) const {
    if (!fitted_) throw std::runtime_error("Binner not fitted");
    const int n = static_cast<int>(X_z_cont.rows());
    const int d = static_cast<int>(X_z_cont.cols());

    using BinMat = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    BinMat out(n, d);

    for (int f = 0; f < d; ++f) {
        const Eigen::VectorXd& e = edges_[f];
        int nb = static_cast<int>(e.size()) - 1; // number of bins

        if (binary_mask_[f]) {
            // binary: 0 if value <= first edge, else 1
            double thresh = (e[0] + e[1]) * 0.5;
            for (int i = 0; i < n; ++i) {
                out(i, f) = (X_z_cont(i, f) > thresh) ? 1u : 0u;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                double v = X_z_cont(i, f);
                // upper_bound gives first edge > v → bin = that index - 1
                auto it = std::upper_bound(e.data(), e.data() + e.size(), v);
                int bin = static_cast<int>(it - e.data()) - 1;
                bin = std::clamp(bin, 0, nb - 1);
                out(i, f) = static_cast<uint8_t>(bin);
            }
        }
    }
    return out;
}

double Binner::get_threshold(int feat, int bin) const {
    if (!fitted_) throw std::runtime_error("Binner not fitted");
    const Eigen::VectorXd& e = edges_[feat];
    int nb = static_cast<int>(e.size()) - 1;
    if (bin < 0 || bin >= nb) throw std::out_of_range("bin out of range");
    return 0.5 * (e[bin] + e[bin + 1]);
}

} // namespace hvrt
