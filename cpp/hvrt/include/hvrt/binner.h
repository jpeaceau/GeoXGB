#pragma once
#include <Eigen/Dense>
#include <vector>

namespace hvrt {

// Quantile discretisation for continuous features.
// Produces uint8 binned matrix suitable for histogram sweeps in the tree.
class Binner {
public:
    // fit: compute bin edges per column of X_z_cont (n x d_cont).
    // n_bins: maximum number of bins (actual may be lower for low-cardinality features).
    // Returns a boolean mask of length d_cont:
    //   true  → feature has ≤ 2 unique values (binary stream, no binning)
    //   false → feature gets n_bins binning
    std::vector<bool> fit(const Eigen::MatrixXd& X_z_cont, int n_bins);

    // transform: map X_z_cont → binned matrix (n x d_cont), values in [0, n_bins-1].
    // Columns flagged as binary (from fit) are simply thresholded at 0.
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    transform(const Eigen::MatrixXd& X_z_cont) const;

    // get_threshold: midpoint between bin edges for feature f, bin b.
    double get_threshold(int feat, int bin) const;

    int n_bins()      const { return n_bins_; }
    int n_features()  const { return static_cast<int>(edges_.size()); }
    const std::vector<bool>&           binary_mask() const { return binary_mask_; }
    const std::vector<Eigen::VectorXd>& edges()      const { return edges_; }
    bool fitted()     const { return fitted_; }

    // Number of actual bins for feature f
    int n_bins_for(int f) const { return static_cast<int>(edges_[f].size()) - 1; }

private:
    int n_bins_ = 32;
    std::vector<Eigen::VectorXd> edges_;   // edges_[f] has n_bins_for(f)+1 entries
    std::vector<bool> binary_mask_;        // true → binary (skip histogram binning)
    bool fitted_ = false;
};

} // namespace hvrt
