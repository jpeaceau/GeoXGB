#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include "hvrt/types.h"

namespace hvrt {

// ── Tree node ─────────────────────────────────────────────────────────────────

struct TreeNode {
    int    feature_idx   = -1;     // global feature index (in X_z)
    double threshold     = 0.0;
    bool   is_binary     = false;  // true → binary-stream split (value <= threshold)
    int    left          = -1;     // child node index
    int    right         = -1;
    bool   is_leaf       = false;
    int    partition_id  = -1;     // assigned at leaf (HVRT partition routing)
    double leaf_value    = 0.0;    // mean target at leaf (used for GBT prediction)
};

// ── Partition tree ────────────────────────────────────────────────────────────

class PartitionTree {
public:
    // build: construct tree on X_z, using pre-computed binned continuous matrix
    // and binary feature matrix.
    //
    // X_z        : n x d_full  (whitened, all features)
    // X_binned   : n x d_cont  (row-major uint8, continuous features only)
    // cont_cols  : which columns in X_z are continuous (for X_binned indexing)
    // binary_cols: which columns in X_z are binary
    // target     : n-vector synthetic target
    // cfg        : HVRT config (n_partitions, min_samples_leaf, max_depth)
    //
    // Returns partition_ids: n-vector of integer partition labels (0-based).
    Eigen::VectorXi build(
        const Eigen::MatrixXd& X_z,
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_binned,
        const std::vector<int>& cont_cols,
        const std::vector<int>& binary_cols,
        const Eigen::VectorXd& target,
        const HVRTConfig& cfg);

    // apply: assign partition IDs to new samples using the fitted tree.
    Eigen::VectorXi apply(const Eigen::MatrixXd& X_z) const;

    // predict: return leaf_value for each sample (used for GBT weak learner prediction).
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;

    // predict_into: write leaf values into a pre-allocated buffer.
    // Avoids the heap allocation inside predict(); use inside parallel loops.
    void predict_into(const Eigen::MatrixXd& X, Eigen::VectorXd& out) const;

    // Feature importance: fraction of total variance-reduction gain per feature.
    // Length d_full (all features in X_z used during fit).
    const std::vector<double>& feature_importances() const { return feature_importances_; }

    int n_leaves() const { return n_leaves_; }
    bool fitted()  const { return fitted_; }

    // Inject pre-computed bin edges so the next build() skips the O(n·d·log n) sort.
    // Edges must match the Binner that produced X_binned passed to build().
    void inject_bin_edges(const std::vector<Eigen::VectorXd>& edges,
                          const std::vector<int>& cont_cols,
                          int n_bins) {
        bin_edges_        = edges;
        cont_cols_cached_ = cont_cols;
        n_bins_cached_    = n_bins;
        bin_edges_valid_  = true;
    }

    // Auto-tune helpers (static, exposed for HVRT orchestrator)
    static std::pair<int, int> auto_tune_params(int n, int d, bool for_reduction);

private:
    // ── Split evaluation helpers ──────────────────────────────────────────────
    struct SplitResult {
        bool  valid      = false;
        int   feature    = -1;
        double threshold = 0.0;
        bool  is_binary  = false;
        double gain      = 0.0;
        int   bin        = -1;
    };

    static SplitResult evaluate_continuous_splits(
        const std::vector<int>& indices,
        const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_binned,
        const std::vector<int>& cont_cols,
        const Eigen::VectorXd& target,
        const std::vector<Eigen::VectorXd>& bin_edges,
        int n_bins,
        int min_samples_leaf,
        SplitStrategy strategy = SplitStrategy::Best,
        uint64_t rng_state = 0);

    SplitResult evaluate_binary_splits(
        const std::vector<int>& indices,
        const Eigen::MatrixXd& X_z,
        const std::vector<int>& binary_cols,
        const Eigen::VectorXd& target) const;

    // ── State ─────────────────────────────────────────────────────────────────
    std::vector<TreeNode> nodes_;
    std::vector<double>   feature_importances_;
    int d_full_  = 0;
    int n_leaves_= 0;
    bool fitted_ = false;

    // cached bin_edges from Binner (needed during build + apply)
    std::vector<Eigen::VectorXd> bin_edges_;
    std::vector<int> cont_cols_cached_;
    std::vector<int> binary_cols_cached_;
    int n_bins_cached_ = 32;

    // When true, bin_edges_ is valid and can be reused without resorting X_z.
    // Set after first build(); cleared if cont_cols change.
    bool bin_edges_valid_ = false;
};

} // namespace hvrt
