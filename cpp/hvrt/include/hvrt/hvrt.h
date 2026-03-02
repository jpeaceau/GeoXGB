#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <optional>
#include "hvrt/types.h"
#include "hvrt/whitener.h"
#include "hvrt/binner.h"
#include "hvrt/tree.h"
#include "hvrt/expand.h"

namespace hvrt {

class HVRT {
public:
    explicit HVRT(HVRTConfig cfg = HVRTConfig{});

    // ── Fit ───────────────────────────────────────────────────────────────────
    // X            : n x d  (raw, not whitened)
    // y            : optional target variable (used for y_weight blending)
    // feature_types: accepted for API compatibility; encoding is user-managed upstream.
    //               All columns are treated as continuous (histogram-based splits).
    HVRT& fit(const Eigen::MatrixXd& X,
              std::optional<Eigen::VectorXd> y = std::nullopt,
              std::optional<std::vector<std::string>> feature_types = std::nullopt);

    // ── Refit (fast path when X is unchanged) ────────────────────────────────
    // Reuses the cached whitener output (X_z_), binner output (X_binned_cache_),
    // and geometry target from the previous fit().  Only re-runs tree_.build()
    // with a new blended target and expander_.prepare() for the new partitions.
    // Saves O(n·d·log n) whitening + binning + geometry-target cost per refit.
    // Must call fit() at least once before refit().
    HVRT& refit(std::optional<Eigen::VectorXd> y = std::nullopt);

    // ── Reduce ────────────────────────────────────────────────────────────────
    // Returns selected rows from original X (un-whitened).
    Eigen::MatrixXd reduce(
        std::optional<int>    n       = std::nullopt,
        std::optional<double> ratio   = std::nullopt,
        const std::string&    method  = "centroid_fps",
        bool var_weighted             = true,
        std::optional<int>    n_parts = std::nullopt) const;

    // Return indices only
    std::vector<int> reduce_indices(
        std::optional<int>    n       = std::nullopt,
        std::optional<double> ratio   = std::nullopt,
        const std::string&    method  = "centroid_fps",
        bool var_weighted             = true,
        std::optional<int>    n_parts = std::nullopt) const;

    // ── Expand ────────────────────────────────────────────────────────────────
    // Returns n synthetic samples in the original (un-whitened) feature space.
    Eigen::MatrixXd expand(
        int n,
        bool var_weighted                        = true,
        std::optional<float> bandwidth           = std::nullopt,
        const std::string& strategy              = "auto",
        bool adaptive_bandwidth                  = false,
        std::optional<int> n_parts               = std::nullopt) const;

    // ── Augment ───────────────────────────────────────────────────────────────
    // Augment = expand n rows and concatenate with original X.
    Eigen::MatrixXd augment(
        int n,
        bool var_weighted    = true,
        std::optional<int> n_parts = std::nullopt) const;

    // ── Partitioning info ─────────────────────────────────────────────────────
    std::vector<PartitionInfo> get_partitions() const;

    // ── Novelty ───────────────────────────────────────────────────────────────
    // Minimum distance from each row of X_new to the training set.
    Eigen::VectorXd compute_novelty(const Eigen::MatrixXd& X_new) const;

    // ── Tree apply ────────────────────────────────────────────────────────────
    // Assign partition IDs to new data.
    Eigen::VectorXi apply(const Eigen::MatrixXd& X_new) const;

    // ── Static helpers ────────────────────────────────────────────────────────
    static ParamRecommendation recommend_params(const Eigen::MatrixXd& X);

    // ── Stored attributes (for Python bindings / GeoXGB) ─────────────────────
    const Eigen::MatrixXd&  X_z()                const { return X_z_; }
    const Eigen::VectorXi&  partition_ids()       const { return partition_ids_; }
    std::vector<int>         unique_partitions()  const;
    const Whitener&          whitener()           const { return whitener_; }
    const PartitionTree&     tree()               const { return tree_; }

    // _to_z: whiten new data (same as whitener_.transform)
    Eigen::MatrixXd to_z(const Eigen::MatrixXd& X) const;

    bool fitted() const { return fitted_; }

    // Returns true if the last refit() produced identical partition assignments.
    // When true, the KDE parameters (expander_) are unchanged; callers may reuse
    // previously generated synthetic samples and skip predict_from_trees on them.
    bool last_refit_stable() const { return last_refit_stable_; }

private:
    HVRTConfig cfg_;

    // Components
    Whitener     whitener_;
    Binner       binner_;
    PartitionTree tree_;
    Expander     expander_;

    // Stored fit state
    Eigen::MatrixXd  X_z_;          // whitened training data
    Eigen::MatrixXd  X_orig_;       // original training data
    Eigen::VectorXi  partition_ids_;
    bool fitted_ = false;

    // Refit cache — populated by fit(), reused by refit()
    using BinMat = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    BinMat              X_binned_cache_;      // binner output (same X → same bins)
    std::vector<int>    hist_cont_cols_;      // continuous column indices
    std::vector<int>    binary_cols_;         // binary column indices
    Eigen::VectorXd     geom_target_cache_;   // geometry-only target (no y blend)

    // Set by refit(): true when partition assignments are identical to the previous call.
    bool last_refit_stable_ = false;

    // Helpers
    static ReductionMethod    parse_reduction_method(const std::string& s);
    static GenerationStrategy parse_generation_strategy(const std::string& s);
};

} // namespace hvrt
