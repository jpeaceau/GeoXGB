#pragma once
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <functional>
#include "geoxgb/types.h"
#include "hvrt/hvrt.h"
#include "hvrt/binner.h"
#include "hvrt/tree.h"

namespace geoxgb {

// ── GeoXGBBase ────────────────────────────────────────────────────────────────
//
// Shared C++ core for GeoXGBRegressor and GeoXGBClassifier.
// Subclasses supply:
//   init_prediction(y)       → scalar baseline (mean for regression,
//                              log-odds for classification)
//   gradients(y, preds)      → negative gradient vector
//   to_probability(raw)      → optional sigmoid transform (classifier only)

class GeoXGBMulticlassClassifier;  // forward decl

class GeoXGBBase {
    friend class GeoXGBMulticlassClassifier;
public:
    explicit GeoXGBBase(GeoXGBConfig cfg = GeoXGBConfig{});
    virtual ~GeoXGBBase() = default;

    // Returns raw leaf-sum predictions (log-odds for classifier).
    Eigen::VectorXd predict_raw(const Eigen::MatrixXd& X) const;

    bool is_fitted() const { return fitted_; }

    // Convergence info
    int  convergence_round() const { return convergence_round_; }

    // Noise modulation from last resample (0=noise, 1=clean)
    double last_noise_modulation() const { return last_noise_mod_; }

    // Noise modulation from the INITIAL resample (proxy for dataset SNR).
    double init_noise_modulation() const { return init_noise_mod_; }

    // Original training n (before any block cycling subsetting).
    int n_train()       const { return n_train_arg_; }

    // Samples kept after the initial HVRT reduce (before expansion).
    int n_init_reduced() const { return n_init_reduced_; }

    // Per-refit trace of |ρ(geom_target, residuals)| and effective y_weight.
    // Length = number of refits performed.  Empty when adaptive_y_weight=false
    // or refit_interval=0.  Use for gradient analysis post-fit.
    const std::vector<double>& rho_trace()    const { return rho_trace_; }
    const std::vector<double>& yw_eff_trace() const { return yw_eff_trace_; }

    // ── Geometry / interpretability accessors ─────────────────────────────────
    // Populated at the end of fit_boosting() from the most recent HVRT geometry.
    // X_z(): whitened training data (full X, not just reduced).
    // partition_ids(): per-sample HVRT partition assignment.
    // train_predictions(): ensemble raw predictions on full training X.
    // to_z(): whiten new query points using the same whitener.
    // apply(): assign new query points to partition IDs.
    const Eigen::MatrixXd& X_z()               const { return X_z_; }
    const Eigen::VectorXi& partition_ids()     const { return partition_ids_; }
    const Eigen::VectorXd& train_predictions() const { return train_predictions_; }
    Eigen::MatrixXd        to_z(const Eigen::MatrixXd& X_new) const;
    Eigen::VectorXi        apply(const Eigen::MatrixXd& X_new) const;

    // Aggregate impurity-based feature importance across all GBT weak learners.
    std::vector<double> feature_importances() const;

    // Serialization: export/import full model state as binary blob.
    // Restores prediction capability and geometry state (X_z, partition_ids,
    // train_predictions).  Does NOT restore HVRT state (to_z/apply will throw).
    std::vector<uint8_t> to_bytes() const;
    void from_bytes(const std::vector<uint8_t>& data);

protected:
    // Called by subclass fit(); handles the full boosting + resampling loop.
    void fit_boosting(
        const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y);

    // Subclass interface
    virtual double             init_prediction(const Eigen::VectorXd& y) const = 0;
    virtual Eigen::VectorXd    gradients(const Eigen::VectorXd& y,
                                         const Eigen::VectorXd& preds) const = 0;
    // Reconstruct training targets from gradients + current predictions.
    // Regression: y = preds + grads.  Classifier: handled in subclass.
    virtual Eigen::VectorXd    targets_from_gradients(
                                    const Eigen::VectorXd& grads,
                                    const Eigen::VectorXd& preds) const;

    GeoXGBConfig cfg_;
    bool         fitted_            = false;
    int          convergence_round_ = -1;
    double       last_noise_mod_    = 1.0;
    double       init_noise_mod_    = 1.0;    // noise mod at initial resample
    int          n_train_arg_       = 0;      // full training n passed to fit_boosting
    int          n_init_reduced_    = 0;      // red_idx.size() at initial resample
    std::vector<double> rho_trace_;     // |ρ(geom, residuals)| at each refit
    std::vector<double> yw_eff_trace_;  // effective y_weight used at each refit
    std::vector<double> convergence_losses_;  // mean |grad| at each refit boundary

private:
    // ── Per-tree state ────────────────────────────────────────────────────────
    struct WeakLearner {
        hvrt::PartitionTree tree;
        std::vector<int>    cont_cols; // always 0..d-1 for GBT trees
        double              lr;
    };

    // ── Resampling state ──────────────────────────────────────────────────────
    struct ResampleResult {
        Eigen::MatrixXd X;      // training features (real + optional synthetic)
        Eigen::VectorXd y;      // training targets (gradients reconstructed)
        Eigen::VectorXi red_idx;// indices into original X for real samples
        double noise_mod;
        int    n_expanded;
    };

    ResampleResult do_resample(
        const Eigen::MatrixXd& X_full,
        const Eigen::VectorXd& y_signal,
        int seed_offset,
        std::shared_ptr<hvrt::HVRT>& hvrt_out);

    // k-NN y-assignment for synthetic samples (global IDW, k=3)
    Eigen::VectorXd knn_assign_y(
        const Eigen::MatrixXd& X_syn_z,
        const Eigen::MatrixXd& X_red_z,
        const Eigen::VectorXd& y_red) const;

    // Predict on Xr using trees built so far (used at refit when expansion active)
    Eigen::VectorXd predict_from_trees(
        const Eigen::MatrixXd& X,
        int up_to_tree) const;

    // ── Fitted state ─────────────────────────────────────────────────────────
    double                          init_pred_  = 0.0;
    std::vector<WeakLearner>        trees_;

    // ── Persistent geometry (populated at end of fit_boosting) ────────────────
    std::shared_ptr<hvrt::HVRT>  last_hvrt_;          // HVRT at last refit
    Eigen::MatrixXd              X_z_;                // (n_train × d) whitened full training data
    Eigen::VectorXi              partition_ids_;       // (n_train,) partition membership
    Eigen::VectorXd              train_predictions_;   // (n_train,) ensemble preds on X_arg

    // ── kNN scratch buffer ────────────────────────────────────────────────────
    // knn_assign_y allocates a (n_syn × n_red) distance matrix each call.
    // For n_syn=500, n_red=3500, d=10 that is ~14 MB per call × 200 calls/fit.
    // Caching it as a mutable member makes resize() a no-op after round 1.
    mutable Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> knn_D_;

    // Cached full-dataset binning (done once at fit start, reused every round)
    hvrt::Binner                         full_binner_;
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X_bin_full_;
    std::vector<int>                     cont_cols_full_;
    // Bin edges extracted from full_binner_ after fit; injected into each GBT
    // weak learner before build() to skip per-round O(n·d·log n) re-sort.
    std::vector<Eigen::VectorXd>         gbt_bin_edges_;

    // Predict stride: count of trees whose full-dataset predict is deferred
    int pending_trees_for_preds_ = 0;

    // Gradient amplification: per-partition power-scaled weight cache.
    // Recomputed at each refit boundary, reused between refits.
    std::vector<double> grad_amp_weights_;

    // ── Sample-without-replacement tracking ──────────────────────────────
    // When cfg_.sample_without_replacement is true, tracks which samples
    // (indices into X_cur) have been consumed by a refit window.
    // Reset when fewer than n_keep unused samples remain (new epoch).
    std::vector<bool> used_samples_;
};

} // namespace geoxgb
