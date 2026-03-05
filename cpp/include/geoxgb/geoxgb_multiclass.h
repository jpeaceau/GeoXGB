#pragma once
#include "geoxgb/geoxgb_base.h"
#include <cmath>

namespace geoxgb {

// Shared-geometry multiclass classifier.
//
// Instead of K independent OvR binary ensembles (each with its own HVRT),
// this class trains K class-specific tree ensembles that share a single
// HVRT geometry.  Benefits:
//   - 1 HVRT fit/refit per cycle instead of K  (dominant cost savings)
//   - Coherent partition structure across all classes
//   - Combined gradient signal drives geometry toward the hardest boundaries
//
// The HVRT y_signal is the L2 norm of the K-dimensional gradient vector,
// so partitions focus on samples where the model struggles most across
// all classes simultaneously.

class GeoXGBMulticlassClassifier : public GeoXGBBase {
public:
    explicit GeoXGBMulticlassClassifier(GeoXGBConfig cfg = GeoXGBConfig{})
        : GeoXGBBase(std::move(cfg)) {}

    // Y_onehot: n × K binary matrix (one-hot encoded).
    // class_weights: K-length vector (empty = uniform).
    GeoXGBMulticlassClassifier& fit(
        const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y_onehot,
        const Eigen::VectorXd& class_weights = Eigen::VectorXd());

    // Raw logits: n × K matrix.
    Eigen::MatrixXd predict_raw_multi(const Eigen::MatrixXd& X) const;

    // Softmax probabilities: n × K matrix.
    Eigen::MatrixXd predict_proba_multi(const Eigen::MatrixXd& X) const;

    // Argmax class labels: n-vector of ints {0, ..., K-1}.
    Eigen::VectorXi predict_multi(const Eigen::MatrixXd& X) const;

    int n_classes() const { return n_classes_; }

    // Per-class training predictions: n × K matrix of raw logits.
    const Eigen::MatrixXd& train_predictions_multi() const { return train_predictions_multi_; }

    // Aggregate feature importance across all K × n_rounds trees.
    std::vector<double> feature_importances_multi() const;

protected:
    // Required by GeoXGBBase (not used — multiclass has its own loop).
    double          init_prediction(const Eigen::VectorXd&) const override { return 0.0; }
    Eigen::VectorXd gradients(const Eigen::VectorXd&, const Eigen::VectorXd&) const override {
        return Eigen::VectorXd();
    }

private:
    int n_classes_ = 0;
    Eigen::VectorXd init_preds_;      // (K,) per-class log-odds baseline
    Eigen::VectorXd class_weights_;   // (K,) or empty

    // mc_trees_[k] = per-round weak learners for class k.
    std::vector<std::vector<WeakLearner>> mc_trees_;

    // Per-class training predictions on last X_cur (n × K raw logits).
    Eigen::MatrixXd train_predictions_multi_;
};

} // namespace geoxgb
