#pragma once
#include "geoxgb/geoxgb_base.h"
#include <cmath>

namespace geoxgb {

// Binary classifier using Exponential Loss (AdaBoost-style).
//
// Exponential loss:
//   L_i = exp(-y_signed_i · F_i)
//
// where y_signed = 2y - 1 ∈ {-1, +1} and F is the raw boosting score.
//
// Gradient (negative):
//   g_i = y_signed_i · exp(-y_signed_i · F_i)
//
// Properties:
// - Exponentially penalizes misclassified samples
// - Well-classified samples contribute near-zero gradient
// - Equivalent to AdaBoost in the limit of weak learners
// - More aggressive than log-loss on outliers / mislabeled data
// - Deterministic, no pairwise computation
//
// Risk: less robust to label noise than log-loss (exponential penalty
// on misclassified noisy labels can dominate learning).
// Clamp raw predictions to [-10, 10] to prevent numerical overflow.

class GeoXGBExpClassifier : public GeoXGBBase {
public:
    explicit GeoXGBExpClassifier(GeoXGBConfig cfg = GeoXGBConfig{})
        : GeoXGBBase(std::move(cfg)) {}

    GeoXGBExpClassifier& fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        fit_boosting(X, y);
        return *this;
    }

    Eigen::VectorXd predict_proba(const Eigen::MatrixXd& X) const {
        Eigen::VectorXd raw = predict_raw(X);
        // Convert exp-loss scores to probabilities via sigmoid
        return raw.unaryExpr([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd& X) const {
        Eigen::VectorXd p = predict_proba(X);
        return p.unaryExpr([](double v) -> int { return v >= 0.5 ? 1 : 0; })
                .cast<int>();
    }

protected:
    double init_prediction(const Eigen::VectorXd& y) const override {
        double p = y.mean();
        p = std::max(1e-7, std::min(1.0 - 1e-7, p));
        return 0.5 * std::log(p / (1.0 - p));  // half-logit for AdaBoost
    }

    Eigen::VectorXd gradients(const Eigen::VectorXd& y,
                               const Eigen::VectorXd& preds) const override {
        const int n = static_cast<int>(y.size());
        Eigen::VectorXd g(n);
        for (int i = 0; i < n; ++i) {
            double y_s = 2.0 * y[i] - 1.0;  // {0,1} -> {-1,+1}
            double F = std::max(-10.0, std::min(10.0, preds[i]));  // clamp
            double w = std::exp(-y_s * F);
            g[i] = y_s * w;
            if (cfg_.pos_class_weight != 1.0 && y[i] > 0.5)
                g[i] *= cfg_.pos_class_weight;
        }
        return g;
    }

    Eigen::VectorXd targets_from_gradients(
            const Eigen::VectorXd& grads,
            const Eigen::VectorXd& preds) const override {
        // Approximate: sigmoid(preds) + grads (same as other classifiers)
        Eigen::VectorXd sig = preds.unaryExpr([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
        return sig + grads;
    }
};

} // namespace geoxgb
