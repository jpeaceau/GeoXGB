#pragma once
#include "geoxgb/geoxgb_base.h"
#include <cmath>

namespace geoxgb {

// Binary classifier using log-loss (logistic regression leaf values).

class GeoXGBClassifier : public GeoXGBBase {
public:
    explicit GeoXGBClassifier(GeoXGBConfig cfg = GeoXGBConfig{})
        : GeoXGBBase(std::move(cfg)) {}

    GeoXGBClassifier& fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        // y must be 0/1 float-encoded
        fit_boosting(X, y);
        return *this;
    }

    // Returns predicted class probabilities in [0,1].
    Eigen::VectorXd predict_proba(const Eigen::MatrixXd& X) const {
        Eigen::VectorXd raw = predict_raw(X);
        return raw.unaryExpr([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
    }

    // Returns hard class labels (0 or 1) at threshold 0.5.
    Eigen::VectorXi predict(const Eigen::MatrixXd& X) const {
        Eigen::VectorXd p = predict_proba(X);
        return p.unaryExpr([](double v) -> int { return v >= 0.5 ? 1 : 0; })
                .cast<int>();
    }

protected:
    double init_prediction(const Eigen::VectorXd& y) const override {
        double p = y.mean();
        p = std::max(1e-7, std::min(1.0 - 1e-7, p));
        return std::log(p / (1.0 - p));  // log-odds
    }

    Eigen::VectorXd gradients(const Eigen::VectorXd& y,
                               const Eigen::VectorXd& preds) const override {
        // log-loss gradient: y - sigmoid(preds)
        Eigen::VectorXd sig = preds.unaryExpr([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
        return y - sig;
    }

    Eigen::VectorXd targets_from_gradients(
            const Eigen::VectorXd& grads,
            const Eigen::VectorXd& preds) const override {
        // y_true = sigmoid(preds) + grads
        Eigen::VectorXd sig = preds.unaryExpr([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
        return sig + grads;
    }
};

} // namespace geoxgb
