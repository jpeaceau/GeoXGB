#pragma once
#include "geoxgb/geoxgb_base.h"
#include <cmath>

namespace geoxgb {

// Binary classifier using Gini-impurity loss.
//
// Gini impurity:  G(p) = 2·p·(1 − p)
//
// In the boosting framework with raw scores F and p = σ(F):
//   Loss per sample:  L_i = 2·p_i·(1 − p_i)
//   dL/dF = 2·(1 − 2p)·p·(1 − p)
//
// The pseudo-residual (negative gradient) used for tree fitting:
//   g_i = y_i − p_i,  weighted by  w_i = |1 − 2·p_i|
//
// This concentrates learning effort on samples near the decision boundary
// (p ≈ 0.5) less aggressively than log-loss, giving more uniform attention.
// Samples already well-classified (p near 0 or 1) still contribute
// proportionally to their Gini gradient.
//
// Compared to log-loss:
// - Bounded gradient magnitudes (no log-divergence near 0 or 1)
// - Smoother loss surface near the boundary
// - May be more robust to label noise (less aggressive on confident errors)

class GeoXGBGiniClassifier : public GeoXGBBase {
public:
    explicit GeoXGBGiniClassifier(GeoXGBConfig cfg = GeoXGBConfig{})
        : GeoXGBBase(std::move(cfg)) {}

    GeoXGBGiniClassifier& fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
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
        return std::log(p / (1.0 - p));  // log-odds init (same as log-loss)
    }

    Eigen::VectorXd gradients(const Eigen::VectorXd& y,
                               const Eigen::VectorXd& preds) const override {
        // Gini loss gradient: dL/dF = 2·(1 − 2p)·p·(1 − p)
        // Negative gradient (pseudo-residual direction):
        //   For y=1: want to increase F → push p up → g > 0 when p < 1
        //   For y=0: want to decrease F → push p down → g < 0 when p > 0
        //
        // We use: g_i = (y_i − p_i) · |1 − 2·p_i|
        // This is the Gini-weighted version of the log-loss gradient.
        // The |1-2p| factor downweights samples near the boundary (p≈0.5)
        // relative to confident predictions, while keeping gradient sign correct.
        const int n = static_cast<int>(y.size());
        Eigen::VectorXd sig = preds.unaryExpr([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
        Eigen::VectorXd g(n);
        for (int i = 0; i < n; ++i) {
            double p = sig[i];
            double w = std::abs(1.0 - 2.0 * p);
            g[i] = (y[i] - p) * w;
            if (cfg_.pos_class_weight != 1.0 && y[i] > 0.5)
                g[i] *= cfg_.pos_class_weight;
        }
        return g;
    }

    Eigen::VectorXd targets_from_gradients(
            const Eigen::VectorXd& grads,
            const Eigen::VectorXd& preds) const override {
        // Reconstruct pseudo-targets from Gini gradients.
        // Since g = (y - p) * |1-2p|, we can't perfectly invert.
        // Use the same approach as log-loss: y_approx = sigmoid(preds) + grads
        // This is an approximation but works well for HVRT target construction.
        Eigen::VectorXd sig = preds.unaryExpr([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
        return sig + grads;
    }
};

} // namespace geoxgb
