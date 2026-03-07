#pragma once
#include "geoxgb/geoxgb_base.h"
#include <cmath>

namespace geoxgb {

// Binary classifier using Squared Hinge Loss.
//
// Squared hinge loss:
//   L_i = max(0, 1 - y_signed_i · F_i)²
//
// where y_signed = 2y - 1 ∈ {-1, +1} and F is the raw boosting score.
//
// Gradient (negative):
//   g_i = 2 · y_signed_i · max(0, 1 - y_signed_i · F_i)
//       = 0 when correctly classified with margin >= 1
//
// Properties:
// - Margin-based: once a sample is correctly classified with margin >= 1,
//   it contributes zero gradient (efficient, no wasted capacity)
// - Smooth (differentiable everywhere, unlike hinge loss)
// - Geometrically interpretable: seeks a separating hyperplane with margin
// - Bounded gradient magnitude (max |g| = 2)
// - Deterministic, O(n) per round
//
// In GeoXGB context: the HVRT geometry naturally defines local coordinate
// systems where margin separation is geometrically meaningful.

class GeoXGBHingeClassifier : public GeoXGBBase {
public:
    explicit GeoXGBHingeClassifier(GeoXGBConfig cfg = GeoXGBConfig{})
        : GeoXGBBase(std::move(cfg)) {}

    GeoXGBHingeClassifier& fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        fit_boosting(X, y);
        return *this;
    }

    Eigen::VectorXd predict_proba(const Eigen::MatrixXd& X) const {
        Eigen::VectorXd raw = predict_raw(X);
        // Convert margin-based scores to probabilities via sigmoid
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
        // Start at 0 for margin-based loss (no prior bias)
        // Could also use log-odds, but 0 is more natural for hinge
        double p = y.mean();
        p = std::max(1e-7, std::min(1.0 - 1e-7, p));
        return std::log(p / (1.0 - p));
    }

    Eigen::VectorXd gradients(const Eigen::VectorXd& y,
                               const Eigen::VectorXd& preds) const override {
        const int n = static_cast<int>(y.size());
        Eigen::VectorXd g(n);
        for (int i = 0; i < n; ++i) {
            double y_s = 2.0 * y[i] - 1.0;  // {0,1} -> {-1,+1}
            double margin = y_s * preds[i];
            if (margin >= 1.0) {
                g[i] = 0.0;  // correctly classified with sufficient margin
            } else {
                g[i] = 2.0 * y_s * (1.0 - margin);
            }
            if (cfg_.pos_class_weight != 1.0 && y[i] > 0.5)
                g[i] *= cfg_.pos_class_weight;
        }
        return g;
    }

    Eigen::VectorXd targets_from_gradients(
            const Eigen::VectorXd& grads,
            const Eigen::VectorXd& preds) const override {
        Eigen::VectorXd sig = preds.unaryExpr([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
        return sig + grads;
    }
};

} // namespace geoxgb
