#pragma once
#include "geoxgb/geoxgb_base.h"
#include <cmath>

namespace geoxgb {

// Binary classifier using Focal Loss.
//
// Focal loss (Lin et al., 2017):
//   L_i = -(1-p_t)^γ · log(p_t)
//
// where p_t = p if y=1, (1-p) if y=0, and γ >= 0 is the focusing parameter.
//
// γ=0 recovers standard log-loss. γ>0 downweights easy examples (p_t near 1)
// and focuses learning on hard, misclassified examples near the boundary.
//
// Gradient w.r.t. F (raw score):
//   For y=1: g = p_t^γ · [(1-p)·(γ·log(p_t) - (1-p_t)/p_t)]  ... complex
//
// Simplified: we use the standard log-loss gradient weighted by (1-p_t)^γ:
//   g_i = (y_i - p_i) · (1 - p_t_i)^γ
//
// This is the commonly used approximation that preserves gradient direction
// while modulating magnitude by the focal weight.
//
// With γ=2 (default): easy samples (p_t ≈ 0.95) get weight 0.0025,
// while hard samples (p_t ≈ 0.5) get weight 0.25.

class GeoXGBFocalClassifier : public GeoXGBBase {
public:
    explicit GeoXGBFocalClassifier(GeoXGBConfig cfg = GeoXGBConfig{},
                                    double gamma = 2.0)
        : GeoXGBBase(std::move(cfg)), gamma_(gamma) {}

    GeoXGBFocalClassifier& fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        fit_boosting(X, y);
        return *this;
    }

    Eigen::VectorXd predict_proba(const Eigen::MatrixXd& X) const {
        Eigen::VectorXd raw = predict_raw(X);
        return raw.unaryExpr([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
    }

    Eigen::VectorXi predict(const Eigen::MatrixXd& X) const {
        Eigen::VectorXd p = predict_proba(X);
        return p.unaryExpr([](double v) -> int { return v >= 0.5 ? 1 : 0; })
                .cast<int>();
    }

    double gamma() const { return gamma_; }

protected:
    double gamma_;

    double init_prediction(const Eigen::VectorXd& y) const override {
        double p = y.mean();
        p = std::max(1e-7, std::min(1.0 - 1e-7, p));
        return std::log(p / (1.0 - p));
    }

    Eigen::VectorXd gradients(const Eigen::VectorXd& y,
                               const Eigen::VectorXd& preds) const override {
        const int n = static_cast<int>(y.size());
        Eigen::VectorXd sig = preds.unaryExpr([](double v) {
            return 1.0 / (1.0 + std::exp(-v));
        });
        Eigen::VectorXd g(n);
        for (int i = 0; i < n; ++i) {
            double p = sig[i];
            // p_t = probability assigned to the true class
            double p_t = (y[i] > 0.5) ? p : (1.0 - p);
            p_t = std::max(1e-7, p_t);
            // Focal weight: (1 - p_t)^gamma
            double focal_w = std::pow(1.0 - p_t, gamma_);
            g[i] = (y[i] - p) * focal_w;
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
