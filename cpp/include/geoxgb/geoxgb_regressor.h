#pragma once
#include "geoxgb/geoxgb_base.h"

namespace geoxgb {

class GeoXGBRegressor : public GeoXGBBase {
public:
    explicit GeoXGBRegressor(GeoXGBConfig cfg = GeoXGBConfig{})
        : GeoXGBBase(std::move(cfg)) {}

    GeoXGBRegressor& fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
        fit_boosting(X, y);
        return *this;
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const {
        return predict_raw(X);
    }

protected:
    double init_prediction(const Eigen::VectorXd& y) const override {
        return y.mean();
    }
    Eigen::VectorXd gradients(const Eigen::VectorXd& y,
                               const Eigen::VectorXd& preds) const override {
        return y - preds;  // negative gradient of squared-error
    }
};

} // namespace geoxgb
