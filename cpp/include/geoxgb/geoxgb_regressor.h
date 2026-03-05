#pragma once
#include "geoxgb/geoxgb_base.h"
#include <algorithm>
#include <vector>

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
        if (cfg_.loss == "absolute_error") {
            std::vector<double> v(y.data(), y.data() + y.size());
            const int mid = static_cast<int>(v.size()) / 2;
            std::nth_element(v.begin(), v.begin() + mid, v.end());
            return v[mid];  // median
        }
        return y.mean();
    }

    Eigen::VectorXd gradients(const Eigen::VectorXd& y,
                               const Eigen::VectorXd& preds) const override {
        if (cfg_.loss == "absolute_error")
            return (y - preds).array().sign().matrix();
        return y - preds;  // negative gradient of squared-error
    }
};

} // namespace geoxgb
