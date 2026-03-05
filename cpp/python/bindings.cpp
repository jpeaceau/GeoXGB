#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "geoxgb/geoxgb_regressor.h"
#include "geoxgb/geoxgb_classifier.h"
#include "hvrt/reduce.h"

namespace py = pybind11;
using namespace geoxgb;

PYBIND11_MODULE(_geoxgb_cpp, m) {
    m.doc() = "GeoXGB C++ backend — geometry-aware gradient boosting";

    // ── Debug utilities ───────────────────────────────────────────────────────
    m.def("variance_ordered_enable_timing",
          &hvrt::variance_ordered_enable_timing,
          py::arg("enable"),
          "Enable/disable per-call timing prints inside variance_ordered().");

    // ── Config ────────────────────────────────────────────────────────────────
    py::class_<GeoXGBConfig>(m, "GeoXGBConfig")
        .def(py::init<>())
        .def_readwrite("n_rounds",              &GeoXGBConfig::n_rounds)
        .def_readwrite("learning_rate",         &GeoXGBConfig::learning_rate)
        .def_readwrite("max_depth",             &GeoXGBConfig::max_depth)
        .def_readwrite("min_samples_leaf",      &GeoXGBConfig::min_samples_leaf)
        .def_readwrite("reduce_ratio",          &GeoXGBConfig::reduce_ratio)
        .def_readwrite("expand_ratio",          &GeoXGBConfig::expand_ratio)
        .def_readwrite("y_weight",              &GeoXGBConfig::y_weight)
        .def_readwrite("refit_interval",        &GeoXGBConfig::refit_interval)
        .def_readwrite("auto_noise",            &GeoXGBConfig::auto_noise)
        .def_readwrite("noise_guard",           &GeoXGBConfig::noise_guard)
        .def_readwrite("refit_noise_floor",     &GeoXGBConfig::refit_noise_floor)
        .def_readwrite("auto_expand",           &GeoXGBConfig::auto_expand)
        .def_readwrite("min_train_samples",     &GeoXGBConfig::min_train_samples)
        .def_readwrite("bandwidth",             &GeoXGBConfig::bandwidth)
        .def_readwrite("hvrt_min_samples_leaf", &GeoXGBConfig::hvrt_min_samples_leaf)
        .def_readwrite("hvrt_n_partitions",     &GeoXGBConfig::hvrt_n_partitions)
        .def_readwrite("n_bins",                &GeoXGBConfig::n_bins)
        .def_readwrite("random_state",          &GeoXGBConfig::random_state)
        .def_readwrite("variance_weighted",     &GeoXGBConfig::variance_weighted)
        .def_readwrite("adaptive_y_weight",     &GeoXGBConfig::adaptive_y_weight)
        .def_readwrite("blend_cross_term",      &GeoXGBConfig::blend_cross_term)
        .def_readwrite("syn_partition_correct", &GeoXGBConfig::syn_partition_correct)
        .def_readwrite("y_geom_coupling",       &GeoXGBConfig::y_geom_coupling)
        .def_readwrite("selective_target",      &GeoXGBConfig::selective_target)
        .def_readwrite("selective_k_pairs",     &GeoXGBConfig::selective_k_pairs)
        .def_readwrite("d_geom_threshold",        &GeoXGBConfig::d_geom_threshold)
        .def_readwrite("residual_correct_lambda", &GeoXGBConfig::residual_correct_lambda)
        .def_readwrite("partitioner",           &GeoXGBConfig::partitioner)
        .def_readwrite("reduce_method",         &GeoXGBConfig::reduce_method)
        .def_readwrite("generation_strategy",   &GeoXGBConfig::generation_strategy)
        .def_readwrite("adaptive_reduce_ratio", &GeoXGBConfig::adaptive_reduce_ratio)
        .def_readwrite("sample_block_n",        &GeoXGBConfig::sample_block_n)
        .def_readwrite("leave_last_block_out",  &GeoXGBConfig::leave_last_block_out)
        .def("__repr__", [](const GeoXGBConfig& c) {
            return "<GeoXGBConfig n_rounds=" + std::to_string(c.n_rounds) +
                   " lr=" + std::to_string(c.learning_rate) +
                   " max_depth=" + std::to_string(c.max_depth) + ">";
        });

    // ── Regressor ─────────────────────────────────────────────────────────────
    py::class_<GeoXGBRegressor>(m, "CppGeoXGBRegressor")
        .def(py::init<GeoXGBConfig>(), py::arg("cfg") = GeoXGBConfig{})
        .def("fit",
             [](GeoXGBRegressor& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X,
                const Eigen::Ref<const Eigen::VectorXd>& y) -> GeoXGBRegressor& {
                 return self.fit(X, y);
             },
             py::arg("X"), py::arg("y"),
             py::return_value_policy::reference,
             "Fit the regressor. X: (n, d) float64, y: (n,) float64")
        .def("predict",
             [](const GeoXGBRegressor& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.predict(X);
             },
             py::arg("X"),
             "Predict continuous values. Returns ndarray (n,).")
        .def("is_fitted",             &GeoXGBRegressor::is_fitted)
        .def("convergence_round",     &GeoXGBRegressor::convergence_round)
        .def("last_noise_modulation", &GeoXGBRegressor::last_noise_modulation)
        .def("rho_trace",    [](const GeoXGBRegressor& r) {
            return std::vector<double>(r.rho_trace());
        }, "Per-refit |ρ(geom, residuals)| trace. Empty if adaptive_y_weight=False.")
        .def("yw_eff_trace", [](const GeoXGBRegressor& r) {
            return std::vector<double>(r.yw_eff_trace());
        }, "Per-refit effective y_weight trace (= y_weight * |ρ|).")
        .def("__repr__", [](const GeoXGBRegressor& r) {
            return std::string("<CppGeoXGBRegressor fitted=") +
                   (r.is_fitted() ? "True" : "False") + ">";
        });

    // ── Classifier ────────────────────────────────────────────────────────────
    py::class_<GeoXGBClassifier>(m, "CppGeoXGBClassifier")
        .def(py::init<GeoXGBConfig>(), py::arg("cfg") = GeoXGBConfig{})
        .def("fit",
             [](GeoXGBClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X,
                const Eigen::Ref<const Eigen::VectorXd>& y) -> GeoXGBClassifier& {
                 return self.fit(X, y);
             },
             py::arg("X"), py::arg("y"),
             py::return_value_policy::reference,
             "Fit the classifier. y must be float-encoded 0/1.")
        .def("predict",
             [](const GeoXGBClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.predict(X);
             },
             py::arg("X"),
             "Hard class labels (0 or 1). Returns ndarray (n,) int.")
        .def("predict_proba",
             [](const GeoXGBClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 Eigen::VectorXd p1 = self.predict_proba(X);
                 const int n = static_cast<int>(p1.size());
                 Eigen::MatrixXd out(n, 2);
                 out.col(0) = 1.0 - p1.array();
                 out.col(1) = p1;
                 return out;
             },
             py::arg("X"),
             "Class probabilities. Returns ndarray (n, 2) with [P(0), P(1)].")
        .def("is_fitted",             &GeoXGBClassifier::is_fitted)
        .def("convergence_round",     &GeoXGBClassifier::convergence_round)
        .def("last_noise_modulation", &GeoXGBClassifier::last_noise_modulation)
        .def("rho_trace",    [](const GeoXGBClassifier& c) {
            return std::vector<double>(c.rho_trace());
        }, "Per-refit |ρ(geom, residuals)| trace. Empty if adaptive_y_weight=False.")
        .def("yw_eff_trace", [](const GeoXGBClassifier& c) {
            return std::vector<double>(c.yw_eff_trace());
        }, "Per-refit effective y_weight trace (= y_weight * |ρ|).")
        .def("__repr__", [](const GeoXGBClassifier& c) {
            return std::string("<CppGeoXGBClassifier fitted=") +
                   (c.is_fitted() ? "True" : "False") + ">";
        });
}
