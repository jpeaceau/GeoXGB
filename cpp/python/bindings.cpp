#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "geoxgb/geoxgb_regressor.h"
#include "geoxgb/geoxgb_classifier.h"
#include "geoxgb/geoxgb_multiclass.h"
#include "geoxgb/geoxgb_gini_classifier.h"
#include "geoxgb/geoxgb_focal_classifier.h"
#include "geoxgb/geoxgb_exp_classifier.h"
#include "geoxgb/geoxgb_hinge_classifier.h"
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
        .def_readwrite("loss",                  &GeoXGBConfig::loss)
        .def_readwrite("convergence_tol",       &GeoXGBConfig::convergence_tol)
        .def_readwrite("pos_class_weight",      &GeoXGBConfig::pos_class_weight)
        .def_readwrite("e3_target_lambda",      &GeoXGBConfig::e3_target_lambda)
        .def_readwrite("lazy_refit_tol",        &GeoXGBConfig::lazy_refit_tol)
        .def_readwrite("fixed_geometry",        &GeoXGBConfig::fixed_geometry)
        .def_readwrite("progressive_expand",    &GeoXGBConfig::progressive_expand)
        // fast_refit removed — was introducing unnecessary randomness
        .def_readwrite("sample_without_replacement", &GeoXGBConfig::sample_without_replacement)
        .def_readwrite("colsample_bytree",     &GeoXGBConfig::colsample_bytree)
        .def_readwrite("predict_stride",       &GeoXGBConfig::predict_stride)
        .def_readwrite("grad_budget_weight",   &GeoXGBConfig::grad_budget_weight)
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
        // ── Geometry / interpretability ──────────────────────────────────────
        .def("X_z",               [](const GeoXGBRegressor& m) -> Eigen::MatrixXd {
            return m.X_z(); })
        .def("partition_ids",     [](const GeoXGBRegressor& m) -> Eigen::VectorXi {
            return m.partition_ids(); })
        .def("train_predictions", [](const GeoXGBRegressor& m) -> Eigen::VectorXd {
            return m.train_predictions(); })
        .def("to_z",  [](const GeoXGBRegressor& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::MatrixXd {
            return m.to_z(X); })
        .def("apply", [](const GeoXGBRegressor& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::VectorXi {
            return m.apply(X); })
        .def("feature_importances", [](const GeoXGBRegressor& m) {
            return m.feature_importances(); })
        .def("init_noise_modulation", &GeoXGBRegressor::init_noise_modulation)
        .def("n_train",              &GeoXGBRegressor::n_train)
        .def("n_init_reduced",       &GeoXGBRegressor::n_init_reduced)
        .def("__repr__", [](const GeoXGBRegressor& r) {
            return std::string("<CppGeoXGBRegressor fitted=") +
                   (r.is_fitted() ? "True" : "False") + ">";
        })
        .def(py::pickle(
            [](const GeoXGBRegressor& self) {
                auto bytes = self.to_bytes();
                return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            },
            [](py::bytes data) {
                std::string s = data.cast<std::string>();
                std::vector<uint8_t> bytes(s.begin(), s.end());
                GeoXGBRegressor obj;
                obj.from_bytes(bytes);
                return obj;
            }
        ));

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
        // ── Geometry / interpretability ──────────────────────────────────────
        .def("X_z",               [](const GeoXGBClassifier& m) -> Eigen::MatrixXd {
            return m.X_z(); })
        .def("partition_ids",     [](const GeoXGBClassifier& m) -> Eigen::VectorXi {
            return m.partition_ids(); })
        .def("train_predictions", [](const GeoXGBClassifier& m) -> Eigen::VectorXd {
            return m.train_predictions(); })
        .def("to_z",  [](const GeoXGBClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::MatrixXd {
            return m.to_z(X); })
        .def("apply", [](const GeoXGBClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::VectorXi {
            return m.apply(X); })
        .def("feature_importances", [](const GeoXGBClassifier& m) {
            return m.feature_importances(); })
        .def("init_noise_modulation", &GeoXGBClassifier::init_noise_modulation)
        .def("n_train",              &GeoXGBClassifier::n_train)
        .def("n_init_reduced",       &GeoXGBClassifier::n_init_reduced)
        .def("__repr__", [](const GeoXGBClassifier& c) {
            return std::string("<CppGeoXGBClassifier fitted=") +
                   (c.is_fitted() ? "True" : "False") + ">";
        })
        .def(py::pickle(
            [](const GeoXGBClassifier& self) {
                auto bytes = self.to_bytes();
                return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            },
            [](py::bytes data) {
                std::string s = data.cast<std::string>();
                std::vector<uint8_t> bytes(s.begin(), s.end());
                GeoXGBClassifier obj;
                obj.from_bytes(bytes);
                return obj;
            }
        ));

    // ── Gini Classifier ──────────────────────────────────────────────────────
    py::class_<GeoXGBGiniClassifier>(m, "CppGeoXGBGiniClassifier")
        .def(py::init<GeoXGBConfig>(), py::arg("cfg") = GeoXGBConfig{})
        .def("fit",
             [](GeoXGBGiniClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X,
                const Eigen::Ref<const Eigen::VectorXd>& y) -> GeoXGBGiniClassifier& {
                 return self.fit(X, y);
             },
             py::arg("X"), py::arg("y"),
             py::return_value_policy::reference,
             "Fit the Gini classifier. y must be float-encoded 0/1.")
        .def("predict",
             [](const GeoXGBGiniClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.predict(X);
             },
             py::arg("X"),
             "Hard class labels (0 or 1). Returns ndarray (n,) int.")
        .def("predict_proba",
             [](const GeoXGBGiniClassifier& self,
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
        .def("is_fitted",             &GeoXGBGiniClassifier::is_fitted)
        .def("convergence_round",     &GeoXGBGiniClassifier::convergence_round)
        .def("last_noise_modulation", &GeoXGBGiniClassifier::last_noise_modulation)
        .def("rho_trace",    [](const GeoXGBGiniClassifier& c) {
            return std::vector<double>(c.rho_trace());
        })
        .def("yw_eff_trace", [](const GeoXGBGiniClassifier& c) {
            return std::vector<double>(c.yw_eff_trace());
        })
        .def("X_z",               [](const GeoXGBGiniClassifier& m) -> Eigen::MatrixXd {
            return m.X_z(); })
        .def("partition_ids",     [](const GeoXGBGiniClassifier& m) -> Eigen::VectorXi {
            return m.partition_ids(); })
        .def("train_predictions", [](const GeoXGBGiniClassifier& m) -> Eigen::VectorXd {
            return m.train_predictions(); })
        .def("to_z",  [](const GeoXGBGiniClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::MatrixXd {
            return m.to_z(X); })
        .def("apply", [](const GeoXGBGiniClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::VectorXi {
            return m.apply(X); })
        .def("feature_importances", [](const GeoXGBGiniClassifier& m) {
            return m.feature_importances(); })
        .def("init_noise_modulation", &GeoXGBGiniClassifier::init_noise_modulation)
        .def("n_train",              &GeoXGBGiniClassifier::n_train)
        .def("n_init_reduced",       &GeoXGBGiniClassifier::n_init_reduced)
        .def("__repr__", [](const GeoXGBGiniClassifier& c) {
            return std::string("<CppGeoXGBGiniClassifier fitted=") +
                   (c.is_fitted() ? "True" : "False") + ">";
        })
        .def(py::pickle(
            [](const GeoXGBGiniClassifier& self) {
                auto bytes = self.to_bytes();
                return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            },
            [](py::bytes data) {
                std::string s = data.cast<std::string>();
                std::vector<uint8_t> bytes(s.begin(), s.end());
                GeoXGBGiniClassifier obj;
                obj.from_bytes(bytes);
                return obj;
            }
        ));

    // ── Focal Classifier ─────────────────────────────────────────────────────
    py::class_<GeoXGBFocalClassifier>(m, "CppGeoXGBFocalClassifier")
        .def(py::init<GeoXGBConfig, double>(),
             py::arg("cfg") = GeoXGBConfig{}, py::arg("gamma") = 2.0)
        .def("fit",
             [](GeoXGBFocalClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X,
                const Eigen::Ref<const Eigen::VectorXd>& y) -> GeoXGBFocalClassifier& {
                 return self.fit(X, y);
             },
             py::arg("X"), py::arg("y"),
             py::return_value_policy::reference)
        .def("predict",
             [](const GeoXGBFocalClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) { return self.predict(X); },
             py::arg("X"))
        .def("predict_proba",
             [](const GeoXGBFocalClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 Eigen::VectorXd p1 = self.predict_proba(X);
                 const int n = static_cast<int>(p1.size());
                 Eigen::MatrixXd out(n, 2);
                 out.col(0) = 1.0 - p1.array();
                 out.col(1) = p1;
                 return out;
             },
             py::arg("X"))
        .def("is_fitted",             &GeoXGBFocalClassifier::is_fitted)
        .def("convergence_round",     &GeoXGBFocalClassifier::convergence_round)
        .def("last_noise_modulation", &GeoXGBFocalClassifier::last_noise_modulation)
        .def("gamma",                 &GeoXGBFocalClassifier::gamma)
        .def("X_z",               [](const GeoXGBFocalClassifier& m) -> Eigen::MatrixXd {
            return m.X_z(); })
        .def("partition_ids",     [](const GeoXGBFocalClassifier& m) -> Eigen::VectorXi {
            return m.partition_ids(); })
        .def("train_predictions", [](const GeoXGBFocalClassifier& m) -> Eigen::VectorXd {
            return m.train_predictions(); })
        .def("to_z",  [](const GeoXGBFocalClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::MatrixXd {
            return m.to_z(X); })
        .def("apply", [](const GeoXGBFocalClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::VectorXi {
            return m.apply(X); })
        .def("feature_importances", [](const GeoXGBFocalClassifier& m) {
            return m.feature_importances(); })
        .def("init_noise_modulation", &GeoXGBFocalClassifier::init_noise_modulation)
        .def("n_train",              &GeoXGBFocalClassifier::n_train)
        .def("n_init_reduced",       &GeoXGBFocalClassifier::n_init_reduced)
        .def("__repr__", [](const GeoXGBFocalClassifier& c) {
            return std::string("<CppGeoXGBFocalClassifier fitted=") +
                   (c.is_fitted() ? "True" : "False") +
                   " gamma=" + std::to_string(c.gamma()) + ">";
        })
        .def(py::pickle(
            [](const GeoXGBFocalClassifier& self) {
                auto bytes = self.to_bytes();
                // Append gamma as extra 8 bytes
                double g = self.gamma();
                auto gp = reinterpret_cast<const char*>(&g);
                std::string s(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                s.append(gp, 8);
                return py::bytes(s);
            },
            [](py::bytes data) {
                std::string s = data.cast<std::string>();
                double gamma;
                std::memcpy(&gamma, s.data() + s.size() - 8, 8);
                std::vector<uint8_t> bytes(s.begin(), s.end() - 8);
                GeoXGBConfig cfg;
                GeoXGBFocalClassifier obj(cfg, gamma);
                obj.from_bytes(bytes);
                return obj;
            }
        ));

    // ── Exponential Classifier ────────────────────────────────────────────────
    py::class_<GeoXGBExpClassifier>(m, "CppGeoXGBExpClassifier")
        .def(py::init<GeoXGBConfig>(), py::arg("cfg") = GeoXGBConfig{})
        .def("fit",
             [](GeoXGBExpClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X,
                const Eigen::Ref<const Eigen::VectorXd>& y) -> GeoXGBExpClassifier& {
                 return self.fit(X, y);
             },
             py::arg("X"), py::arg("y"),
             py::return_value_policy::reference)
        .def("predict",
             [](const GeoXGBExpClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) { return self.predict(X); },
             py::arg("X"))
        .def("predict_proba",
             [](const GeoXGBExpClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 Eigen::VectorXd p1 = self.predict_proba(X);
                 const int n = static_cast<int>(p1.size());
                 Eigen::MatrixXd out(n, 2);
                 out.col(0) = 1.0 - p1.array();
                 out.col(1) = p1;
                 return out;
             },
             py::arg("X"))
        .def("is_fitted",             &GeoXGBExpClassifier::is_fitted)
        .def("convergence_round",     &GeoXGBExpClassifier::convergence_round)
        .def("last_noise_modulation", &GeoXGBExpClassifier::last_noise_modulation)
        .def("X_z",               [](const GeoXGBExpClassifier& m) -> Eigen::MatrixXd {
            return m.X_z(); })
        .def("partition_ids",     [](const GeoXGBExpClassifier& m) -> Eigen::VectorXi {
            return m.partition_ids(); })
        .def("train_predictions", [](const GeoXGBExpClassifier& m) -> Eigen::VectorXd {
            return m.train_predictions(); })
        .def("to_z",  [](const GeoXGBExpClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::MatrixXd {
            return m.to_z(X); })
        .def("apply", [](const GeoXGBExpClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::VectorXi {
            return m.apply(X); })
        .def("feature_importances", [](const GeoXGBExpClassifier& m) {
            return m.feature_importances(); })
        .def("init_noise_modulation", &GeoXGBExpClassifier::init_noise_modulation)
        .def("n_train",              &GeoXGBExpClassifier::n_train)
        .def("n_init_reduced",       &GeoXGBExpClassifier::n_init_reduced)
        .def("__repr__", [](const GeoXGBExpClassifier& c) {
            return std::string("<CppGeoXGBExpClassifier fitted=") +
                   (c.is_fitted() ? "True" : "False") + ">";
        })
        .def(py::pickle(
            [](const GeoXGBExpClassifier& self) {
                auto bytes = self.to_bytes();
                return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            },
            [](py::bytes data) {
                std::string s = data.cast<std::string>();
                std::vector<uint8_t> bytes(s.begin(), s.end());
                GeoXGBExpClassifier obj;
                obj.from_bytes(bytes);
                return obj;
            }
        ));

    // ── Squared Hinge Classifier ──────────────────────────────────────────────
    py::class_<GeoXGBHingeClassifier>(m, "CppGeoXGBHingeClassifier")
        .def(py::init<GeoXGBConfig>(), py::arg("cfg") = GeoXGBConfig{})
        .def("fit",
             [](GeoXGBHingeClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X,
                const Eigen::Ref<const Eigen::VectorXd>& y) -> GeoXGBHingeClassifier& {
                 return self.fit(X, y);
             },
             py::arg("X"), py::arg("y"),
             py::return_value_policy::reference)
        .def("predict",
             [](const GeoXGBHingeClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) { return self.predict(X); },
             py::arg("X"))
        .def("predict_proba",
             [](const GeoXGBHingeClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 Eigen::VectorXd p1 = self.predict_proba(X);
                 const int n = static_cast<int>(p1.size());
                 Eigen::MatrixXd out(n, 2);
                 out.col(0) = 1.0 - p1.array();
                 out.col(1) = p1;
                 return out;
             },
             py::arg("X"))
        .def("is_fitted",             &GeoXGBHingeClassifier::is_fitted)
        .def("convergence_round",     &GeoXGBHingeClassifier::convergence_round)
        .def("last_noise_modulation", &GeoXGBHingeClassifier::last_noise_modulation)
        .def("X_z",               [](const GeoXGBHingeClassifier& m) -> Eigen::MatrixXd {
            return m.X_z(); })
        .def("partition_ids",     [](const GeoXGBHingeClassifier& m) -> Eigen::VectorXi {
            return m.partition_ids(); })
        .def("train_predictions", [](const GeoXGBHingeClassifier& m) -> Eigen::VectorXd {
            return m.train_predictions(); })
        .def("to_z",  [](const GeoXGBHingeClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::MatrixXd {
            return m.to_z(X); })
        .def("apply", [](const GeoXGBHingeClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::VectorXi {
            return m.apply(X); })
        .def("feature_importances", [](const GeoXGBHingeClassifier& m) {
            return m.feature_importances(); })
        .def("init_noise_modulation", &GeoXGBHingeClassifier::init_noise_modulation)
        .def("n_train",              &GeoXGBHingeClassifier::n_train)
        .def("n_init_reduced",       &GeoXGBHingeClassifier::n_init_reduced)
        .def("__repr__", [](const GeoXGBHingeClassifier& c) {
            return std::string("<CppGeoXGBHingeClassifier fitted=") +
                   (c.is_fitted() ? "True" : "False") + ">";
        })
        .def(py::pickle(
            [](const GeoXGBHingeClassifier& self) {
                auto bytes = self.to_bytes();
                return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            },
            [](py::bytes data) {
                std::string s = data.cast<std::string>();
                std::vector<uint8_t> bytes(s.begin(), s.end());
                GeoXGBHingeClassifier obj;
                obj.from_bytes(bytes);
                return obj;
            }
        ));

    // ── Multiclass Classifier ───────────────────────────────────────────────
    py::class_<GeoXGBMulticlassClassifier>(m, "CppGeoXGBMulticlassClassifier")
        .def(py::init<GeoXGBConfig>(), py::arg("cfg") = GeoXGBConfig{})
        .def("fit",
             [](GeoXGBMulticlassClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X,
                const Eigen::Ref<const Eigen::MatrixXd>& Y,
                const Eigen::Ref<const Eigen::VectorXd>& class_weights)
                    -> GeoXGBMulticlassClassifier& {
                 return self.fit(X, Y, class_weights);
             },
             py::arg("X"), py::arg("Y"), py::arg("class_weights"),
             py::return_value_policy::reference,
             "Fit multiclass classifier. Y: (n, K) one-hot, class_weights: (K,).")
        .def("predict",
             [](const GeoXGBMulticlassClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.predict_multi(X);
             },
             py::arg("X"),
             "Argmax class labels. Returns ndarray (n,) int.")
        .def("predict_proba",
             [](const GeoXGBMulticlassClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.predict_proba_multi(X);
             },
             py::arg("X"),
             "Softmax probabilities. Returns ndarray (n, K).")
        .def("predict_raw",
             [](const GeoXGBMulticlassClassifier& self,
                const Eigen::Ref<const Eigen::MatrixXd>& X) {
                 return self.predict_raw_multi(X);
             },
             py::arg("X"),
             "Raw logits. Returns ndarray (n, K).")
        .def("is_fitted",             &GeoXGBMulticlassClassifier::is_fitted)
        .def("convergence_round",     &GeoXGBMulticlassClassifier::convergence_round)
        .def("last_noise_modulation", &GeoXGBMulticlassClassifier::last_noise_modulation)
        .def("n_classes",             &GeoXGBMulticlassClassifier::n_classes)
        .def("X_z",               [](const GeoXGBMulticlassClassifier& m) -> Eigen::MatrixXd {
            return m.X_z(); })
        .def("partition_ids",     [](const GeoXGBMulticlassClassifier& m) -> Eigen::VectorXi {
            return m.partition_ids(); })
        .def("to_z",  [](const GeoXGBMulticlassClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::MatrixXd {
            return m.to_z(X); })
        .def("apply", [](const GeoXGBMulticlassClassifier& m,
                          const Eigen::Ref<const Eigen::MatrixXd>& X) -> Eigen::VectorXi {
            return m.apply(X); })
        .def("feature_importances", [](const GeoXGBMulticlassClassifier& m) {
            return m.feature_importances_multi(); })
        .def("train_predictions_multi", [](const GeoXGBMulticlassClassifier& m) -> Eigen::MatrixXd {
            return m.train_predictions_multi(); })
        .def("init_noise_modulation", &GeoXGBMulticlassClassifier::init_noise_modulation)
        .def("n_train",              &GeoXGBMulticlassClassifier::n_train)
        .def("n_init_reduced",       &GeoXGBMulticlassClassifier::n_init_reduced)
        .def("__repr__", [](const GeoXGBMulticlassClassifier& m) {
            return std::string("<CppGeoXGBMulticlassClassifier fitted=") +
                   (m.is_fitted() ? "True" : "False") +
                   " K=" + std::to_string(m.n_classes()) + ">";
        });
}
