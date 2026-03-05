#include "geoxgb/geoxgb_multiclass.h"
#include "geoxgb/noise.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace geoxgb {

// ── Sigmoid helper ───────────────────────────────────────────────────────────

static inline double sigmoid(double v) {
    return 1.0 / (1.0 + std::exp(-v));
}

// ── fit ──────────────────────────────────────────────────────────────────────

GeoXGBMulticlassClassifier& GeoXGBMulticlassClassifier::fit(
        const Eigen::MatrixXd& X_arg,
        const Eigen::MatrixXd& Y_arg,
        const Eigen::VectorXd& class_weights)
{
    const int n_arg = static_cast<int>(X_arg.rows());
    const int d     = static_cast<int>(X_arg.cols());
    const int K     = static_cast<int>(Y_arg.cols());
    n_classes_      = K;

    // Store class weights
    if (class_weights.size() == K) {
        class_weights_ = class_weights;
    } else {
        class_weights_ = Eigen::VectorXd::Ones(K);
    }

    // ── Block cycling setup (same as GeoXGBBase::fit_boosting) ───────────────
    const bool block_cycle = (cfg_.sample_block_n > 0 && n_arg > cfg_.sample_block_n);
    const int  blk_sz      = block_cycle ? cfg_.sample_block_n : n_arg;

    Eigen::MatrixXd X_cur;
    Eigen::MatrixXd Y_cur;  // n × K

    std::vector<int> blk_perm;
    int blk_ctr = 0, blk_epoch = 0, n_usable_blocks = 1;

    auto lcg_permute = [&](int epoch_seed) {
        std::iota(blk_perm.begin(), blk_perm.end(), 0);
        uint64_t lcg = static_cast<uint64_t>(cfg_.random_state + epoch_seed)
                       * 6364136223846793005ULL + 1442695040888963407ULL;
        for (int i = 0; i < n_arg - 1; ++i) {
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = i + static_cast<int>((lcg >> 33)
                        % static_cast<uint64_t>(n_arg - i));
            std::swap(blk_perm[i], blk_perm[j]);
        }
    };

    auto slice_block = [&]() {
        const int start = blk_ctr * blk_sz;
        X_cur.resize(blk_sz, d);
        Y_cur.resize(blk_sz, K);
        for (int i = 0; i < blk_sz; ++i) {
            X_cur.row(i) = X_arg.row(blk_perm[start + i]);
            Y_cur.row(i) = Y_arg.row(blk_perm[start + i]);
        }
    };

    auto advance_block = [&]() {
        blk_ctr++;
        if (blk_ctr >= n_usable_blocks) {
            blk_ctr = 0;
            blk_epoch++;
            lcg_permute(blk_epoch);
        }
        slice_block();
    };

    if (block_cycle) {
        const int n_total = n_arg / blk_sz;
        n_usable_blocks = std::max(1, n_total - (cfg_.leave_last_block_out ? 1 : 0));
        blk_perm.resize(n_arg);
        lcg_permute(0);
        slice_block();
    } else {
        X_cur = X_arg;
        Y_cur = Y_arg;
    }

    int n = blk_sz;

    // ── Init storage ─────────────────────────────────────────────────────────
    mc_trees_.resize(K);
    for (int k = 0; k < K; ++k) {
        mc_trees_[k].clear();
        mc_trees_[k].reserve(cfg_.n_rounds);
    }
    trees_.clear();  // not used by multiclass, but clear inherited member
    convergence_round_ = -1;
    rho_trace_.clear();
    yw_eff_trace_.clear();
    convergence_losses_.clear();
    n_train_arg_ = n_arg;

    // ── Binner ───────────────────────────────────────────────────────────────
    cont_cols_full_.resize(d);
    std::iota(cont_cols_full_.begin(), cont_cols_full_.end(), 0);
    full_binner_.fit(block_cycle ? X_arg : X_cur, cfg_.n_bins);
    X_bin_full_ = full_binner_.transform(X_cur);
    gbt_bin_edges_ = full_binner_.edges();

    // ── Per-class init predictions (log-odds of class frequency) ─────────────
    init_preds_.resize(K);
    for (int k = 0; k < K; ++k) {
        double p = Y_cur.col(k).mean();
        p = std::max(1e-7, std::min(1.0 - 1e-7, p));
        init_preds_[k] = std::log(p / (1.0 - p));
    }
    init_pred_ = 0.0;  // unused for multiclass

    // Predictions: n × K on full block, n_r × K on reduced set
    Eigen::MatrixXd preds_X(n, K);
    for (int k = 0; k < K; ++k)
        preds_X.col(k).setConstant(init_preds_[k]);

    // ── Combined gradient signal for initial HVRT ────────────────────────────
    // Use class 0's binary target for the initial HVRT fit.
    // The first refit will switch to combined gradient magnitude.
    Eigen::VectorXd y_combined = Y_cur.col(0).cast<double>();

    // ── Initial resample (shared geometry) ───────────────────────────────────
    std::shared_ptr<hvrt::HVRT> last_hvrt;
    ResampleResult res = do_resample(X_cur, y_combined, 0, last_hvrt);

    Eigen::MatrixXd Xr       = res.X;
    Eigen::VectorXi red_idx  = res.red_idx;
    double last_noise_mod    = res.noise_mod;
    last_noise_mod_          = last_noise_mod;
    init_noise_mod_          = last_noise_mod;
    n_init_reduced_          = static_cast<int>(res.red_idx.size());
    int last_n_expanded      = res.n_expanded;

    const int n_red = static_cast<int>(red_idx.size());

    // ── Build per-class targets on reduced set ───────────────────────────────
    Eigen::MatrixXd Y_r(Xr.rows(), K);
    for (int k = 0; k < K; ++k) {
        // Real samples: copy from Y_cur
        for (int i = 0; i < n_red; ++i)
            Y_r(i, k) = Y_cur(red_idx[i], k);
        // Synthetic samples: kNN-assign from real
        if (last_n_expanded > 0) {
            Eigen::MatrixXd X_red_z(n_red, last_hvrt->X_z().cols());
            for (int i = 0; i < n_red; ++i)
                X_red_z.row(i) = last_hvrt->X_z().row(red_idx[i]);
            Eigen::MatrixXd X_syn_z = last_hvrt->to_z(Xr.bottomRows(last_n_expanded));
            Eigen::VectorXd y_red_k(n_red);
            for (int i = 0; i < n_red; ++i) y_red_k[i] = Y_r(i, k);
            Eigen::VectorXd y_syn_k = knn_assign_y(X_syn_z, X_red_z, y_red_k);
            for (int i = 0; i < last_n_expanded; ++i)
                Y_r(n_red + i, k) = y_syn_k[i];
        }
    }

    // Per-class predictions on reduced set
    Eigen::MatrixXd preds_r(Xr.rows(), K);
    for (int k = 0; k < K; ++k)
        preds_r.col(k).setConstant(init_preds_[k]);

    bool expansion_risk = cfg_.auto_expand && (n < cfg_.min_train_samples);

    // Binned reduced set cache
    Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X_bin_r;
    bool xr_changed = true;

    // Sync preds helper: update preds_r from preds_X for all K classes
    auto sync_preds_multi = [&](int n_exp) {
        const int nr = static_cast<int>(red_idx.size());
        if (n_exp == 0) {
            preds_r.resize(nr, K);
            for (int i = 0; i < nr; ++i)
                preds_r.row(i) = preds_X.row(red_idx[i]);
        } else {
            const int n_syn = static_cast<int>(Xr.rows()) - nr;
            preds_r.resize(nr + n_syn, K);
            for (int i = 0; i < nr; ++i)
                preds_r.row(i) = preds_X.row(red_idx[i]);
            if (n_syn > 0) {
                Eigen::MatrixXd X_syn = Xr.bottomRows(n_syn);
                for (int k = 0; k < K; ++k) {
                    Eigen::VectorXd p = Eigen::VectorXd::Constant(n_syn, init_preds_[k]);
                    for (const auto& wl : mc_trees_[k]) {
                        Eigen::VectorXd tmp = wl.tree.predict(X_syn);
                        p.noalias() += wl.lr * tmp;
                    }
                    preds_r.col(k).tail(n_syn) = p;
                }
            }
        }
    };

    // ── Boosting loop ────────────────────────────────────────────────────────
    bool converged_early = false;
    for (int i = 0; i < cfg_.n_rounds; ++i) {

        // ── Refit ────────────────────────────────────────────────────────────
        if (cfg_.refit_interval > 0 && i > 0 && (i % cfg_.refit_interval == 0)) {
            bool block_just_advanced = false;
            if (block_cycle) {
                advance_block();
                n = blk_sz;
                X_bin_full_ = full_binner_.transform(X_cur);
                xr_changed  = true;
                last_hvrt   = nullptr;
                // Recompute preds_X on new block
                preds_X.resize(n, K);
                for (int k = 0; k < K; ++k) {
                    Eigen::VectorXd p = Eigen::VectorXd::Constant(n, init_preds_[k]);
                    for (const auto& wl : mc_trees_[k]) {
                        Eigen::VectorXd tmp = wl.tree.predict(X_cur);
                        p.noalias() += wl.lr * tmp;
                    }
                    preds_X.col(k) = p;
                }
                expansion_risk = cfg_.auto_expand && (n < cfg_.min_train_samples);
                block_just_advanced = true;
                last_noise_mod = 1.0;
            }

            // Compute K gradient vectors and combined signal
            Eigen::MatrixXd grads_orig(n, K);
            Eigen::VectorXd y_comb(n);
            for (int j = 0; j < n; ++j) {
                double sq = 0.0;
                for (int k = 0; k < K; ++k) {
                    double g = Y_cur(j, k) - sigmoid(preds_X(j, k));
                    if (class_weights_[k] != 1.0 && Y_cur(j, k) > 0.5)
                        g *= class_weights_[k];
                    grads_orig(j, k) = g;
                    sq += g * g;
                }
                y_comb[j] = std::sqrt(sq);
            }

            // Convergence check on combined signal
            if (cfg_.convergence_tol > 0.0) {
                double loss_now = y_comb.mean();
                convergence_losses_.push_back(loss_now);
                if (convergence_losses_.size() >= 3) {
                    double prev = convergence_losses_[convergence_losses_.size() - 3];
                    double rel  = (prev - loss_now) / (prev + 1e-12);
                    if (rel < cfg_.convergence_tol) {
                        convergence_round_ = i;
                        converged_early    = true;
                    }
                }
            }
            if (converged_early) break;

            bool skip_refit = !block_just_advanced
                              && cfg_.noise_guard && cfg_.auto_noise
                              && expansion_risk
                              && (last_noise_mod < cfg_.refit_noise_floor);

            if (!skip_refit) {
                // Shared resample with combined gradient signal
                ResampleResult new_res = do_resample(X_cur, y_comb, i, last_hvrt);
                last_noise_mod  = new_res.noise_mod;
                last_noise_mod_ = last_noise_mod;

                bool finite_ok = new_res.X.allFinite() && new_res.y.allFinite();
                bool discard = !finite_ok
                               || (cfg_.noise_guard && cfg_.auto_noise
                                   && expansion_risk
                                   && (new_res.noise_mod < cfg_.refit_noise_floor));

                if (!discard) {
                    Xr       = new_res.X;
                    red_idx  = new_res.red_idx;
                    last_n_expanded = new_res.n_expanded;
                    xr_changed = true;

                    // Build per-class Y_r
                    const int nr2 = static_cast<int>(red_idx.size());
                    Y_r.resize(Xr.rows(), K);
                    for (int k = 0; k < K; ++k) {
                        for (int j = 0; j < nr2; ++j)
                            Y_r(j, k) = grads_orig(red_idx[j], k);
                    }

                    // kNN-assign synthetic y for each class (shared distance matrix)
                    if (new_res.n_expanded > 0 && last_hvrt && last_hvrt->fitted()) {
                        Eigen::MatrixXd X_red_z(nr2, last_hvrt->X_z().cols());
                        for (int j = 0; j < nr2; ++j)
                            X_red_z.row(j) = last_hvrt->X_z().row(red_idx[j]);
                        Eigen::MatrixXd X_syn_z = last_hvrt->to_z(
                            Xr.bottomRows(new_res.n_expanded));
                        for (int k = 0; k < K; ++k) {
                            Eigen::VectorXd y_red_k(nr2);
                            for (int j = 0; j < nr2; ++j)
                                y_red_k[j] = Y_r(j, k);
                            Eigen::VectorXd y_syn_k = knn_assign_y(
                                X_syn_z, X_red_z, y_red_k);
                            for (int j = 0; j < new_res.n_expanded; ++j)
                                Y_r(nr2 + j, k) = y_syn_k[j];
                        }
                    }

                    // Sync predictions and reconstruct targets
                    sync_preds_multi(new_res.n_expanded);
                    // Y_r currently holds gradients; reconstruct targets
                    for (int k = 0; k < K; ++k) {
                        for (int j = 0; j < static_cast<int>(Xr.rows()); ++j) {
                            Y_r(j, k) = sigmoid(preds_r(j, k)) + Y_r(j, k);
                        }
                    }
                } else if (block_just_advanced) {
                    // Recompute preds on existing Xr
                    preds_r.resize(Xr.rows(), K);
                    for (int k = 0; k < K; ++k) {
                        Eigen::VectorXd p = Eigen::VectorXd::Constant(
                            static_cast<int>(Xr.rows()), init_preds_[k]);
                        for (const auto& wl : mc_trees_[k]) {
                            Eigen::VectorXd tmp = wl.tree.predict(Xr);
                            p.noalias() += wl.lr * tmp;
                        }
                        preds_r.col(k) = p;
                    }
                } else {
                    sync_preds_multi(last_n_expanded);
                }
            } else {
                sync_preds_multi(last_n_expanded);
            }
        }

        // ── Bin reduced set ──────────────────────────────────────────────────
        if (xr_changed) {
            if (last_n_expanded == 0) {
                const int nr = static_cast<int>(red_idx.size());
                X_bin_r.resize(nr, d);
                for (int s = 0; s < nr; ++s)
                    X_bin_r.row(s) = X_bin_full_.row(red_idx[s]);
            } else {
                X_bin_r = full_binner_.transform(Xr);
            }
            xr_changed = false;
        }

        // ── Fit K trees (one per class) ──────────────────────────────────────
        hvrt::HVRTConfig tcfg;
        tcfg.n_partitions     = 2 * (1 << cfg_.max_depth);
        tcfg.min_samples_leaf = cfg_.min_samples_leaf;
        tcfg.max_depth        = cfg_.max_depth;
        tcfg.n_bins           = cfg_.n_bins;
        tcfg.auto_tune        = false;
        tcfg.split_strategy   = hvrt::SplitStrategy::Random;

        for (int k = 0; k < K; ++k) {
            // Compute per-class gradient
            Eigen::VectorXd grads_k(Xr.rows());
            for (int j = 0; j < static_cast<int>(Xr.rows()); ++j) {
                grads_k[j] = Y_r(j, k) - sigmoid(preds_r(j, k));
                if (class_weights_[k] != 1.0 && Y_r(j, k) > 0.5)
                    grads_k[j] *= class_weights_[k];
            }
            if (!grads_k.allFinite()) grads_k.setZero();

            tcfg.random_state = cfg_.random_state + i * K + k;

            WeakLearner wl;
            wl.cont_cols = cont_cols_full_;
            wl.lr        = cfg_.learning_rate;
            wl.tree.inject_bin_edges(gbt_bin_edges_, cont_cols_full_, cfg_.n_bins);
            wl.tree.build(Xr, X_bin_r, cont_cols_full_, {}, grads_k, tcfg);

            // Accumulate predictions
            Eigen::VectorXd lp_Xr = wl.tree.predict(Xr);
            Eigen::VectorXd lp_X  = wl.tree.predict(X_cur);
            preds_r.col(k) += cfg_.learning_rate * lp_Xr;
            preds_X.col(k) += cfg_.learning_rate * lp_X;

            mc_trees_[k].push_back(std::move(wl));
        }
    }

    fitted_ = true;

    // Persist geometry
    if (last_hvrt && last_hvrt->fitted()) {
        last_hvrt_     = last_hvrt;
        X_z_           = last_hvrt->X_z();
        partition_ids_ = last_hvrt->partition_ids();
        train_predictions_ = Eigen::VectorXd();  // scalar not used for multiclass
    }

    // Per-class training predictions on current block
    train_predictions_multi_ = preds_X;  // (n × K) raw logits

    return *this;
}

// ── predict_raw_multi ────────────────────────────────────────────────────────

Eigen::MatrixXd GeoXGBMulticlassClassifier::predict_raw_multi(
        const Eigen::MatrixXd& X) const
{
    if (!fitted_) throw std::runtime_error("Model not fitted");
    const int n = static_cast<int>(X.rows());
    Eigen::MatrixXd logits(n, n_classes_);
    for (int k = 0; k < n_classes_; ++k) {
        Eigen::VectorXd p = Eigen::VectorXd::Constant(n, init_preds_[k]);
        Eigen::VectorXd tmp(n);
        for (const auto& wl : mc_trees_[k]) {
            wl.tree.predict_into(X, tmp);
            p.noalias() += wl.lr * tmp;
        }
        logits.col(k) = p;
    }
    return logits;
}

// ── predict_proba_multi ──────────────────────────────────────────────────────

Eigen::MatrixXd GeoXGBMulticlassClassifier::predict_proba_multi(
        const Eigen::MatrixXd& X) const
{
    Eigen::MatrixXd logits = predict_raw_multi(X);
    // Softmax: subtract row-max for stability, exp, normalise
    Eigen::VectorXd maxes = logits.rowwise().maxCoeff();
    logits.colwise() -= maxes;
    logits = logits.array().exp().matrix();
    Eigen::VectorXd sums = logits.rowwise().sum();
    for (int k = 0; k < n_classes_; ++k)
        logits.col(k).array() /= sums.array();
    return logits;
}

// ── predict_multi ────────────────────────────────────────────────────────────

Eigen::VectorXi GeoXGBMulticlassClassifier::predict_multi(
        const Eigen::MatrixXd& X) const
{
    Eigen::MatrixXd proba = predict_proba_multi(X);
    const int n = static_cast<int>(proba.rows());
    Eigen::VectorXi labels(n);
    for (int i = 0; i < n; ++i) {
        int best = 0;
        double best_v = proba(i, 0);
        for (int k = 1; k < n_classes_; ++k) {
            if (proba(i, k) > best_v) {
                best_v = proba(i, k);
                best = k;
            }
        }
        labels[i] = best;
    }
    return labels;
}

// ── feature_importances_multi ────────────────────────────────────────────────

std::vector<double> GeoXGBMulticlassClassifier::feature_importances_multi() const {
    if (mc_trees_.empty()) return {};
    const int d = static_cast<int>(cont_cols_full_.size());
    std::vector<double> total(d, 0.0);
    for (int k = 0; k < n_classes_; ++k) {
        for (const auto& wl : mc_trees_[k]) {
            const auto& fi = wl.tree.feature_importances();
            const int nd = static_cast<int>(std::min(fi.size(), total.size()));
            for (int i = 0; i < nd; ++i)
                total[i] += fi[i];
        }
    }
    double s = 0.0;
    for (double v : total) s += v;
    if (s > 0.0) for (double& v : total) v /= s;
    return total;
}

} // namespace geoxgb
