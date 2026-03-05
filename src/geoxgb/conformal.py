"""
Conformal prediction wrapper for GeoXGB.

Provides calibrated prediction intervals using:
  - Per-HVRT-partition residual standard deviation (local noise structure)
  - z-space k-NN density (geometric coverage uncertainty)

The conformal guarantee: if calibration and test data are exchangeable with
training data, P(y ∈ interval) >= 1 − α exactly (split conformal, Venn–Abers).

Usage
-----
from geoxgb.conformal import ConformalGeoXGBRegressor, ConformalGeoXGBClassifier

# Regression
model = ConformalGeoXGBRegressor(n_rounds=500, learning_rate=0.1, max_depth=3,
                                  min_samples_leaf=5, reduce_ratio=0.7,
                                  y_weight=0.2, refit_interval=5,
                                  auto_expand=True, expand_ratio=0.1,
                                  min_train_samples=100, n_bins=64)
model.fit(X_train, y_train)
model.calibrate(X_cal, y_cal)           # held-out calibration set
lo, hi = model.predict_interval(X_test, alpha=0.10)   # 90% intervals
model.coverage_summary(X_test, y_test)  # empirical coverage report

# Using train/cal split internally:
model.fit_calibrate(X_train, y_train, cal_fraction=0.2)

Uncertainty signal
------------------
sigma(x) = sigma_partition(x) × (1 + beta × density_rank(x))

  sigma_partition  — std of calibration residuals in x's HVRT partition.
                     Falls back to global calibration std if the partition
                     has < min_partition_cal samples.
  density_rank     — percentile of x's mean z-space k-NN distance relative
                     to calibration data (0 = same density as training,
                     1 = fully out-of-distribution). beta=0 disables.

Partition assignment uses tree_.apply(X) on the last-fitted HVRT.
z-space embedding uses _to_z(X) (the HVRT whitening transform).
"""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors

EPS = 1e-9


# ── Internal sigma estimator ───────────────────────────────────────────────────

class _SigmaEstimator:
    """
    Combines partition-level residual std with z-space density scaling.
    Fitted on calibration data; applied to arbitrary test data.
    """

    def __init__(self, model, beta: float = 0.10, k_density: int = 5,
                 min_partition_cal: int = 5):
        self._model = model
        self._beta  = beta
        self._k     = k_density
        self._min_p = min_partition_cal

        self._global_sigma: float = 1.0
        self._partition_sigma: dict[int, float] = {}

        # For density scaling
        self._z_cal: np.ndarray | None = None
        self._cal_nn_dist: np.ndarray | None = None   # (n_cal,) mean k-NN dist
        self._nn_model: NearestNeighbors | None = None

    def fit(self, X_cal: np.ndarray, resids_cal: np.ndarray) -> "_SigmaEstimator":
        """Fit sigma estimator from calibration residuals."""
        # ── Partition-level sigma ──────────────────────────────────────────────
        try:
            pids = np.asarray(self._model.tree_.apply(X_cal), dtype=int)
            self._global_sigma = max(float(np.std(resids_cal)), EPS)
            for pid in np.unique(pids):
                mask = pids == pid
                if mask.sum() >= self._min_p:
                    self._partition_sigma[pid] = max(float(np.std(resids_cal[mask])), EPS)
        except Exception:
            self._global_sigma = max(float(np.std(resids_cal)), EPS)

        # ── z-space density ────────────────────────────────────────────────────
        if self._beta > 0:
            try:
                self._z_cal = np.asarray(self._model._to_z(X_cal))
                nn = NearestNeighbors(n_neighbors=min(self._k, len(X_cal) - 1),
                                      algorithm="auto")
                nn.fit(self._z_cal)
                dists, _ = nn.kneighbors(self._z_cal)
                self._cal_nn_dist = dists.mean(axis=1)  # (n_cal,) reference dist
                self._nn_model = nn
            except Exception:
                self._beta = 0.0   # disable density scaling if z unavailable

        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample sigma array of shape (n,)."""
        n = len(X)
        sigmas = np.full(n, self._global_sigma)

        # Partition sigma
        try:
            pids = np.asarray(self._model.tree_.apply(X), dtype=int)
            for i, pid in enumerate(pids):
                sigmas[i] = self._partition_sigma.get(int(pid), self._global_sigma)
        except Exception:
            pass

        # z-space density scaling
        if self._beta > 0 and self._nn_model is not None and self._z_cal is not None:
            try:
                z_test = np.asarray(self._model._to_z(X))
                dists, _ = self._nn_model.kneighbors(z_test)
                mean_dist = dists.mean(axis=1)    # (n,)
                # Rank each test point against calibration distances (0→dense, 1→OOD)
                rank = np.mean(mean_dist[:, None] > self._cal_nn_dist[None, :], axis=1)
                sigmas = sigmas * (1.0 + self._beta * rank)
            except Exception:
                pass

        return sigmas


# ── Conformal regressor ────────────────────────────────────────────────────────

class ConformalGeoXGBRegressor:
    """
    Split-conformal prediction intervals for GeoXGB regression.

    Parameters
    ----------
    beta : float
        Weight for z-space density scaling of sigma (0 = partition-only).
    k_density : int
        k for z-space k-NN density estimate.
    min_partition_cal : int
        Minimum calibration samples in a partition to use its sigma
        (falls back to global sigma otherwise).
    **geo_kwargs :
        All GeoXGB hyperparameters forwarded to make_cpp_config().
    """

    def __init__(self, beta: float = 0.10, k_density: int = 5,
                 min_partition_cal: int = 5, **geo_kwargs):
        self._geo_kwargs      = geo_kwargs
        self._beta            = beta
        self._k_density       = k_density
        self._min_p           = min_partition_cal
        self._model           = None
        self._sigma_est       = None
        self._cal_scores      = None   # nonconformity scores |resid|/sigma
        self._n_cal           = 0
        self._fitted          = False
        self._calibrated      = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalGeoXGBRegressor":
        """Fit the underlying GeoXGB model."""
        from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor
        cfg = make_cpp_config(**self._geo_kwargs)
        self._model = CppGeoXGBRegressor(cfg)
        self._model.fit(np.asarray(X, dtype=np.float64),
                        np.asarray(y, dtype=np.float64))
        self._fitted = True
        return self

    def calibrate(self, X_cal: np.ndarray,
                  y_cal: np.ndarray) -> "ConformalGeoXGBRegressor":
        """
        Fit sigma estimator and compute nonconformity scores on held-out data.
        Must be called after fit().
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before calibrate().")

        X_cal = np.asarray(X_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.float64)

        preds_cal = self._model.predict(X_cal)
        resids_cal = y_cal - preds_cal

        self._sigma_est = _SigmaEstimator(
            self._model, beta=self._beta,
            k_density=self._k_density, min_partition_cal=self._min_p,
        ).fit(X_cal, resids_cal)

        sigmas_cal = self._sigma_est(X_cal)
        self._cal_scores = np.abs(resids_cal) / (sigmas_cal + EPS)
        self._n_cal      = len(self._cal_scores)
        self._calibrated = True
        return self

    def fit_calibrate(self, X: np.ndarray, y: np.ndarray,
                      cal_fraction: float = 0.20,
                      random_state: int = 42) -> "ConformalGeoXGBRegressor":
        """
        Convenience: split X/y into train/cal, fit on train, calibrate on cal.
        cal_fraction of data is held out for calibration.
        """
        rng = np.random.default_rng(random_state)
        n   = len(y)
        idx = rng.permutation(n)
        n_cal = max(int(n * cal_fraction), 10)
        cal_idx, tr_idx = idx[:n_cal], idx[n_cal:]
        self.fit(X[tr_idx], y[tr_idx])
        self.calibrate(X[cal_idx], y[cal_idx])
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point predictions (no intervals)."""
        return self._model.predict(np.asarray(X, dtype=np.float64))

    def predict_interval(self, X: np.ndarray,
                         alpha: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
        """
        Return calibrated (1−alpha) prediction intervals as (lower, upper).

        The conformal quantile q = quantile(cal_scores, (1−α)(1+1/n_cal))
        ensures P(y ∈ interval) >= 1−α under exchangeability.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() or fit_calibrate() before predict_interval().")

        X = np.asarray(X, dtype=np.float64)
        preds  = self._model.predict(X)
        sigmas = self._sigma_est(X)

        q_level = min((1.0 - alpha) * (1.0 + 1.0 / self._n_cal), 1.0)
        q       = float(np.quantile(self._cal_scores, q_level))

        half = q * sigmas
        return preds - half, preds + half

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (point_predictions, sigma_estimates).
        sigma is NOT the conformal half-width; multiply by the calibration
        quantile (stored in self.calibration_quantile(alpha)) to get intervals.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() first.")
        X = np.asarray(X, dtype=np.float64)
        return self._model.predict(X), self._sigma_est(X)

    def calibration_quantile(self, alpha: float = 0.10) -> float:
        """Return the conformal quantile q for a given alpha."""
        if not self._calibrated:
            raise RuntimeError("Call calibrate() first.")
        q_level = min((1.0 - alpha) * (1.0 + 1.0 / self._n_cal), 1.0)
        return float(np.quantile(self._cal_scores, q_level))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def coverage_summary(self, X_test: np.ndarray, y_test: np.ndarray,
                         alphas: list[float] | None = None) -> dict:
        """
        Report empirical coverage vs nominal (1−alpha) at several alpha levels.
        Also reports mean interval width and Spearman sharpness
        (correlation between half-width and |residual| — positive = calibrated).

        Returns a dict with per-alpha results for programmatic use.
        """
        from scipy.stats import spearmanr

        if alphas is None:
            alphas = [0.01, 0.05, 0.10, 0.20, 0.30]

        X_test  = np.asarray(X_test, dtype=np.float64)
        y_test  = np.asarray(y_test, dtype=np.float64)
        preds   = self._model.predict(X_test)
        sigmas  = self._sigma_est(X_test)
        abs_res = np.abs(y_test - preds)

        rho, _ = spearmanr(sigmas, abs_res)

        print(f"\n  Conformal coverage report")
        print(f"  n_test={len(y_test)}  n_cal={self._n_cal}  "
              f"sharpness ρ(sigma,|resid|)={rho:+.3f}")
        print(f"  {'alpha':>6}  {'nominal':>8}  {'actual':>8}  "
              f"{'mean_width':>11}  {'overcoverage':>13}")
        print("  " + "-" * 54)

        results = {}
        for alpha in alphas:
            q_level = min((1.0 - alpha) * (1.0 + 1.0 / self._n_cal), 1.0)
            q       = float(np.quantile(self._cal_scores, q_level))
            half    = q * sigmas
            covered = float((abs_res <= half).mean())
            width   = float(2.0 * half.mean())
            over    = covered - (1.0 - alpha)
            print(f"  {alpha:>6.2f}  {1-alpha:>8.3f}  {covered:>8.3f}  "
                  f"{width:>11.4f}  {over:>+13.4f}")
            results[alpha] = dict(nominal=1.0-alpha, actual=covered,
                                   mean_width=width, overcoverage=over)

        print()
        return dict(sharpness_rho=float(rho), per_alpha=results)

    def sigma_summary(self, X_test: np.ndarray) -> None:
        """Print distribution of sigma estimates for a test set (diagnostic)."""
        X_test = np.asarray(X_test, dtype=np.float64)
        sigmas = self._sigma_est(X_test)
        ps = np.percentile(sigmas, [5, 25, 50, 75, 95])
        print(f"  sigma distribution (n={len(X_test)}):  "
              f"p5={ps[0]:.4f}  p25={ps[1]:.4f}  p50={ps[2]:.4f}  "
              f"p75={ps[3]:.4f}  p95={ps[4]:.4f}  "
              f"ratio_p95/p5={ps[4]/(ps[0]+EPS):.2f}x")


# ── Conformal classifier ───────────────────────────────────────────────────────

class ConformalGeoXGBClassifier:
    """
    Split-conformal prediction sets for GeoXGB binary classification.

    predict_set(X, alpha) returns a boolean mask (n, 2) where True means
    the label is in the prediction set at coverage level (1−alpha).

    For classification the nonconformity score is 1 − p(true_class),
    and the prediction set at quantile q includes all labels with
    1 − p(label) <= q.
    """

    def __init__(self, **geo_kwargs):
        self._geo_kwargs = geo_kwargs
        self._model      = None
        self._cal_scores = None
        self._n_cal      = 0
        self._fitted     = False
        self._calibrated = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalGeoXGBClassifier":
        from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBClassifier
        cfg = make_cpp_config(**self._geo_kwargs)
        self._model = CppGeoXGBClassifier(cfg)
        self._model.fit(np.asarray(X, dtype=np.float64),
                        np.asarray(y, dtype=np.float64))
        self._fitted = True
        return self

    def calibrate(self, X_cal: np.ndarray,
                  y_cal: np.ndarray) -> "ConformalGeoXGBClassifier":
        if not self._fitted:
            raise RuntimeError("Call fit() before calibrate().")
        X_cal = np.asarray(X_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal, dtype=np.float64)

        proba = self._model.predict_proba(X_cal)   # (n, 2)
        # Nonconformity: 1 - p(true label)
        y_int = y_cal.astype(int)
        p_true = proba[np.arange(len(y_int)), y_int]
        self._cal_scores = 1.0 - p_true
        self._n_cal      = len(self._cal_scores)
        self._calibrated = True
        return self

    def fit_calibrate(self, X: np.ndarray, y: np.ndarray,
                      cal_fraction: float = 0.20,
                      random_state: int = 42) -> "ConformalGeoXGBClassifier":
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(len(y))
        n_cal = max(int(len(y) * cal_fraction), 10)
        cal_idx, tr_idx = idx[:n_cal], idx[n_cal:]
        self.fit(X[tr_idx], y[tr_idx])
        self.calibrate(X[cal_idx], y[cal_idx])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(np.asarray(X, dtype=np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(np.asarray(X, dtype=np.float64))

    def predict_set(self, X: np.ndarray,
                    alpha: float = 0.10) -> np.ndarray:
        """
        Return prediction set as boolean mask (n, 2).
        predict_set[i, c] = True means label c is in the prediction set for sample i.
        Empty set possible at small alpha; full set possible at large alpha.
        """
        if not self._calibrated:
            raise RuntimeError("Call calibrate() or fit_calibrate() first.")

        X = np.asarray(X, dtype=np.float64)
        proba   = self._model.predict_proba(X)  # (n, 2)
        q_level = min((1.0 - alpha) * (1.0 + 1.0 / self._n_cal), 1.0)
        q       = float(np.quantile(self._cal_scores, q_level))

        scores  = 1.0 - proba            # nonconformity for each label
        in_set  = scores <= q            # (n, 2) boolean
        return in_set

    def coverage_summary(self, X_test: np.ndarray, y_test: np.ndarray,
                         alphas: list[float] | None = None) -> dict:
        if alphas is None:
            alphas = [0.01, 0.05, 0.10, 0.20]

        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64).astype(int)

        print(f"\n  Conformal coverage report (classification)")
        print(f"  n_test={len(y_test)}  n_cal={self._n_cal}")
        print(f"  {'alpha':>6}  {'nominal':>8}  {'actual':>8}  {'mean_set_size':>14}")
        print("  " + "-" * 44)

        results = {}
        for alpha in alphas:
            in_set  = self.predict_set(X_test, alpha)         # (n, 2)
            covered = float(in_set[np.arange(len(y_test)), y_test].mean())
            set_sz  = float(in_set.sum(axis=1).mean())
            print(f"  {alpha:>6.2f}  {1-alpha:>8.3f}  {covered:>8.3f}  {set_sz:>14.3f}")
            results[alpha] = dict(nominal=1.0-alpha, actual=covered, mean_set_size=set_sz)

        print()
        return results
