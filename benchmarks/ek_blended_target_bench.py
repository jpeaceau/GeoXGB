"""
Benchmark: Blended HVRT target T + λ·e₃.

HVRT takes a scalar y. We pass y_signal = T + λ·e₃ (both normalized to
unit variance), so HVRT builds partitions that separate regions in BOTH
degree-2 and degree-3 structure simultaneously.

e₃ is noise-invariant at degree 3 (Newton's identity cancellation), and
preserves sign — unlike T² which is just T amplified.

Key question: does a 2-degree blended HVRT target improve the GBT that
trains within those partitions?

Also tests: what if we just give HVRT the ACTUAL y-target with higher
y_weight, letting gradient signal guide partitioning? This is the simplest
"let the data decide" baseline.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
from sklearn.datasets import (
    make_friedman1, make_friedman2, make_friedman3,
    load_diabetes, fetch_california_housing, load_wine,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from hvrt import HVRT

from geoxgb.regressor import GeoXGBRegressor
from geoxgb.experimental._e3_augment import (
    _compute_e2_scalar,
    _compute_e3_scalar,
    _compute_e4_scalar,
)


def _robust_whiten(X):
    center = np.median(X, axis=0)
    mad = np.median(np.abs(X - center), axis=0)
    mad[mad < 1e-12] = 1.0
    scale = mad * 1.4826
    return (X - center) / scale, center, scale


def _make_blended_target(Z, lam3=1.0, lam4=0.0):
    """T + λ₃·e₃_norm + λ₄·e₄_norm, each normalized to unit variance."""
    S = Z.sum(axis=1)
    Q = (Z ** 2).sum(axis=1)
    T = S ** 2 - Q

    target = T / max(np.std(T), 1e-12)

    if lam3 > 0 and Z.shape[1] >= 3:
        e3 = _compute_e3_scalar(Z)
        s3 = np.std(e3)
        if s3 > 1e-12:
            target = target + lam3 * (e3 / s3)

    if lam4 > 0 and Z.shape[1] >= 4:
        e4 = _compute_e4_scalar(Z)
        s4 = np.std(e4)
        if s4 > 1e-12:
            target = target + lam4 * (e4 / s4)

    return target


class BlendedHVRTRegressor:
    """HVRT with blended T + λ·e₃ target, then HistGBT on reduced data."""

    def __init__(self, lam3=0.0, lam4=0.0, reduce_ratio=0.8,
                 y_weight=0.25, random_state=42):
        self.lam3 = lam3
        self.lam4 = lam4
        self.reduce_ratio = reduce_ratio
        self.y_weight = y_weight
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        Z, _, _ = _robust_whiten(X)
        blended = _make_blended_target(Z, self.lam3, self.lam4)

        hvrt = HVRT(y_weight=self.y_weight, random_state=self.random_state)
        hvrt.fit(X, y=blended)

        n_keep = max(10, int(len(X) * self.reduce_ratio))
        _, red_idx = hvrt.reduce(
            n=n_keep, method='variance_ordered',
            variance_weighted=True, return_indices=True,
        )

        self._model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5,
            random_state=self.random_state,
        )
        self._model.fit(X[red_idx], y[red_idx])
        return self

    def predict(self, X):
        return self._model.predict(np.asarray(X, dtype=np.float64))


class YSignalHVRTRegressor:
    """
    HVRT with the ACTUAL y-target as signal (high y_weight).
    'Let the data decide' baseline — no hand-crafted geometric target.
    """

    def __init__(self, reduce_ratio=0.8, y_weight=0.75, random_state=42):
        self.reduce_ratio = reduce_ratio
        self.y_weight = y_weight
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        hvrt = HVRT(y_weight=self.y_weight, random_state=self.random_state)
        hvrt.fit(X, y=y)

        n_keep = max(10, int(len(X) * self.reduce_ratio))
        _, red_idx = hvrt.reduce(
            n=n_keep, method='variance_ordered',
            variance_weighted=True, return_indices=True,
        )

        self._model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5,
            random_state=self.random_state,
        )
        self._model.fit(X[red_idx], y[red_idx])
        return self

    def predict(self, X):
        return self._model.predict(np.asarray(X, dtype=np.float64))


# ── Synthetics ───────────────────────────────────────────────────────────────

def make_degree2(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, 3*X[:,0]*X[:,1] + 2*X[:,2]*X[:,3] + noise*rng.randn(n)

def make_degree3(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X[:,0]+X[:,1]+X[:,2]+2*X[:,0]*X[:,1]*X[:,2] + noise*rng.randn(n)

def make_degree4(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X[:,0]+X[:,1]+X[:,2]+X[:,3]+1.5*X[:,0]*X[:,1]*X[:,2]*X[:,3] + noise*rng.randn(n)

def make_mixed(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, 2*X[:,0]*X[:,1]+1.5*X[:,0]*X[:,1]*X[:,2]+X[:,0]*X[:,1]*X[:,2]*X[:,3] + noise*rng.randn(n)

def make_additive(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X.sum(axis=1) + noise*rng.randn(n)

def make_ratio(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X[:,0]*X[:,1]/(1+X[:,2]**2)+X[:,3]*X[:,4] + noise*rng.randn(n)

def make_sparse_hd(n=2000, noise=0.5, d=50, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X[:,0]+X[:,1]+X[:,2]+1.5*X[:,0]*X[:,1]*X[:,2]+0.5*X[:,3] + noise*rng.randn(n)


# ── Runner ───────────────────────────────────────────────────────────────────

def run(name, X, y, rr=0.8):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    configs = [
        ("HistGBT", lambda: HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5, random_state=42)),
        ("HVRT(T)", lambda: BlendedHVRTRegressor(lam3=0.0, reduce_ratio=rr)),
        ("T+e3", lambda: BlendedHVRTRegressor(lam3=1.0, reduce_ratio=rr)),
        ("T+e3+e4", lambda: BlendedHVRTRegressor(lam3=1.0, lam4=1.0, reduce_ratio=rr)),
        ("T+2e3", lambda: BlendedHVRTRegressor(lam3=2.0, reduce_ratio=rr)),
        ("y-signal", lambda: YSignalHVRTRegressor(reduce_ratio=rr, y_weight=0.75)),
        ("GeoXGB", lambda: GeoXGBRegressor(n_rounds=500, random_state=42)),
    ]

    results = {}
    for label, make in configs:
        t0 = time.perf_counter()
        m = make()
        m.fit(X_tr, y_tr)
        el = time.perf_counter() - t0
        r2 = r2_score(y_te, m.predict(X_te))
        results[label] = (r2, el)

    base_r2 = results["HistGBT"][0]
    print(f"\n  {name}")
    for label, (r2, el) in results.items():
        d = r2 - base_r2
        ds = f"{d:+.4f}" if label != "HistGBT" else "  ---"
        print(f"    {label:14s}  R2={r2:.4f}  {ds}  ({el:.1f}s)")

    return results


if __name__ == "__main__":
    print("=" * 78)
    print("Blended HVRT Target: T + lambda * e3")
    print("=" * 78)

    synthetic = [
        ("Additive (no interactions)", make_additive()),
        ("Degree-2 (3ab+2cd)", make_degree2()),
        ("Degree-3 (a+b+c+2abc)", make_degree3()),
        ("Degree-4 (a+b+c+d+1.5abcd)", make_degree4()),
        ("Mixed (2ab+1.5abc+abcd)", make_mixed()),
        ("Ratio (ab/(1+c²)+de)", make_ratio()),
        ("Sparse d=50 (degree-3)", make_sparse_hd()),
    ]

    _db = load_diabetes()
    _wine = load_wine()
    _cal = fetch_california_housing()
    _cal_idx = np.random.RandomState(42).choice(len(_cal.data), 5000, replace=False)

    real = [
        ("Diabetes (n=442, d=10)", _db.data, _db.target),
        ("Friedman #1 (n=2k, d=10)", *make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=42)),
        ("Friedman #2 (n=2k, d=4)", *make_friedman2(n_samples=2000, noise=50.0, random_state=42)),
        ("Friedman #3 (n=2k, d=4)", *make_friedman3(n_samples=2000, noise=0.1, random_state=42)),
        ("Wine (n=178, d=13)", _wine.data, _wine.target.astype(float)),
        ("CalHousing (n=5k, d=8)", _cal.data[_cal_idx], _cal.target[_cal_idx]),
    ]

    all_results = []

    print("\n" + "-" * 78)
    print("A. SYNTHETIC")
    print("-" * 78)
    for name, (X, y) in synthetic:
        all_results.append((name, run(name, X, y)))

    print("\n" + "-" * 78)
    print("B. REAL")
    print("-" * 78)
    for name, X, y in real:
        all_results.append((name, run(name, X, y)))

    # Summary
    print("\n\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    labels = ["HVRT(T)", "T+e3", "T+e3+e4", "T+2e3", "y-signal", "GeoXGB"]
    print(f"\n  {'Dataset':35s} {'Base':>6s}", end="")
    for l in labels:
        print(f"  {l:>10s}", end="")
    print()
    print("  " + "-" * (37 + 12 * len(labels)))

    deltas = {l: [] for l in labels}
    for name, results in all_results:
        base = results["HistGBT"][0]
        print(f"  {name:35s} {base:6.4f}", end="")
        for l in labels:
            d = results[l][0] - base
            deltas[l].append(d)
            m = "*" if d > 0.005 else ("!" if d < -0.005 else " ")
            print(f"  {d:+.4f}{m:>5s}", end="")
        print()

    print("  " + "-" * (37 + 12 * len(labels)))
    print(f"  {'Mean':35s} {'':6s}", end="")
    for l in labels:
        print(f"  {np.mean(deltas[l]):+.4f}     ", end="")
    print()

    # Head-to-head
    print(f"\n  Head-to-head vs HVRT(T) (margin=0.002):")
    for l in ["T+e3", "T+e3+e4", "T+2e3", "y-signal"]:
        wins = sum(1 for a, b in zip(deltas[l], deltas["HVRT(T)"]) if a > b + 0.002)
        losses = sum(1 for a, b in zip(deltas[l], deltas["HVRT(T)"]) if b > a + 0.002)
        ties = len(all_results) - wins - losses
        diff = np.mean(deltas[l]) - np.mean(deltas["HVRT(T)"])
        print(f"    {l:10s}  W={wins} L={losses} T={ties}  mean diff={diff:+.4f}")
