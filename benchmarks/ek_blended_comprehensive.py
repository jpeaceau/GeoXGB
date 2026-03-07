"""
Comprehensive benchmark: T+λ·e₃ blended HVRT target with HPO.

Tests across a wide range of synthetic and real datasets.
For each dataset, runs a small HPO sweep over λ₃ and reduce_ratio
to determine optimal blending, then compares against T-only and no-reduction.

Datasets:
  Synthetic (12): additive, degree-2, degree-3, degree-4, mixed, xor, ratio,
                  sparse-d50, signflip, heteroscedastic, nonlinear-additive, degree-5
  Real (8): diabetes, friedman1/2/3, wine, calhousing-5k/10k, boston-like
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import warnings
import numpy as np
from sklearn.datasets import (
    make_friedman1, make_friedman2, make_friedman3,
    load_diabetes, fetch_california_housing, load_wine,
    make_regression,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

from hvrt import HVRT

from geoxgb.regressor import GeoXGBRegressor
from geoxgb.experimental._e3_augment import (
    _compute_e2_scalar,
    _compute_e3_scalar,
    _compute_e4_scalar,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _robust_whiten(X):
    center = np.median(X, axis=0)
    mad = np.median(np.abs(X - center), axis=0)
    mad[mad < 1e-12] = 1.0
    scale = mad * 1.4826
    return (X - center) / scale, center, scale


def _make_blended_target(Z, lam3=1.0, lam4=0.0):
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


class BlendedHVRTRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lam3=0.0, lam4=0.0, reduce_ratio=0.8,
                 y_weight=0.25, max_iter=500, learning_rate=0.05,
                 max_depth=5, random_state=42):
        self.lam3 = lam3
        self.lam4 = lam4
        self.reduce_ratio = reduce_ratio
        self.y_weight = y_weight
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
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
            max_iter=self.max_iter, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state,
        )
        self._model.fit(X[red_idx], y[red_idx])
        return self

    def predict(self, X):
        return self._model.predict(np.asarray(X, dtype=np.float64))



# ── Synthetic generators ────────────────────────────────────────────────────

def make_additive(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X.sum(axis=1) + noise*rng.randn(n)

def make_nonlinear_additive(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, np.sin(X[:,0]) + X[:,1]**2 + np.exp(-X[:,2]**2) + noise*rng.randn(n)

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
    return X, 2*X[:,0]*X[:,1]+1.5*X[:,0]*X[:,1]*X[:,2]+X[:,0]*X[:,1]*X[:,2]*X[:,3]+noise*rng.randn(n)

def make_xor(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, np.sign(X[:,0])*np.sign(X[:,1])+np.sign(X[:,2])*np.sign(X[:,3])+noise*rng.randn(n)

def make_ratio(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X[:,0]*X[:,1]/(1+X[:,2]**2)+X[:,3]*X[:,4]+noise*rng.randn(n)

def make_signflip(n=2000, noise=0.5, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X[:,0]+2*X[:,0]*X[:,1]*np.sign(X[:,2]*X[:,3])+noise*rng.randn(n)

def make_sparse_hd(n=2000, noise=0.5, d=50, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X[:,0]+X[:,1]+X[:,2]+1.5*X[:,0]*X[:,1]*X[:,2]+0.5*X[:,3]+noise*rng.randn(n)

def make_degree5(n=3000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X[:,0]*X[:,1]*X[:,2]*X[:,3]*X[:,4]+noise*rng.randn(n)

def make_heteroscedastic(n=2000, noise=1.0, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    return X, X[:,0]*X[:,1]*X[:,2]+np.abs(X[:,0])*noise*rng.randn(n)


# ── HPO for blended target ──────────────────────────────────────────────────

def hpo_blended(X_train, y_train, cv=3, random_state=42):
    """
    Compact grid search over lambda3, reduce_ratio, y_weight.
    18 trials (6 × 3 configs) to keep runtime reasonable.
    """
    configs = [
        # (lam3, rr, yw) — targeted grid
        (0.0, 0.8, 0.25),   # T-only baseline
        (0.0, 1.0, 0.25),   # no reduction baseline
        (1.0, 0.8, 0.25),   # T+e3 default
        (2.0, 0.8, 0.25),   # T+2e3 default
        (3.0, 0.8, 0.25),   # T+3e3
        (2.0, 0.6, 0.25),   # T+2e3 aggressive reduce
        (2.0, 1.0, 0.25),   # T+2e3 no reduce
        (1.0, 0.8, 0.15),   # lower y_weight
        (2.0, 0.8, 0.15),   # T+2e3 lower yw
        (1.0, 0.8, 0.40),   # higher y_weight
        (2.0, 0.8, 0.40),   # T+2e3 higher yw
        (0.5, 0.8, 0.25),   # mild blend
        (1.5, 0.8, 0.25),   # moderate blend
        (2.0, 0.9, 0.25),   # light reduce
        (1.0, 1.0, 0.25),   # T+e3 no reduce
        (3.0, 0.8, 0.15),   # strong blend low yw
    ]

    best_score = -np.inf
    best_params = {}

    for lam3, rr, yw in configs:
        model = BlendedHVRTRegressor(
            lam3=lam3, reduce_ratio=rr, y_weight=yw,
            random_state=random_state,
        )
        try:
            scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring='r2',
            )
            mean_cv = scores.mean()
        except Exception:
            mean_cv = -999

        if mean_cv > best_score:
            best_score = mean_cv
            best_params = {'lam3': lam3, 'rr': rr, 'yw': yw, 'cv_r2': mean_cv}

    return best_params, []


# ── Main runner ──────────────────────────────────────────────────────────────

def run_dataset(name, X, y, do_hpo=True):
    """Run full comparison for one dataset."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # 1. HistGBT baseline (no reduction)
    t0 = time.perf_counter()
    m = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=5, random_state=42)
    m.fit(X_tr, y_tr)
    results["HistGBT"] = (r2_score(y_te, m.predict(X_te)), time.perf_counter()-t0)

    # 2. HVRT(T) — T-only reduction, rr=0.8
    t0 = time.perf_counter()
    m = BlendedHVRTRegressor(lam3=0.0, reduce_ratio=0.8)
    m.fit(X_tr, y_tr)
    results["HVRT(T) rr=.8"] = (r2_score(y_te, m.predict(X_te)), time.perf_counter()-t0)

    # 3. T+2e3 fixed (best from prior bench)
    t0 = time.perf_counter()
    m = BlendedHVRTRegressor(lam3=2.0, reduce_ratio=0.8)
    m.fit(X_tr, y_tr)
    results["T+2e3 rr=.8"] = (r2_score(y_te, m.predict(X_te)), time.perf_counter()-t0)

    # 4. T+2e3 no reduction
    t0 = time.perf_counter()
    m = BlendedHVRTRegressor(lam3=2.0, reduce_ratio=1.0)
    m.fit(X_tr, y_tr)
    results["T+2e3 rr=1"] = (r2_score(y_te, m.predict(X_te)), time.perf_counter()-t0)

    # 5. HPO blended
    hpo_info = ""
    if do_hpo:
        t0 = time.perf_counter()
        best, _ = hpo_blended(X_tr, y_tr, cv=3)
        hpo_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        m = BlendedHVRTRegressor(
            lam3=best['lam3'], reduce_ratio=best['rr'], y_weight=best['yw'])
        m.fit(X_tr, y_tr)
        r2 = r2_score(y_te, m.predict(X_te))
        results["HPO Blend"] = (r2, time.perf_counter()-t0)
        hpo_info = f"  [l3={best['lam3']:.1f} rr={best['rr']:.1f} yw={best['yw']:.2f} cv={best['cv_r2']:.4f} hpo={hpo_time:.0f}s]"

    # 6. GeoXGB reference
    t0 = time.perf_counter()
    m = GeoXGBRegressor(n_rounds=500, random_state=42)
    m.fit(X_tr, y_tr)
    results["GeoXGB"] = (r2_score(y_te, m.predict(X_te)), time.perf_counter()-t0)

    # Print
    base_r2 = results["HistGBT"][0]
    print(f"\n  {name}")
    for label, (r2, el) in results.items():
        d = r2 - base_r2
        ds = f"{d:+.4f}" if label != "HistGBT" else "  ---"
        extra = hpo_info if label == "HPO Blend" else ""
        print(f"    {label:16s}  R2={r2:.4f}  {ds}  ({el:.1f}s){extra}")
    sys.stdout.flush()

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("Comprehensive T+λ·e₃ Blended HVRT Target with HPO")
    print("=" * 80)
    print("  HPO: 16 targeted configs, 3-fold CV")

    # ── Synthetic ────────────────────────────────────────────────────────────
    synthetic = [
        ("Additive (no interactions)", make_additive()),
        ("Nonlinear additive", make_nonlinear_additive()),
        ("Degree-2 (3ab+2cd)", make_degree2()),
        ("Degree-3 (a+b+c+2abc)", make_degree3()),
        ("Degree-4 (abcd)", make_degree4()),
        ("Mixed (ab+abc+abcd)", make_mixed()),
        ("XOR-like", make_xor()),
        ("Ratio (ab/(1+c²)+de)", make_ratio()),
        ("SignFlip (a+2ab·sgn(cd))", make_signflip()),
        ("Sparse d=50 (degree-3)", make_sparse_hd()),
        ("Degree-5 (abcde)", make_degree5()),
        ("Heteroscedastic", make_heteroscedastic()),
    ]

    # ── Real ─────────────────────────────────────────────────────────────────
    _db = load_diabetes()
    _wine = load_wine()
    _cal = fetch_california_housing()
    _rng = np.random.RandomState(42)
    _cal5k = _rng.choice(len(_cal.data), 5000, replace=False)
    _cal10k = _rng.choice(len(_cal.data), 10000, replace=False)

    # sklearn make_regression as a "Boston-like" stand-in
    _Xmr, _ymr = make_regression(
        n_samples=1000, n_features=13, n_informative=8,
        noise=20.0, random_state=42)

    real = [
        ("Diabetes (n=442, d=10)", _db.data, _db.target),
        ("Friedman #1 (n=2k, d=10)", *make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=42)),
        ("Friedman #2 (n=2k, d=4)", *make_friedman2(n_samples=2000, noise=50.0, random_state=42)),
        ("Friedman #3 (n=2k, d=4)", *make_friedman3(n_samples=2000, noise=0.1, random_state=42)),
        ("Wine (n=178, d=13)", _wine.data, _wine.target.astype(float)),
        ("CalHousing (n=5k, d=8)", _cal.data[_cal5k], _cal.target[_cal5k]),
        ("CalHousing (n=10k, d=8)", _cal.data[_cal10k], _cal.target[_cal10k]),
        ("Regression (n=1k, d=13)", _Xmr, _ymr),
    ]

    print("\n" + "-" * 80)
    print("A. SYNTHETIC")
    print("-" * 80)

    all_results = []
    for name, (X, y) in synthetic:
        r = run_dataset(name, X, y)
        all_results.append((name, r))

    print("\n" + "-" * 80)
    print("B. REAL")
    print("-" * 80)

    for name, X, y in real:
        r = run_dataset(name, X, y)
        all_results.append((name, r))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SUMMARY: Delta R² vs HistGBT")
    print("=" * 80)

    labels = ["HVRT(T) rr=.8", "T+2e3 rr=.8", "T+2e3 rr=1", "HPO Blend", "GeoXGB"]
    print(f"\n  {'Dataset':32s} {'Base':>6s}", end="")
    for l in labels:
        print(f"  {l:>14s}", end="")
    print()
    print("  " + "-" * (34 + 16 * len(labels)))

    deltas = {l: [] for l in labels}
    for name, results in all_results:
        base = results["HistGBT"][0]
        print(f"  {name:32s} {base:6.4f}", end="")
        for l in labels:
            if l in results:
                d = results[l][0] - base
                deltas[l].append(d)
                m = "*" if d > 0.005 else ("!" if d < -0.005 else " ")
                print(f"  {d:+.4f}{m:>9s}", end="")
            else:
                print(f"  {'N/A':>14s}", end="")
        print()

    print("  " + "-" * (34 + 16 * len(labels)))
    print(f"  {'Mean':32s} {'':6s}", end="")
    for l in labels:
        if deltas[l]:
            print(f"  {np.mean(deltas[l]):+.4f}         ", end="")
        else:
            print(f"  {'N/A':>14s}", end="")
    print()
    print(f"  {'Median':32s} {'':6s}", end="")
    for l in labels:
        if deltas[l]:
            print(f"  {np.median(deltas[l]):+.4f}         ", end="")
        else:
            print(f"  {'N/A':>14s}", end="")
    print()

    # Win/Loss counts
    print(f"\n  Win/Neutral/Loss vs HistGBT (threshold ±0.005):")
    for l in labels:
        if not deltas[l]:
            continue
        w = sum(1 for d in deltas[l] if d > 0.005)
        n = sum(1 for d in deltas[l] if -0.005 <= d <= 0.005)
        lo = sum(1 for d in deltas[l] if d < -0.005)
        print(f"    {l:16s}  W={w:2d} N={n:2d} L={lo:2d}")

    # Head-to-head: HPO Blend vs HVRT(T)
    print(f"\n  Head-to-head HPO Blend vs HVRT(T) rr=.8 (margin=0.002):")
    hpo_d = deltas.get("HPO Blend", [])
    t_d = deltas.get("HVRT(T) rr=.8", [])
    if hpo_d and t_d:
        wins = sum(1 for a, b in zip(hpo_d, t_d) if a > b + 0.002)
        losses = sum(1 for a, b in zip(hpo_d, t_d) if b > a + 0.002)
        ties = len(hpo_d) - wins - losses
        print(f"    HPO wins: {wins}  T wins: {losses}  Ties: {ties}")
        print(f"    HPO mean: {np.mean(hpo_d):+.4f}  T mean: {np.mean(t_d):+.4f}  diff: {np.mean(hpo_d)-np.mean(t_d):+.4f}")

    # What lambda did HPO pick?
    print(f"\n  HPO selected λ₃ distribution:")
    # (can't easily extract from results dict, but the per-dataset printout shows it)
