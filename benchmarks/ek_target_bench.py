"""
Benchmark: Multiplicative composite HVRT target statistics.

Tests whether changing what HVRT partitions/reduces BY (not what features
the GBT sees) improves performance on datasets with higher-order interactions.

Configs:
1. HistGBT          — no reduction (baseline)
2. HVRT(T) + GBT    — standard T-target HVRT reduction
3. HVRT(T·e₃) + GBT — multiplicative degree-2×3 target
4. HVRT(rank prod)   — rank-normalized product (scale-invariant)
5. HVRT(T·e₃) + ek  — composite target + e_k augmentation (both)
6. HVRT(T) + ek      — T-only target + e_k augmentation (control)
7. GeoXGB            — full GeoXGB reference
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

from geoxgb.regressor import GeoXGBRegressor
from geoxgb.experimental._ek_target import (
    CompositeHVRTRegressor,
    composite_target_t,
    composite_target_t_e3,
    composite_target_product_normalized,
)


# ── Synthetic datasets ───────────────────────────────────────────────────────

def make_degree2(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = 3*X[:,0]*X[:,1] + 2*X[:,2]*X[:,3] + noise*rng.randn(n)
    return X, y

def make_degree3(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0] + X[:,1] + X[:,2] + 2*X[:,0]*X[:,1]*X[:,2] + noise*rng.randn(n)
    return X, y

def make_degree4(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = (X[:,0] + X[:,1] + X[:,2] + X[:,3]
         + 1.5*X[:,0]*X[:,1]*X[:,2]*X[:,3] + noise*rng.randn(n))
    return X, y

def make_mixed(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = 2*X[:,0]*X[:,1] + 1.5*X[:,0]*X[:,1]*X[:,2] + X[:,0]*X[:,1]*X[:,2]*X[:,3] + noise*rng.randn(n)
    return X, y

def make_additive(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X.sum(axis=1) + noise*rng.randn(n)
    return X, y

def make_xor(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = np.sign(X[:,0])*np.sign(X[:,1]) + np.sign(X[:,2])*np.sign(X[:,3]) + noise*rng.randn(n)
    return X, y

def make_ratio(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0]*X[:,1]/(1+X[:,2]**2) + X[:,3]*X[:,4] + noise*rng.randn(n)
    return X, y

def make_sparse_hd(n=2000, noise=0.5, d=50, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0] + X[:,1] + X[:,2] + 1.5*X[:,0]*X[:,1]*X[:,2] + 0.5*X[:,3] + noise*rng.randn(n)
    return X, y


# ── Runner ───────────────────────────────────────────────────────────────────

def run(name, X, y, reduce_ratio=0.8):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    configs = [
        ("HistGBT", lambda: HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5, random_state=42)),
        ("HVRT(T)", lambda: CompositeHVRTRegressor(
            target_fn=composite_target_t, reduce_ratio=reduce_ratio)),
        ("HVRT(T*e3)", lambda: CompositeHVRTRegressor(
            target_fn=composite_target_t_e3, reduce_ratio=reduce_ratio)),
        ("HVRT(rank)", lambda: CompositeHVRTRegressor(
            target_fn=composite_target_product_normalized, reduce_ratio=reduce_ratio)),
        ("HVRT(T*e3)+ek", lambda: CompositeHVRTRegressor(
            target_fn=composite_target_t_e3, reduce_ratio=reduce_ratio,
            augment_ek=True, max_degree=4)),
        ("HVRT(T)+ek", lambda: CompositeHVRTRegressor(
            target_fn=composite_target_t, reduce_ratio=reduce_ratio,
            augment_ek=True, max_degree=4)),
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
        print(f"    {label:16s}  R2={r2:.4f}  {ds}  ({el:.1f}s)")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 78)
    print("Composite HVRT Target Benchmark")
    print("=" * 78)
    print("  HVRT partitions/reduces by composite target, then HistGBT trains")

    synthetic = [
        ("Additive (no interactions)", make_additive()),
        ("Degree-2 (3ab+2cd)", make_degree2()),
        ("Degree-3 (a+b+c+2abc)", make_degree3()),
        ("Degree-4 (a+b+c+d+1.5abcd)", make_degree4()),
        ("Mixed (2ab+1.5abc+abcd)", make_mixed()),
        ("XOR-like (sign interactions)", make_xor()),
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

    print("\n" + "-" * 78)
    print("A. SYNTHETIC")
    print("-" * 78)

    all_results = []
    for name, (X, y) in synthetic:
        r = run(name, X, y)
        all_results.append((name, r))

    print("\n" + "-" * 78)
    print("B. REAL")
    print("-" * 78)

    for name, X, y in real:
        r = run(name, X, y)
        all_results.append((name, r))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 78)
    print("SUMMARY: Delta R² vs HistGBT")
    print("=" * 78)

    labels = ["HVRT(T)", "HVRT(T*e3)", "HVRT(rank)", "HVRT(T*e3)+ek", "HVRT(T)+ek", "GeoXGB"]
    print(f"\n  {'Dataset':35s} {'Base':>6s}", end="")
    for l in labels:
        print(f" {l:>14s}", end="")
    print()
    print("  " + "-" * (37 + 15 * len(labels)))

    deltas = {l: [] for l in labels}
    for name, results in all_results:
        base = results["HistGBT"][0]
        print(f"  {name:35s} {base:6.4f}", end="")
        for l in labels:
            d = results[l][0] - base
            deltas[l].append(d)
            m = "*" if d > 0.005 else ("!" if d < -0.005 else " ")
            print(f" {d:+.4f}{m:>9s}", end="")
        print()

    print("  " + "-" * (37 + 15 * len(labels)))
    print(f"  {'Mean':35s} {'':6s}", end="")
    for l in labels:
        print(f" {np.mean(deltas[l]):+.4f}         ", end="")
    print()

    # Key: HVRT(T*e3) vs HVRT(T) — does composite target help?
    print("\n  Key: Does composite target improve over T-only?")
    t_d = deltas["HVRT(T)"]
    te3_d = deltas["HVRT(T*e3)"]
    rk_d = deltas["HVRT(rank)"]
    wins_te3 = sum(1 for a, b in zip(te3_d, t_d) if a > b + 0.002)
    wins_t = sum(1 for a, b in zip(te3_d, t_d) if b > a + 0.002)
    print(f"    T*e₃ vs T:   T*e₃ wins {wins_te3}, T wins {wins_t}, ties {len(t_d)-wins_te3-wins_t}")
    print(f"    T*e₃ mean: {np.mean(te3_d):+.4f}  vs  T mean: {np.mean(t_d):+.4f}  diff: {np.mean(te3_d)-np.mean(t_d):+.4f}")

    wins_rk = sum(1 for a, b in zip(rk_d, t_d) if a > b + 0.002)
    wins_t2 = sum(1 for a, b in zip(rk_d, t_d) if b > a + 0.002)
    print(f"    Rank vs T:   Rank wins {wins_rk}, T wins {wins_t2}, ties {len(t_d)-wins_rk-wins_t2}")
    print(f"    Rank mean: {np.mean(rk_d):+.4f}  vs  T mean: {np.mean(t_d):+.4f}  diff: {np.mean(rk_d)-np.mean(t_d):+.4f}")
