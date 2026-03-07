"""
Ablation benchmark: Does joint multi-degree reduction make e_k features work?

Isolates the reduction-alignment variable from GeoXGB's HVRT by using
sklearn HistGradientBoostingRegressor as the GBT backend.

Configurations tested (all use same HistGBT hyperparams):
1. HistGBT         — no reduction, no augmentation (baseline)
2. T-reduce+GBT   — T-only reduction, no augmentation (does reduction hurt?)
3. HistGBT+ek      — no reduction, e_k augmentation (do e_k features help at all?)
4. T-reduce+ek     — T-only reduction + e_k augmentation (HVRT-style)
5. Joint-reduce+ek — Joint e₂/e₃/e₄ reduction + e_k augmentation (our hypothesis)
6. GeoXGB Base     — full GeoXGB for reference
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
from sklearn.datasets import (
    make_friedman1, make_friedman2, make_friedman3,
    load_diabetes, fetch_california_housing, load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from geoxgb.regressor import GeoXGBRegressor
from geoxgb.experimental._ek_aligned import (
    PlainHistGBT,
    TReducedHistGBT,
    PlainEkRegressor,
    TReducedEkRegressor,
    EkAlignedRegressor,
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

def run_ablation(name, X, y, reduce_ratio=0.8):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    configs = [
        ("HistGBT",         lambda: PlainHistGBT(random_state=42)),
        ("T-red+GBT",       lambda: TReducedHistGBT(reduce_ratio=reduce_ratio, random_state=42)),
        ("HistGBT+ek",      lambda: PlainEkRegressor(max_degree=4, random_state=42)),
        ("T-red+ek",        lambda: TReducedEkRegressor(max_degree=4, reduce_ratio=reduce_ratio, random_state=42)),
        ("Joint-red+ek",    lambda: EkAlignedRegressor(max_degree=4, reduce_ratio=reduce_ratio, random_state=42)),
        ("GeoXGB",          lambda: GeoXGBRegressor(n_rounds=500, random_state=42)),
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
    print("Ablation: Joint Multi-Degree Reduction vs T-Only Reduction")
    print("=" * 78)
    print("  reduce_ratio=0.8 for all reduced configs")
    print("  HistGBT: max_iter=500, lr=0.05, max_depth=5")

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
        r = run_ablation(name, X, y)
        all_results.append((name, r))

    print("\n" + "-" * 78)
    print("B. REAL")
    print("-" * 78)

    for name, X, y in real:
        r = run_ablation(name, X, y)
        all_results.append((name, r))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 78)
    print("SUMMARY: Delta R² vs HistGBT baseline")
    print("=" * 78)

    labels = ["T-red+GBT", "HistGBT+ek", "T-red+ek", "Joint-red+ek", "GeoXGB"]
    header = f"  {'Dataset':40s}  {'Base':>6s}"
    for l in labels:
        header += f"  {l:>13s}"
    print(header)
    print("  " + "-" * (42 + 15 * len(labels)))

    deltas = {l: [] for l in labels}
    for name, results in all_results:
        base = results["HistGBT"][0]
        line = f"  {name:40s}  {base:6.4f}"
        for l in labels:
            d = results[l][0] - base
            deltas[l].append(d)
            marker = "*" if d > 0.005 else ("!" if d < -0.005 else " ")
            line += f"  {d:+.4f}{marker:>8s}"
        print(line)

    print("  " + "-" * (42 + 15 * len(labels)))
    means_line = f"  {'Mean':40s}  {'':6s}"
    for l in labels:
        means_line += f"  {np.mean(deltas[l]):+.4f}        "
    print(means_line)

    # Key comparison: Joint-reduce+ek vs T-reduce+ek
    print("\n  Key comparison (Joint vs T-aligned reduction with e_k):")
    joint_wins = sum(1 for j, t in zip(deltas["Joint-red+ek"], deltas["T-red+ek"]) if j > t + 0.002)
    t_wins = sum(1 for j, t in zip(deltas["Joint-red+ek"], deltas["T-red+ek"]) if t > j + 0.002)
    ties = len(all_results) - joint_wins - t_wins
    print(f"    Joint wins: {joint_wins}  T wins: {t_wins}  Ties: {ties}  (margin=0.002)")

    j_mean = np.mean(deltas["Joint-red+ek"])
    t_mean = np.mean(deltas["T-red+ek"])
    print(f"    Joint mean delta: {j_mean:+.4f}  T mean delta: {t_mean:+.4f}  diff: {j_mean - t_mean:+.4f}")


# ── Dataset helpers ──────────────────────────────────────────────────────────

def _load_diabetes():
    from sklearn.datasets import load_diabetes
    d = load_diabetes()
    return d.data, d.target

def _load_wine():
    from sklearn.datasets import load_wine
    d = load_wine()
    return d.data, d.target.astype(float)

def _load_calhousing(n):
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    idx = np.random.RandomState(42).choice(len(d.data), n, replace=False)
    return d.data[idx], d.target[idx]
