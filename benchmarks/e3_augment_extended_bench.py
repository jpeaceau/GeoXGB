"""
Extended benchmark: AdaptiveEkGeoXGBRegressor on diverse datasets.

Synthetic datasets test specific interaction structures.
Real datasets test practical value on tabular regression tasks.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
from sklearn.datasets import (
    make_friedman1, make_friedman2, make_friedman3,
    load_diabetes, fetch_california_housing,
    load_wine,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

from geoxgb.regressor import GeoXGBRegressor
from geoxgb.experimental._e3_augment import AdaptiveEkGeoXGBRegressor


# ── Synthetic dataset generators ─────────────────────────────────────────────

def make_degree2_pure(n=2000, noise=0.3, d=10, rs=42):
    """Pure pairwise: y = 3*x0*x1 + 2*x2*x3 + noise."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = 3*X[:,0]*X[:,1] + 2*X[:,2]*X[:,3] + noise*rng.randn(n)
    return X, y

def make_degree3_abc(n=2000, noise=0.3, d=10, rs=42):
    """Degree-3: y = a + b + c + 2*a*b*c + noise."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0] + X[:,1] + X[:,2] + 2*X[:,0]*X[:,1]*X[:,2] + noise*rng.randn(n)
    return X, y

def make_degree4_abcd(n=2000, noise=0.3, d=10, rs=42):
    """Degree-4: y = a+b+c+d + 1.5*a*b*c*d + noise."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = (X[:,0] + X[:,1] + X[:,2] + X[:,3]
         + 1.5*X[:,0]*X[:,1]*X[:,2]*X[:,3] + noise*rng.randn(n))
    return X, y

def make_mixed_degree(n=2000, noise=0.3, d=10, rs=42):
    """Mixed degrees: y = 2*ab + 1.5*abc + 1.0*abcd + noise."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    a, b, c, dd = X[:,0], X[:,1], X[:,2], X[:,3]
    y = 2*a*b + 1.5*a*b*c + 1.0*a*b*c*dd + noise*rng.randn(n)
    return X, y

def make_signflip(n=2000, noise=0.5, d=10, rs=42):
    """Sign-flip: y = a + 2*a*b*sign(c*d) + noise."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0] + 2*X[:,0]*X[:,1]*np.sign(X[:,2]*X[:,3]) + noise*rng.randn(n)
    return X, y

def make_sparse_high_d(n=2000, noise=0.5, d=50, rs=42):
    """High-d sparse: only 4 of 50 features matter, degree-3 interaction."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0] + X[:,1] + X[:,2] + 1.5*X[:,0]*X[:,1]*X[:,2] + 0.5*X[:,3] + noise*rng.randn(n)
    return X, y

def make_additive_only(n=2000, noise=0.3, d=10, rs=42):
    """Pure additive (no interactions): y = sum(x_i) + noise. Should NOT benefit."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X.sum(axis=1) + noise*rng.randn(n)
    return X, y

def make_nonlinear_additive(n=2000, noise=0.3, d=10, rs=42):
    """Nonlinear additive: y = sin(x0) + x1^2 + exp(-x2^2) + noise. No cross-terms."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = np.sin(X[:,0]) + X[:,1]**2 + np.exp(-X[:,2]**2) + noise*rng.randn(n)
    return X, y

def make_xor_like(n=2000, noise=0.3, d=10, rs=42):
    """XOR-like: y = sign(x0)*sign(x1) + sign(x2)*sign(x3) + noise."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = np.sign(X[:,0])*np.sign(X[:,1]) + np.sign(X[:,2])*np.sign(X[:,3]) + noise*rng.randn(n)
    return X, y

def make_ratio_interaction(n=2000, noise=0.3, d=10, rs=42):
    """Ratio interaction: y = x0*x1/(1+x2^2) + x3*x4 + noise."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0]*X[:,1]/(1+X[:,2]**2) + X[:,3]*X[:,4] + noise*rng.randn(n)
    return X, y

def make_degree5(n=3000, noise=0.3, d=10, rs=42):
    """Degree-5: y = a*b*c*d*e + noise (beyond e4 capacity)."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0]*X[:,1]*X[:,2]*X[:,3]*X[:,4] + noise*rng.randn(n)
    return X, y

def make_heteroscedastic(n=2000, noise=1.0, d=10, rs=42):
    """Heteroscedastic: y = x0*x1*x2 + |x0|*noise. Noise scales with signal."""
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0]*X[:,1]*X[:,2] + np.abs(X[:,0])*noise*rng.randn(n)
    return X, y


# ── Real dataset loaders ─────────────────────────────────────────────────────

def load_real_datasets():
    """Return list of (name, X, y) for real regression datasets."""
    datasets = []

    # 1. Diabetes (d=10, n=442)
    data = load_diabetes()
    datasets.append(("Diabetes (n=442, d=10)", data.data, data.target))

    # 2. California Housing (subsample, d=8)
    data = fetch_california_housing()
    idx = np.random.RandomState(42).choice(len(data.data), 5000, replace=False)
    datasets.append(("CalHousing (n=5k, d=8)", data.data[idx], data.target[idx]))

    # 3. Friedman #1 (d=10, known pairwise)
    X, y = make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=42)
    datasets.append(("Friedman #1 (n=2k, d=10)", X, y))

    # 4. Friedman #2 (d=4, nonlinear)
    X, y = make_friedman2(n_samples=2000, noise=50.0, random_state=42)
    datasets.append(("Friedman #2 (n=2k, d=4)", X, y))

    # 5. Friedman #3 (d=4, atan interaction)
    X, y = make_friedman3(n_samples=2000, noise=0.1, random_state=42)
    datasets.append(("Friedman #3 (n=2k, d=4)", X, y))

    # 6. Wine quality as regression (d=13)
    data = load_wine()
    datasets.append(("Wine (n=178, d=13)", data.data, data.target.astype(float)))

    # 7. California Housing full (larger n)
    data = fetch_california_housing()
    idx = np.random.RandomState(99).choice(len(data.data), 10000, replace=False)
    datasets.append(("CalHousing (n=10k, d=8)", data.data[idx], data.target[idx]))

    return datasets


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_single(name, X, y, params, noise_floor=0.3):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    configs = [
        ("Base", lambda **kw: GeoXGBRegressor(**kw)),
        ("Ek d=3", lambda **kw: AdaptiveEkGeoXGBRegressor(
            max_degree=3, top_k_max=10, noise_alpha=1.5,
            noise_floor=noise_floor, **kw)),
        ("Ek d=4", lambda **kw: AdaptiveEkGeoXGBRegressor(
            max_degree=4, top_k_max=10, noise_alpha=1.5,
            noise_floor=noise_floor, **kw)),
        ("Ek d=4+p", lambda **kw: AdaptiveEkGeoXGBRegressor(
            max_degree=4, top_k_max=10, noise_alpha=1.5,
            noise_floor=noise_floor, include_partials=True, **kw)),
    ]

    results = {}
    for label, make in configs:
        t0 = time.perf_counter()
        m = make(**params)
        m.fit(X_tr, y_tr)
        el = time.perf_counter() - t0
        r2 = r2_score(y_te, m.predict(X_te))
        results[label] = (r2, el, m)

    return results


def print_results(name, results, noise_floor):
    base_r2 = results["Base"][0]
    print(f"\n  {name} (nf={noise_floor})")
    for label, (r2, el, m) in results.items():
        d = r2 - base_r2
        ds = f"{d:+.4f}" if label != "Base" else "  ---"
        extra = ""
        if hasattr(m, 'ek_summary'):
            s = m.ek_summary()
            ne = s.get('noise_estimate', 0) or 0
            bud = " ".join(f"k{k}={s['degrees'].get(k,{}).get('effective_k',0)}"
                           for k in range(2, s.get('max_degree',3)+1))
            extra = f"  [ne={ne:.2f} {bud}]"
        print(f"    {label:12s}  R2={r2:.4f}  {ds}  ({el:.1f}s){extra}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 78)
    print("Extended Adaptive E_k Hierarchy Benchmark")
    print("=" * 78)

    p = dict(n_rounds=500, random_state=42)
    p_hpo = dict(n_rounds=1000, learning_rate=0.02, max_depth=3,
                 refit_interval=50, random_state=42)

    # ── A. Synthetic datasets ────────────────────────────────────────────────
    synthetic = [
        ("Additive only (no interactions)", make_additive_only()),
        ("Nonlinear additive (no cross)", make_nonlinear_additive()),
        ("XOR-like (sign interactions)", make_xor_like()),
        ("Pure degree-2 (3ab+2cd)", make_degree2_pure()),
        ("Degree-3 (a+b+c+2abc)", make_degree3_abc()),
        ("Degree-4 (a+b+c+d+1.5abcd)", make_degree4_abcd()),
        ("Mixed (2ab+1.5abc+abcd)", make_mixed_degree()),
        ("SignFlip-4 (a+2ab*sign(cd))", make_signflip()),
        ("Ratio (ab/(1+c^2)+de)", make_ratio_interaction()),
        ("Degree-5 (abcde)", make_degree5()),
        ("Sparse d=50 (degree-3)", make_sparse_high_d()),
        ("Heteroscedastic (abc+|a|*noise)", make_heteroscedastic()),
    ]

    print("\n" + "-" * 78)
    print("A. SYNTHETIC DATASETS (noise_floor=0.3)")
    print("-" * 78)

    syn_summary = []
    for name, (X, y) in synthetic:
        results = run_single(name, X, y, p, noise_floor=0.3)
        print_results(name, results, 0.3)
        syn_summary.append((name, results))

    # ── B. Real datasets ─────────────────────────────────────────────────────
    real = load_real_datasets()

    print("\n" + "-" * 78)
    print("B. REAL DATASETS (noise_floor=0.3)")
    print("-" * 78)

    real_summary = []
    for name, X, y in real:
        # Use HPO-like params for real datasets
        results = run_single(name, X, y, p_hpo, noise_floor=0.3)
        print_results(name, results, 0.3)
        real_summary.append((name, results))

    # ── C. Summary table ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 78)
    print("SUMMARY: Delta R2 vs Base")
    print("=" * 78)
    print(f"\n  {'Dataset':45s}  {'Base':>7s}  {'Ek3':>7s}  {'Ek4':>7s}  {'Ek4+p':>7s}")
    print("  " + "-" * 78)

    all_results = syn_summary + real_summary
    deltas_ek3 = []
    deltas_ek4 = []
    deltas_ek4p = []

    for name, results in all_results:
        base = results["Base"][0]
        d3 = results["Ek d=3"][0] - base
        d4 = results["Ek d=4"][0] - base
        d4p = results["Ek d=4+p"][0] - base
        deltas_ek3.append(d3)
        deltas_ek4.append(d4)
        deltas_ek4p.append(d4p)
        # Color code: positive = good
        def fmt(v):
            if v > 0.005:
                return f"{v:+.4f}*"
            elif v < -0.005:
                return f"{v:+.4f}!"
            else:
                return f"{v:+.4f} "
        print(f"  {name:45s}  {base:7.4f}  {fmt(d3)}  {fmt(d4)}  {fmt(d4p)}")

    print("  " + "-" * 78)
    print(f"  {'Mean delta':45s}  {'':7s}  {np.mean(deltas_ek3):+.4f}  "
          f"{np.mean(deltas_ek4):+.4f}  {np.mean(deltas_ek4p):+.4f}")
    print(f"  {'Median delta':45s}  {'':7s}  {np.median(deltas_ek3):+.4f}  "
          f"{np.median(deltas_ek4):+.4f}  {np.median(deltas_ek4p):+.4f}")

    n_better = lambda ds: sum(1 for d in ds if d > 0.005)
    n_worse = lambda ds: sum(1 for d in ds if d < -0.005)
    n_neutral = lambda ds: sum(1 for d in ds if -0.005 <= d <= 0.005)
    n = len(all_results)

    print(f"\n  Win/Neutral/Loss (threshold=0.005):")
    print(f"    Ek d=3 :  {n_better(deltas_ek3)}/{n_neutral(deltas_ek3)}/{n_worse(deltas_ek3)}  of {n}")
    print(f"    Ek d=4 :  {n_better(deltas_ek4)}/{n_neutral(deltas_ek4)}/{n_worse(deltas_ek4)}  of {n}")
    print(f"    Ek d=4+p: {n_better(deltas_ek4p)}/{n_neutral(deltas_ek4p)}/{n_worse(deltas_ek4p)}  of {n}")

    # Real-only summary
    print(f"\n  Real datasets only:")
    rd3 = [deltas_ek3[i] for i in range(len(syn_summary), len(all_results))]
    rd4 = [deltas_ek4[i] for i in range(len(syn_summary), len(all_results))]
    rd4p = [deltas_ek4p[i] for i in range(len(syn_summary), len(all_results))]
    print(f"    Ek d=3  mean={np.mean(rd3):+.4f}  W/N/L={n_better(rd3)}/{n_neutral(rd3)}/{n_worse(rd3)}")
    print(f"    Ek d=4  mean={np.mean(rd4):+.4f}  W/N/L={n_better(rd4)}/{n_neutral(rd4)}/{n_worse(rd4)}")
    print(f"    Ek d=4+p mean={np.mean(rd4p):+.4f}  W/N/L={n_better(rd4p)}/{n_neutral(rd4p)}/{n_worse(rd4p)}")
