"""
Benchmark: Adaptive e_k hierarchy (degree 2-4).

Two comparisons:
A. Ungated (noise_floor=0): shows raw signal capture potential
B. Gated (noise_floor=0.3): shows production-safe noise robustness

Synthetic datasets have explicit degree-2, 3, 4 interactions.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
from sklearn.datasets import make_friedman1, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from geoxgb.regressor import GeoXGBRegressor
from geoxgb.experimental._e3_augment import AdaptiveEkGeoXGBRegressor


def make_degree3_dataset(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0] + X[:,1] + X[:,2] + 2*X[:,0]*X[:,1]*X[:,2] + noise*rng.randn(n)
    return X, y

def make_degree4_dataset(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = (X[:,0] + X[:,1] + X[:,2] + X[:,3]
         + 1.5*X[:,0]*X[:,1]*X[:,2]*X[:,3] + noise*rng.randn(n))
    return X, y

def make_mixed_dataset(n=2000, noise=0.3, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    a, b, c, d_ = X[:,0], X[:,1], X[:,2], X[:,3]
    y = 2*a*b + 1.5*a*b*c + 1.0*a*b*c*d_ + noise*rng.randn(n)
    return X, y

def make_signflip4_dataset(n=2000, noise=0.5, d=10, rs=42):
    rng = np.random.RandomState(rs)
    X = rng.randn(n, d)
    y = X[:,0] + 2*X[:,0]*X[:,1]*np.sign(X[:,2]*X[:,3]) + noise*rng.randn(n)
    return X, y


def run(name, X, y, params, noise_floor):
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

    base_r2 = results["Base"][0]
    print(f"\n  {name} (noise_floor={noise_floor})")
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

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Adaptive E_k Hierarchy Benchmark (degree 2-4)")
    print("=" * 70)

    p = dict(n_rounds=500, random_state=42)
    p_hpo = dict(n_rounds=1000, learning_rate=0.02, max_depth=3,
                 refit_interval=50, random_state=42)

    datasets = [
        ("Degree-3 (y=a+b+c+2abc)", *make_degree3_dataset()),
        ("Degree-4 (y=a+b+c+d+1.5abcd)", *make_degree4_dataset()),
        ("Mixed (2ab+1.5abc+1.0abcd)", *make_mixed_dataset()),
        ("SignFlip-4 (a+2ab*sign(cd))", *make_signflip4_dataset()),
    ]

    # A. Ungated: noise_floor=0 (raw signal access)
    print("\n--- A. UNGATED (noise_floor=0) ---")
    for name, X, y in datasets:
        run(name, X, y, p, noise_floor=0.0)

    # B. Gated: noise_floor=0.3 (production-safe)
    print("\n--- B. GATED (noise_floor=0.3) ---")
    for name, X, y in datasets:
        run(name, X, y, p, noise_floor=0.3)

    # C. Real datasets (gated)
    print("\n--- C. REAL DATASETS (noise_floor=0.3) ---")
    data = load_diabetes()
    run("Diabetes", data.data, data.target, p_hpo, noise_floor=0.3)

    X, y = make_friedman1(n_samples=2000, n_features=10, noise=1.0, random_state=42)
    run("Friedman #1", X, y, p, noise_floor=0.3)

    data = fetch_california_housing()
    idx = np.random.RandomState(42).choice(len(data.data), 5000, replace=False)
    run("California Housing (5k)", data.data[idx], data.target[idx], p_hpo, noise_floor=0.3)

    # D. Noise robustness sweep (mixed-degree, ungated to show the difference)
    print("\n\n" + "=" * 70)
    print("NOISE ROBUSTNESS: Mixed-degree, training noise sweep")
    print("=" * 70)

    _, X_c, y = datasets[2]  # mixed
    X_c = np.asarray(X_c)
    X_tr_c, X_te_c, y_tr, y_te = train_test_split(X_c, y, test_size=0.2, random_state=42)
    rng = np.random.RandomState(77)

    print(f"\n  {'sigma':>6s}  {'Base':>8s}  {'Ek4 nf=0':>10s}  {'Ek4 nf=.3':>10s}"
          f"  {'k2':>3s} {'k3':>3s} {'k4':>3s} | {'k2':>3s} {'k3':>3s} {'k4':>3s}")
    print("  " + "-" * 72)

    for sigma in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
        tn = sigma * rng.randn(*X_tr_c.shape)
        ten = sigma * rng.randn(*X_te_c.shape)
        Xtr, Xte = X_tr_c + tn, X_te_c + ten

        base = GeoXGBRegressor(n_rounds=500, random_state=42)
        base.fit(Xtr, y_tr)
        br2 = r2_score(y_te, base.predict(Xte))

        ek_open = AdaptiveEkGeoXGBRegressor(
            max_degree=4, top_k_max=10, noise_alpha=1.5,
            noise_floor=0.0, n_rounds=500, random_state=42)
        ek_open.fit(Xtr, y_tr)
        or2 = r2_score(y_te, ek_open.predict(Xte))
        so = ek_open.ek_summary()

        ek_gate = AdaptiveEkGeoXGBRegressor(
            max_degree=4, top_k_max=10, noise_alpha=1.5,
            noise_floor=0.3, n_rounds=500, random_state=42)
        ek_gate.fit(Xtr, y_tr)
        gr2 = r2_score(y_te, ek_gate.predict(Xte))
        sg = ek_gate.ek_summary()

        def bud(s, k): return s['degrees'].get(k,{}).get('effective_k',0)

        print(f"  {sigma:6.2f}  {br2:8.4f}  {or2:10.4f}  {gr2:10.4f}"
              f"  {bud(so,2):3d} {bud(so,3):3d} {bud(so,4):3d}"
              f" | {bud(sg,2):3d} {bud(sg,3):3d} {bud(sg,4):3d}")
