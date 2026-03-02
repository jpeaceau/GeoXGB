"""
GeoXGB Component Timing Benchmark
==================================
Measures individual components in isolation to identify bottlenecks.

Usage:
    cd C:/Users/jakep/ProofOfConcept/GeoXGB
    python benchmarks/micro/component_timing.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time
import numpy as np
from sklearn.datasets import make_friedman1

N_SAMPLES = 5000
N_FEATURES = 10
N_REPEAT = 15
WARMUP = 3

rng = np.random.RandomState(42)
X, y = make_friedman1(n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=0)

print("=" * 68)
print("  GeoXGB Component Timing Benchmark")
print("=" * 68)
print(f"  Dataset: n={N_SAMPLES}, d={N_FEATURES}, n_repeat={N_REPEAT}")
print()

def measure(label, fn, n_repeat=N_REPEAT, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(1000 * (time.perf_counter() - t0))
    arr = np.array(times)
    print(f"  {label:<46} {arr.mean():7.2f} +/- {arr.std():5.2f} ms")
    return arr


# -- Section 1: Python-side proxy benchmarks ----------------------------------
print("-- Python kNN proxy (mirrors C++ knn_assign_y) ----------------------")

n_red = int(N_SAMPLES * 0.7)   # 3500 -- reduce_ratio=0.7
n_syn = int(N_SAMPLES * 0.1)   # 500  -- expand_ratio=0.1
d = N_FEATURES

X_syn_z = rng.randn(n_syn, d).astype(np.float64)
X_red_z = rng.randn(n_red, d).astype(np.float64)
y_red   = rng.randn(n_red)
k = 3

D_buf = np.empty((n_syn, n_red), dtype=np.float64)  # persistent buffer
norm_syn_buf = (X_syn_z ** 2).sum(1)
norm_red_buf = (X_red_z ** 2).sum(1)

def knn_fresh_alloc():
    """Allocates D fresh each call -- mirrors original C++ before fix."""
    norm_syn = (X_syn_z ** 2).sum(1)
    norm_red = (X_red_z ** 2).sum(1)
    D = X_syn_z @ X_red_z.T          # fresh allocation: ~14 MB
    D *= -2
    D += norm_syn[:, None]
    D += norm_red[None, :]
    np.maximum(D, 0, out=D)
    np.argpartition(D, k, axis=1)[:, :k]

def knn_persistent():
    """Reuses pre-allocated D -- mirrors fixed C++ (mutable knn_D_)."""
    # Use np.* with out= instead of augmented assignment to avoid Python closure issue.
    np.dot(X_syn_z, X_red_z.T, out=D_buf)
    np.multiply(D_buf, -2, out=D_buf)
    np.add(D_buf, norm_syn_buf[:, None], out=D_buf)
    np.add(D_buf, norm_red_buf[None, :], out=D_buf)
    np.maximum(D_buf, 0, out=D_buf)
    np.argpartition(D_buf, k, axis=1)[:, :k]

def gemm_only():
    """Isolated GEMM -- eliminates sort and norm cost."""
    return X_syn_z @ X_red_z.T          # fresh allocation

def gemm_persistent():
    """Isolated GEMM with persistent output."""
    np.dot(X_syn_z, X_red_z.T, out=D_buf)

print(f"  (D matrix size: {n_syn}x{n_red}x8 bytes = {n_syn*n_red*8/1e6:.1f} MB)")
a = measure("kNN full  (fresh alloc each call)", knn_fresh_alloc)
b = measure("kNN full  (persistent D buffer)  ", knn_persistent)
measure("GEMM only (fresh alloc)          ", gemm_only)
measure("GEMM only (persistent)           ", gemm_persistent)
print(f"  Buffer speedup: {a.mean()/b.mean():.2f}x")
print()


# -- Section 2: Python HVRT component breakdown -------------------------------
print("-- Python HVRT components -------------------------------------------")
try:
    from hvrt import HVRT as PyHVRT
    h = PyHVRT(y_weight=0.5, random_state=0, auto_tune=True)
    t0 = time.perf_counter()
    h.fit(X, y)
    print(f"  {'HVRT full fit (one-time)':<46} {1000*(time.perf_counter()-t0):7.2f} ms")

    y_noisy = y + rng.randn(N_SAMPLES) * 0.1
    try:
        measure("HVRT refit(y)                    ", lambda: h.refit(y_noisy))
    except AttributeError:
        print("  HVRT refit: not available in Python HVRT")

    X_rand = rng.randn(500, N_FEATURES)
    measure("HVRT expand(500, epanechnikov)   ", lambda: h.expand(
        500, variance_weighted=True, generation_strategy='epanechnikov'))

    red_idx = rng.choice(N_SAMPLES, n_red, replace=False)
    try:
        X_z_np = np.asarray(h.X_z_)
        measure("X_red_z row gather from X_z_     ", lambda: X_z_np[red_idx])
    except AttributeError:
        print("  X_z_ not accessible")

except Exception as e:
    print(f"  (Python HVRT unavailable: {e})")

print()


# -- Section 3: Full GeoXGB fit -- component isolation via config -------------
print("-- Full C++ GeoXGB fit (kNN cost vs expand_ratio) -------------------")

from geoxgb._cpp_backend import CppGeoXGBRegressor, make_cpp_config

BASE_CFG = dict(
    learning_rate=0.05,
    max_depth=2,
    min_samples_leaf=5,
    reduce_ratio=0.7,
    y_weight=0.5,
    refit_interval=5,
    auto_expand=True,
    min_train_samples=5000,
    n_bins=64,
    random_state=0,
)

def run_cpp(n_rounds, expand_ratio):
    cfg = make_cpp_config(n_rounds=n_rounds, expand_ratio=expand_ratio, **BASE_CFG)
    CppGeoXGBRegressor(cfg).fit(X, y)

# Warm up JIT/OS effects once before timing
run_cpp(10, 0.0)

for n_rounds in [100, 1000]:
    # Initial resample always happens; refits at i=5,10,...,(n_rounds-1 rounded down to 5)
    # n_resample = 1 + floor((n_rounds - 1) / 5)
    n_resample = 1 + (n_rounds - 1) // 5

    t_no_exp = measure(f"er=0.0 n={n_rounds:4d} ({n_resample:3d} refits, no kNN)",
                       lambda nr=n_rounds: run_cpp(nr, 0.0), n_repeat=5, warmup=1)
    t_with_exp = measure(f"er=0.1 n={n_rounds:4d} ({n_resample:3d} expands + kNN)",
                         lambda nr=n_rounds: run_cpp(nr, 0.1), n_repeat=5, warmup=1)

    knn_total = t_with_exp.mean() - t_no_exp.mean()
    knn_per   = knn_total / n_resample
    print(f"    -> kNN total: {knn_total:.1f} ms / {n_resample} calls = {knn_per:.2f} ms/call")
    print()

# -- Section 4: HVRT refit cost isolation ------------------------------------
print("-- HVRT refit cost (refit_interval=5 vs disabled) -------------------")

def run_cpp_ri(n_rounds, refit_interval):
    cfg = make_cpp_config(n_rounds=n_rounds, expand_ratio=0.0,
                          refit_interval=refit_interval, **{k: v for k, v in BASE_CFG.items()
                                                            if k != 'refit_interval'})
    CppGeoXGBRegressor(cfg).fit(X, y)

for n_rounds in [100, 500, 1000]:
    # refit_interval=0 disables refits; refit_interval=5 enables them
    n_refits = (n_rounds - 1) // 5   # number of periodic refits (not counting initial)

    t_no_refit = measure(f"n={n_rounds:4d} ri=0  (0 refits, GBT only)     ",
                         lambda nr=n_rounds: run_cpp_ri(nr, 0), n_repeat=5, warmup=1)
    t_refit5   = measure(f"n={n_rounds:4d} ri=5  ({n_refits:3d} refits)           ",
                         lambda nr=n_rounds: run_cpp_ri(nr, 5), n_repeat=5, warmup=1)

    refit_total = t_refit5.mean() - t_no_refit.mean()
    refit_per   = refit_total / n_refits if n_refits > 0 else 0
    gbt_per     = t_no_refit.mean() / n_rounds
    print(f"    -> GBT/round: {gbt_per:.2f} ms | "
          f"refit overhead: {refit_total:.1f} ms / {n_refits} = {refit_per:.2f} ms/refit")
    print()

print("Done.")
