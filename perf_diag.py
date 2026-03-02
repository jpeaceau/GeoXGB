"""
GeoXGB per-component timing diagnostic.

Measures the cost of each major operation in do_resample() by comparing
n_rounds=1 vs n_rounds=2+, isolating:
  A. Baseline  (no variance_ordered, no expansion)
  B. +variance_ordered  (reduce_ratio=0.7, variance_weighted=True)
  C. +KDE expansion     (expand_ratio=0.1, reduce_ratio=1.0)
  D. Full               (B + C)
  E. +n_partitions=50   (many small partitions → smaller BLAS matrices)
  F. +n_partitions=2    (few huge partitions → larger BLAS matrices)

Also compares Cal housing vs random data and tracks partition sizes.
"""
import sys
import time
import numpy as np
from sklearn.datasets import fetch_california_housing

sys.path.insert(0, r"C:\Users\jakep\ProofOfConcept\GeoXGB\src")
from geoxgb._cpp_backend import CppGeoXGBRegressor, make_cpp_config, _CPP_AVAILABLE

assert _CPP_AVAILABLE, "Need C++ backend"

# ── Datasets ─────────────────────────────────────────────────────────────────

np.random.seed(0)
_N, _D = 5000, 8

def load_cal():
    ds = fetch_california_housing()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds.target), _N, replace=False)
    X = ds.data[idx].astype(np.float64)
    y = ds.target[idx].astype(np.float64)
    y = (y - y.mean()) / y.std()
    return X, y

def load_rand():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((_N, _D))
    y = rng.standard_normal(_N)
    return X, y

X_cal, y_cal = load_cal()
X_rand, y_rand = load_rand()

# ── Timing helpers ────────────────────────────────────────────────────────────

def _time_fit(cfg_kwargs, X, y, n_rounds_list, n_warmup=1, n_rep=5):
    """
    For each n_rounds in n_rounds_list, time cfg_kwargs fit and return
    (mean_ms, std_ms) after n_rep runs (with n_warmup warmup).
    """
    results = {}
    for nr in n_rounds_list:
        kw = {**cfg_kwargs, "n_rounds": nr}
        cfg = make_cpp_config(**kw)
        # Warmup
        for _ in range(n_warmup):
            m = CppGeoXGBRegressor(cfg)
            m.fit(X, y)
        # Measure
        times = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            m = CppGeoXGBRegressor(cfg)
            m.fit(X, y)
            times.append((time.perf_counter() - t0) * 1e3)
        results[nr] = (float(np.mean(times)), float(np.std(times)))
    return results


def print_comp(name, results, n1, n2, refit_interval):
    """
    Print timing breakdown: initial do_resample, refit do_resample cost,
    and steady-state cost per additional refit.
    """
    t1, s1 = results[n1]
    t2, s2 = results[n2]
    # n_resample(n_rounds) = 1 + floor(n_rounds / refit_interval)
    def n_rs(nr):
        return 1 + nr // refit_interval

    rs1 = n_rs(n1)
    rs2 = n_rs(n2)
    delta_rs = rs2 - rs1

    gbt_per_round_ms = 0.3  # estimate for fast GBT tree builds on reduced n
    gbt_extra_ms = (n2 - n1) * gbt_per_round_ms

    refit_cost = (t2 - t1 - gbt_extra_ms) / delta_rs if delta_rs > 0 else float("nan")
    print(f"  {name}")
    print(f"    n_rounds={n1}: {t1:.1f}±{s1:.1f} ms  (n_resample={rs1})")
    print(f"    n_rounds={n2}: {t2:.1f}±{s2:.1f} ms  (n_resample={rs2})")
    if delta_rs > 0:
        print(f"    => per-refit cost: {refit_cost:.1f} ms  (delta_rounds={n2-n1}, delta_rs={delta_rs})")


# ── Config variants ───────────────────────────────────────────────────────────

refit_interval = 5

BASE_SHARED = dict(
    max_depth=2, min_samples_leaf=5,
    refit_interval=refit_interval,
    auto_noise=False, noise_guard=False, refit_noise_floor=0.0,
    auto_expand=False, min_train_samples=5000,
    bandwidth="auto", n_bins=64, random_state=42,
    hvrt_min_samples_leaf=-1,  # auto
)

CONFIGS = {
    "A_baseline      (reduce=1.0, vw=False, expand=0.0)": dict(
        **BASE_SHARED,
        reduce_ratio=1.0, variance_weighted=False, expand_ratio=0.0,
        n_partitions=-1,
    ),
    "B_var_ordered   (reduce=0.7, vw=True,  expand=0.0)": dict(
        **BASE_SHARED,
        reduce_ratio=0.7, variance_weighted=True, expand_ratio=0.0,
        n_partitions=-1,
    ),
    "C_expand_only   (reduce=1.0, vw=False, expand=0.1)": dict(
        **BASE_SHARED,
        reduce_ratio=1.0, variance_weighted=False, expand_ratio=0.1,
        n_partitions=-1,
    ),
    "D_full          (reduce=0.7, vw=True,  expand=0.1)": dict(
        **BASE_SHARED,
        reduce_ratio=0.7, variance_weighted=True, expand_ratio=0.1,
        n_partitions=-1,
    ),
    "E_many_parts    (reduce=0.7, vw=True,  np=50)": dict(
        **BASE_SHARED,
        reduce_ratio=0.7, variance_weighted=True, expand_ratio=0.0,
        n_partitions=50,  # forces small partitions ~100 samples each
    ),
    "F_few_parts     (reduce=0.7, vw=True,  np=4)": dict(
        **BASE_SHARED,
        reduce_ratio=0.7, variance_weighted=True, expand_ratio=0.0,
        n_partitions=4,  # forces large partitions ~1250 samples each
    ),
    "G_serial_like   (reduce=0.7, vw=False, expand=0.0)": dict(
        **BASE_SHARED,
        reduce_ratio=0.7, variance_weighted=False, expand_ratio=0.0,
        n_partitions=-1,
    ),
}

N_ROUNDS_PAIRS = [(1, 6)]  # n_rounds=1 → 1 resample; n_rounds=6 → 2 resamples with interval=5

# ── Run ───────────────────────────────────────────────────────────────────────

print("=" * 70)
print(f"GeoXGB per-component timing diagnostic   n={_N} d={_D}")
print(f"refit_interval={refit_interval}  n_rep=5  n_warmup=1")
print("=" * 70)

for ds_name, (X, y) in [("cal_housing", (X_cal, y_cal)), ("random     ", (X_rand, y_rand))]:
    print(f"\n{'='*70}")
    print(f"  Dataset: {ds_name}")
    print(f"{'='*70}")

    for cfg_name, cfg_kwargs in CONFIGS.items():
        try:
            n1, n2 = N_ROUNDS_PAIRS[0]
            results = _time_fit(cfg_kwargs, X, y, [n1, n2], n_warmup=1, n_rep=5)
            print_comp(cfg_name, results, n1, n2, refit_interval)
        except Exception as e:
            print(f"  {cfg_name}  ERROR: {e}")
        print()

# ── Partition size inspection ─────────────────────────────────────────────────

print("=" * 70)
print("Partition size inspection (auto_tune, n=5000, d=8)")
print("=" * 70)

for ds_name, (X, y) in [("cal_housing", (X_cal, y_cal)), ("random     ", (X_rand, y_rand))]:
    cfg = make_cpp_config(**{**BASE_SHARED, "reduce_ratio": 0.7, "variance_weighted": True,
                              "expand_ratio": 0.0, "n_partitions": -1, "n_rounds": 1})
    m = CppGeoXGBRegressor(cfg)
    m.fit(X, y)
    pids = np.array(m.partition_ids_)
    unique, counts = np.unique(pids, return_counts=True)
    print(f"\n  {ds_name}:  n_partitions={len(unique)}")
    print(f"    sizes: min={counts.min()}  max={counts.max()}  mean={counts.mean():.0f}  "
          f"std={counts.std():.0f}")
    print(f"    sizes sorted: {sorted(counts.tolist(), reverse=True)[:10]} ...")
