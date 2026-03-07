"""
Large-n Investigation
=====================
Diagnoses why GeoXGB falls behind XGBoost at n>=10k / d>=10.

Three focused experiments, all at the hardest cell (n=10k or 50k, d=20):

  Exp 1 — Partitioner sweep
    pyramid_hart (default) vs hvrt vs fasthvrt vs XGB-matched
    at n in {10000, 50000}, d=20.
    Q: does a different HVRT variant recover R² when block cycling limits data/refit?

  Exp 2 — Block size sweep
    sample_block_n in {None, large, mid, small, auto} at n=10000 & n=50000, d=20.
    Q: how much R² is lost as the per-refit view shrinks?

  Exp 3 — Geometry bypass (d_geom_threshold)
    d_geom_threshold=21 skips HVRT entirely -> pure GBT path.
    at n in {10000, 50000}, d=20 vs pyramid_hart default vs XGB-matched.
    Q: is geometry net-positive or net-negative at large n?

Protocol: 3-fold KFold(shuffle, seed=42), R², mean fit time per fold.
XGB-matched reference: n_estimators=1000, lr=0.02, depth=3 (capacity-matched).
"""
import sys, os, time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from geoxgb import GeoXGBRegressor

# ---------------------------------------------------------------------------
# Geometry-bypass subclass (exposes d_geom_threshold without touching source)
# ---------------------------------------------------------------------------
class _GeoXGBBypass(GeoXGBRegressor):
    """Same as GeoXGBRegressor but d_geom_threshold is passed to C++ backend."""
    _PARAM_NAMES = GeoXGBRegressor._PARAM_NAMES + ("d_geom_threshold",)

    def __init__(self, d_geom_threshold=0, **kwargs):
        super().__init__(**kwargs)
        self.d_geom_threshold = d_geom_threshold

# ---------------------------------------------------------------------------
# DGP: Friedman #1 + noise dims
# ---------------------------------------------------------------------------
def make_friedman_extended(n, d, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3]
         + 5  * X[:, 4]
         + rng.standard_normal(n))
    return X, y

# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------
N_ROUNDS = 1000
N_SPLITS = 3
SEED     = 42

def run_kfold(model_fn, X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores, times = [], []
    for tr, val in kf.split(X):
        m = model_fn()
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        times.append(time.perf_counter() - t0)
        scores.append(r2_score(y[val], m.predict(X[val])))
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(times))

# XGB-matched baseline (capacity-matched reference used across all experiments)
def xgbm_fn():
    return xgb.XGBRegressor(n_estimators=N_ROUNDS, learning_rate=0.02,
                             max_depth=3, random_state=SEED, n_jobs=-1, verbosity=0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row(label, r, s, t, r_ref, t_ref):
    delta = r - r_ref
    ratio = t / t_ref if t_ref > 0 else float("nan")
    return (label, r, s, t, delta, ratio)

def _print_table(title, header, rows):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    col_w = [max(len(header[i]), max(len(str(r[i])) for r in rows))
             for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*header))
    print("  ".join("-"*w for w in col_w))
    for row in rows:
        print(fmt.format(*row))

def _resolve_block(n, val):
    """Resolve 'auto' to the same formula used by the regressor."""
    if val == "auto":
        return None if n <= 5000 else 500 + (n - 5000) // 50
    return val

# ---------------------------------------------------------------------------
# Experiment 1: Partitioner sweep
# ---------------------------------------------------------------------------
def exp1_partitioner_sweep():
    print("\n\n### Experiment 1 — Partitioner sweep (n=10k & 50k, d=20) ###")
    partitioners = ["pyramid_hart", "hvrt", "fasthvrt"]
    cells = [(10_000, 20), (50_000, 20)]
    all_rows = []

    for n, d in cells:
        X, y = make_friedman_extended(n, d)
        r_ref, s_ref, t_ref = run_kfold(xgbm_fn, X, y)
        print(f"\n  n={n}, d={d}  |  XGB-matched: R2={r_ref:.4f}+/-{s_ref:.4f}  t={t_ref:.2f}s")

        for part in partitioners:
            def _fn(p=part):
                return GeoXGBRegressor(n_rounds=N_ROUNDS, learning_rate=0.02,
                                       max_depth=3, sample_block_n="auto",
                                       partitioner=p, random_state=SEED)
            r, s, t = run_kfold(_fn, X, y)
            delta = r - r_ref
            sign = "+" if delta >= 0 else ""
            print(f"    {part:<16} R2={r:.4f}+/-{s:.4f}  t={t:.2f}s  vs XGB-mat: {sign}{delta:.4f}")
            all_rows.append((n, d, part, f"{r:.4f}", f"{s:.4f}", f"{t:.2f}s",
                             f"{delta:+.4f}", f"{r_ref:.4f}"))

    _print_table(
        "Exp 1: Partitioner sweep (XGB-matched R2 shown in last col)",
        ["n", "d", "partitioner", "R2", "std", "t/fold", "vs XGB-mat", "XGB-mat R2"],
        all_rows,
    )

# ---------------------------------------------------------------------------
# Experiment 2: Block size sweep
# ---------------------------------------------------------------------------
def exp2_block_sweep():
    print("\n\n### Experiment 2 — Block size sweep ###")
    # (n, d, block_values_to_test)
    configs = [
        (10_000, 20, [None, 5000, 2000, 1000, "auto"]),
        (50_000, 20, [None, 20000, 10000, 5000, 2000, "auto"]),
    ]
    all_rows = []

    for n, d, blocks in configs:
        X, y = make_friedman_extended(n, d)
        r_ref, _, t_ref = run_kfold(xgbm_fn, X, y)
        print(f"\n  n={n}, d={d}  |  XGB-matched: R2={r_ref:.4f}  t={t_ref:.2f}s")

        for blk in blocks:
            resolved = _resolve_block(n, blk)
            label = "None (full)" if blk is None else (f"auto({resolved})" if blk == "auto" else str(blk))

            def _fn(b=blk):
                return GeoXGBRegressor(n_rounds=N_ROUNDS, learning_rate=0.02,
                                       max_depth=3, sample_block_n=b,
                                       random_state=SEED)
            r, s, t = run_kfold(_fn, X, y)
            delta = r - r_ref
            sign = "+" if delta >= 0 else ""
            print(f"    block={label:<14} R2={r:.4f}+/-{s:.4f}  t={t:.2f}s  vs XGB-mat: {sign}{delta:.4f}")
            all_rows.append((n, d, label, resolved if resolved is not None else "full",
                             f"{r:.4f}", f"{s:.4f}", f"{t:.2f}s", f"{delta:+.4f}"))

    _print_table(
        "Exp 2: Block size sweep",
        ["n", "d", "sample_block_n", "resolved", "R2", "std", "t/fold", "vs XGB-mat"],
        all_rows,
    )

# ---------------------------------------------------------------------------
# Experiment 3: Geometry bypass (d_geom_threshold)
# ---------------------------------------------------------------------------
def exp3_geometry_bypass():
    print("\n\n### Experiment 3 — Geometry bypass (d_geom_threshold) ###")
    cells = [(10_000, 20), (50_000, 20)]
    all_rows = []

    for n, d in cells:
        X, y = make_friedman_extended(n, d)
        r_ref, s_ref, t_ref = run_kfold(xgbm_fn, X, y)

        # Default GeoXGB
        def _geo_fn():
            return GeoXGBRegressor(n_rounds=N_ROUNDS, learning_rate=0.02,
                                   max_depth=3, sample_block_n="auto",
                                   random_state=SEED)
        r_geo, s_geo, t_geo = run_kfold(_geo_fn, X, y)

        # Pure GBT path (d_geom_threshold > d bypasses HVRT entirely)
        def _bypass_fn(threshold=d + 1):
            return _GeoXGBBypass(d_geom_threshold=threshold,
                                  n_rounds=N_ROUNDS, learning_rate=0.02,
                                  max_depth=3, sample_block_n="auto",
                                  random_state=SEED)
        r_byp, s_byp, t_byp = run_kfold(_bypass_fn, X, y)

        print(f"\n  n={n}, d={d}")
        print(f"    XGB-matched          R2={r_ref:.4f}+/-{s_ref:.4f}  t={t_ref:.2f}s")
        print(f"    GeoXGB (pyramid)     R2={r_geo:.4f}+/-{s_geo:.4f}  t={t_geo:.2f}s  delta={r_geo-r_ref:+.4f}")
        print(f"    GeoXGB (bypass d_thr={d+1}) R2={r_byp:.4f}+/-{s_byp:.4f}  t={t_byp:.2f}s  delta={r_byp-r_ref:+.4f}")

        all_rows.append((n, d, "XGB-matched",       f"{r_ref:.4f}", f"{s_ref:.4f}", f"{t_ref:.2f}s", "---"))
        all_rows.append((n, d, "GeoXGB pyramid_hart", f"{r_geo:.4f}", f"{s_geo:.4f}", f"{t_geo:.2f}s", f"{r_geo-r_ref:+.4f}"))
        all_rows.append((n, d, f"GeoXGB bypass d>{d}", f"{r_byp:.4f}", f"{s_byp:.4f}", f"{t_byp:.2f}s", f"{r_byp-r_ref:+.4f}"))

    _print_table(
        "Exp 3: Geometry bypass (d_geom_threshold)",
        ["n", "d", "model", "R2", "std", "t/fold", "vs XGB-mat"],
        all_rows,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    exp1_partitioner_sweep()
    exp2_block_sweep()
    exp3_geometry_bypass()
    print("\n\nDone.")
