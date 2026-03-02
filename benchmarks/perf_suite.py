"""
GeoXGB Performance Suite
========================
Single authoritative benchmark. Run after every code change to catch regressions
and verify improvements.

Metrics tracked:
  gbt_ms      -- GBT tree build per round (no HVRT, baseline compute cost)
  refit_ms    -- HVRT refit cost per call (dominant cost at ri=5)
  knn_ms      -- kNN y-assign per expand call (expand_ratio=0.1)
  fit_500_ms  -- Full fit, 500 rounds, er=0.1, ri=5 (synthetic end-to-end)
  r2_friedman -- Mean R² on friedman1, 3-fold CV (accuracy guard)

Usage:
  python benchmarks/perf_suite.py           # run and compare to baselines
  python benchmarks/perf_suite.py --update  # accept current numbers as new baseline
  python benchmarks/perf_suite.py --quiet   # timing only, skip accuracy CV

Exits 0 if all checks pass, 1 if any metric regresses beyond tolerance.
"""
import sys
import io
import json
import time
import argparse
import pathlib
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASELINE_FILE = pathlib.Path(__file__).parent / "baselines.json"

# -- Tolerances ---------------------------------------------------------------
# Timing: 20% band (machine load variance). Accuracy: 0.02 absolute.
TIMING_TOL  = 0.20   # 20% above baseline → regression
ACCURACY_TOL = 0.02  # 0.02 below baseline R² → regression

# -- Dataset ------------------------------------------------------------------
N_SAMPLES  = 5_000
N_FEATURES = 10
N_ROUNDS   = 500   # fast but representative (< 15 s total)
N_REPEAT   = 5
WARMUP     = 1

X_all, y_all = make_friedman1(n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=0)


# -- Helpers ------------------------------------------------------------------
def timeit(fn, n_repeat=N_REPEAT, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(1000.0 * (time.perf_counter() - t0))
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def hdr(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def row(label, val, std, baseline=None, tol=TIMING_TOL):
    if baseline is None:
        tag = " [NEW]"
        delta = ""
    else:
        pct = (val - baseline) / max(baseline, 1e-9)
        tag = " [PASS]" if pct <= tol else " [FAIL]"
        sign = "+" if pct >= 0 else ""
        delta = f"  baseline={baseline:.2f}  delta={sign}{100*pct:.1f}%"
    print(f"  {label:<32} {val:8.2f} +/- {std:5.2f} ms{delta}{tag}")
    return tag.strip() == "[FAIL]"


# -- Import backend -----------------------------------------------------------
from geoxgb._cpp_backend import CppGeoXGBRegressor, make_cpp_config

BASE_CFG = dict(
    learning_rate  = 0.05,
    max_depth      = 2,
    min_samples_leaf = 5,
    reduce_ratio   = 0.7,
    y_weight       = 0.5,
    refit_interval = 5,
    auto_expand    = True,
    min_train_samples = 5_000,
    n_bins         = 64,
    random_state   = 0,
)


def make_cfg(**overrides):
    kw = {**BASE_CFG, **overrides}
    n_rounds = kw.pop("n_rounds", N_ROUNDS)
    return make_cpp_config(n_rounds=n_rounds, **kw)


def fit(n_rounds=N_ROUNDS, expand_ratio=0.0, refit_interval=5):
    # Strip keys that are passed explicitly so BASE_CFG doesn't duplicate them.
    base = {k: v for k, v in BASE_CFG.items()
            if k not in ("refit_interval",)}
    cfg = make_cpp_config(n_rounds=n_rounds, expand_ratio=expand_ratio,
                          refit_interval=refit_interval, **base)
    CppGeoXGBRegressor(cfg).fit(X_all, y_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="Save current results as new baseline")
    parser.add_argument("--quiet", action="store_true",
                        help="Skip accuracy CV (timing only)")
    args = parser.parse_args()

    baselines = {}
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            baselines = json.load(f)

    results   = {}
    any_fail  = False

    # ── Warm-up ──────────────────────────────────────────────────────────────
    print("Warming up JIT/OS effects...")
    fit(n_rounds=50, expand_ratio=0.0, refit_interval=0)
    fit(n_rounds=50, expand_ratio=0.1, refit_interval=5)

    # ── Section 1: GBT baseline (no HVRT at all) ─────────────────────────────
    hdr("Section 1 — GBT baseline (refit=0, er=0.0)")
    mean, std = timeit(lambda: fit(N_ROUNDS, expand_ratio=0.0, refit_interval=0))
    gbt_per   = mean / N_ROUNDS
    gbt_std   = std  / N_ROUNDS
    results["gbt_ms"] = gbt_per
    bl = baselines.get("gbt_ms")
    any_fail |= row("GBT per round", gbt_per, gbt_std, bl)

    # ── Section 2: HVRT refit cost ───────────────────────────────────────────
    hdr("Section 2 — HVRT refit overhead (er=0.0, ri=5 vs ri=0)")
    n_refits   = (N_ROUNDS - 1) // 5
    mean_ri5,  std_ri5  = timeit(lambda: fit(N_ROUNDS, 0.0, refit_interval=5))
    mean_ri0,  std_ri0  = timeit(lambda: fit(N_ROUNDS, 0.0, refit_interval=0))
    refit_total = mean_ri5 - mean_ri0
    refit_per   = refit_total / max(n_refits, 1)
    refit_std   = (std_ri5**2 + std_ri0**2)**0.5 / max(n_refits, 1)
    results["refit_ms"] = refit_per
    bl = baselines.get("refit_ms")
    any_fail |= row("HVRT refit per call", refit_per, refit_std, bl)
    print(f"    ({n_refits} refits, total overhead: {refit_total:.1f} ms)")

    # ── Section 3: kNN cost ──────────────────────────────────────────────────
    hdr("Section 3 — kNN y-assign overhead (er=0.1 vs er=0.0, ri=5)")
    mean_exp,  std_exp  = timeit(lambda: fit(N_ROUNDS, 0.1, refit_interval=5))
    n_expands  = n_refits  # expand every refit call
    knn_total  = mean_exp - mean_ri5
    knn_per    = knn_total / max(n_expands, 1)
    knn_std    = (std_exp**2 + std_ri5**2)**0.5 / max(n_expands, 1)
    results["knn_ms"] = knn_per
    bl = baselines.get("knn_ms")
    any_fail |= row("kNN y-assign per call", knn_per, knn_std, bl)
    print(f"    ({n_expands} expands, total: {knn_total:.1f} ms)")

    # ── Section 4: End-to-end fit ─────────────────────────────────────────────
    hdr("Section 4 — Full end-to-end fit (er=0.1, ri=5)")
    results["fit_500_ms"] = mean_exp
    bl = baselines.get("fit_500_ms")
    any_fail |= row(f"Full fit ({N_ROUNDS} rounds)", mean_exp, std_exp, bl)

    # ── Section 5: Accuracy guard ────────────────────────────────────────────
    if not args.quiet:
        hdr("Section 5 — Accuracy (friedman1, 3-fold CV)")
        r2_scores = []
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for fold, (tr, va) in enumerate(kf.split(X_all)):
            cfg = make_cpp_config(n_rounds=300, expand_ratio=0.1, **BASE_CFG)
            m = CppGeoXGBRegressor(cfg)
            m.fit(X_all[tr], y_all[tr])
            preds = m.predict(X_all[va])
            r2_scores.append(r2_score(y_all[va], preds))
            print(f"  fold {fold+1}: R²={r2_scores[-1]:.4f}")
        mean_r2 = float(np.mean(r2_scores))
        std_r2  = float(np.std(r2_scores))
        results["r2_friedman"] = mean_r2
        bl = baselines.get("r2_friedman")
        if bl is None:
            tag = "[NEW]"
        else:
            tag = "[PASS]" if mean_r2 >= bl - ACCURACY_TOL else "[FAIL]"
            if tag == "[FAIL]":
                any_fail = True
        bl_str = f"  baseline={bl:.4f}" if bl is not None else ""
        print(f"  {'Mean R²':<32} {mean_r2:.4f} +/- {std_r2:.4f}{bl_str}  {tag}")

    # ── Summary ──────────────────────────────────────────────────────────────
    hdr("Summary")
    if args.update:
        with open(BASELINE_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Baselines updated -> {BASELINE_FILE}")
        for k, v in results.items():
            print(f"    {k}: {v:.4f}")
    else:
        status = "REGRESSION DETECTED" if any_fail else "ALL CHECKS PASSED"
        print(f"  Status: {status}")
        if not BASELINE_FILE.exists():
            print("  (No baseline file found — run with --update to set baseline)")

    print()
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
