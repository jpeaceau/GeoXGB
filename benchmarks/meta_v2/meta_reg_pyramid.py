"""
meta_reg_pyramid.py
====================
PyramidHART-focused OAT + pairwise hyperparameter study.

Phase 1 — OAT:   sweep max_depth, refit_interval, y_weight, method
                  against a PyramidHART baseline.
Phase 2 — Pairs: full grid for the top-2 parameters by OAT importance.

Metric: val_nmae = MAE / std(y_train)  — lower is better.
Importance = mean |Δnmae vs baseline| across all dataset × seed × fold evals.

Uses CppGeoXGBRegressor (C++ HVRT kernels) for speed; falls back to Python
GeoXGBRegressor if the extension is unavailable.

Usage
-----
  python meta_reg_pyramid.py            # both phases
  python meta_reg_pyramid.py --phase 1  # OAT only
  python meta_reg_pyramid.py --phase 2  # pairwise only (needs phase 1 CSV)
  python meta_reg_pyramid.py --summary  # print summaries from saved CSVs
  python meta_reg_pyramid.py --jobs N   # parallel workers (default: cpu_count)
"""
from __future__ import annotations

import argparse
import csv
import io
import multiprocessing as mp
import os
import sys
import time
import warnings
from itertools import combinations

import numpy as np
from sklearn.datasets import make_friedman1, make_friedman2, make_regression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_OAT   = os.path.join(RESULTS_DIR, "pyramid_oat.csv")
CSV_PAIR  = os.path.join(RESULTS_DIR, "pyramid_pairwise.csv")

# ── Baseline (PyramidHART, depth=4, all empirically validated) ────────────────

BASELINE: dict = dict(
    n_rounds            = 1000,
    learning_rate       = 0.02,
    max_depth           = 4,
    min_samples_leaf    = 5,
    reduce_ratio        = 0.8,
    expand_ratio        = 0.1,
    y_weight            = 0.5,
    refit_interval      = 5,
    auto_noise          = False,
    noise_guard         = False,
    variance_weighted   = False,
    random_state        = 42,
    partitioner         = "pyramid_hart",
    method              = "orthant_stratified",
    generation_strategy = "simplex_mixup",
    adaptive_reduce_ratio = True,
)

# ── OAT sweep grid ────────────────────────────────────────────────────────────

OAT_SWEEP: dict[str, list] = dict(
    max_depth       = [2, 3, 4, 5, 6],
    refit_interval  = [3, 5, 10, 20, 50],
    y_weight        = [0.0, 0.25, 0.5, 0.75, 1.0],
    method          = ["variance_ordered", "orthant_stratified",
                       "residual_stratified"],
)

# ── CV config ─────────────────────────────────────────────────────────────────

N_SEEDS_OAT  = 5
N_FOLDS_OAT  = 3
N_SEEDS_PAIR = 5
N_FOLDS_PAIR = 3

# ── CSV field schemas ─────────────────────────────────────────────────────────

OAT_FIELDS = [
    "phase", "dataset", "n_samples", "n_features",
    "param", "value",
    "seed", "fold",
    "val_nmae", "val_mae", "val_r2",
    "train_time_s", "status",
]

PAIR_FIELDS = [
    "phase", "dataset", "n_samples", "n_features",
    "param1", "val1", "param2", "val2",
    "seed", "fold",
    "val_nmae", "val_mae", "val_r2",
    "train_time_s", "status",
]

# ── Dataset registry (populated once in main process and workers) ─────────────

_DATASETS: dict = {}


def _make_datasets() -> dict:
    rng = np.random.RandomState(0)
    datasets = {}

    X1, y1 = make_friedman1(n_samples=1_000, n_features=10, noise=1.0, random_state=0)
    datasets["friedman1"] = (X1.astype(np.float32), y1.astype(np.float32))

    X2, y2 = make_friedman2(n_samples=1_000, noise=0.0, random_state=0)
    datasets["friedman2"] = (X2.astype(np.float32), y2.astype(np.float32))

    X3, y3 = make_regression(n_samples=2_000, n_features=30, n_informative=8,
                              noise=0.5, random_state=0)
    datasets["reg_sparse"] = (X3.astype(np.float32), y3.astype(np.float32))

    X4, y4 = make_regression(n_samples=5_000, n_features=20, n_informative=15,
                              noise=1.0, random_state=0)
    datasets["reg_large"] = (X4.astype(np.float32), y4.astype(np.float32))

    return datasets


def _worker_init() -> None:
    global _DATASETS
    warnings.filterwarnings("ignore")
    _DATASETS = _make_datasets()


# ── Trial worker ──────────────────────────────────────────────────────────────

def _worker(task: dict) -> dict:
    try:
        from geoxgb._cpp_backend import _CPP_AVAILABLE, make_cpp_config, CppGeoXGBRegressor
        _use_cpp = _CPP_AVAILABLE
    except ImportError:
        _use_cpp = False

    ds_name  = task["dataset"]
    seed     = task["seed"]
    fold_idx = task["fold"]
    n_folds  = task["n_folds"]
    params   = task["params"]

    X, y = _DATASETS[ds_name]
    n_samples, n_features = X.shape

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fi, (tr, va) in enumerate(kf.split(X)):
        if fi == fold_idx:
            train_idx, val_idx = tr, va
            break

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx],   y[val_idx]
    y_std = float(y_tr.std()) + 1e-12

    t0 = time.time()
    try:
        if _use_cpp:
            cfg   = make_cpp_config(**params)
            model = CppGeoXGBRegressor(cfg)
        else:
            from geoxgb import GeoXGBRegressor
            model = GeoXGBRegressor(**{k: v for k, v in params.items()
                                       if k != "n_bins"})
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        mae   = float(mean_absolute_error(y_va, preds))
        nmae  = mae / y_std
        ss_res = float(np.sum((y_va - preds)**2))
        ss_tot = float(np.sum((y_va - np.mean(y_va))**2)) + 1e-12
        r2    = float(1.0 - ss_res / ss_tot)
        status = "ok"
    except Exception as e:
        mae, nmae, r2 = float("nan"), float("nan"), float("nan")
        status = str(e)[:120]

    elapsed = time.time() - t0

    row = dict(
        phase       = task.get("phase", ""),
        dataset     = ds_name,
        n_samples   = n_samples,
        n_features  = n_features,
        param       = task.get("param", ""),
        value       = str(task.get("value", "")),
        param1      = task.get("param1", ""),
        val1        = str(task.get("val1", "")),
        param2      = task.get("param2", ""),
        val2        = str(task.get("val2", "")),
        seed        = seed,
        fold        = fold_idx,
        val_nmae    = nmae,
        val_mae     = mae,
        val_r2      = r2,
        train_time_s = elapsed,
        status      = status,
    )
    return row


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _load_done(csv_path: str, key_fields: list[str]) -> set:
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = tuple(row.get(k, "") for k in key_fields)
            done.add(key)
    return done


def _append_rows(csv_path: str, rows: list[dict], fields: list[str]) -> None:
    write_header = not os.path.exists(csv_path)
    mode = "a" if not write_header else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _run_pool(pool: mp.Pool, tasks: list[dict],
              csv_path: str, fields: list[str],
              key_fields: list[str]) -> list[dict]:
    done = _load_done(csv_path, key_fields)

    pending = []
    for t in tasks:
        key = tuple(str(t.get(k, "")) for k in key_fields)
        if key not in done:
            pending.append(t)

    n_total = len(tasks)
    n_skip  = n_total - len(pending)
    if n_skip:
        print(f"  Resuming: {n_skip}/{n_total} rows already done, "
              f"{len(pending)} remaining.")

    if not pending:
        return []

    batch_size = max(1, len(pending) // 50)
    buf: list[dict] = []
    all_rows: list[dict] = []
    n_done = 0

    for row in pool.imap_unordered(_worker, pending):
        buf.append(row)
        all_rows.append(row)
        n_done += 1
        if len(buf) >= batch_size:
            _append_rows(csv_path, buf, fields)
            buf = []
        if n_done % max(1, len(pending) // 20) == 0:
            pct = 100 * n_done / len(pending)
            print(f"  {pct:5.1f}%  ({n_done}/{len(pending)})", flush=True)

    if buf:
        _append_rows(csv_path, buf, fields)

    return all_rows


# ── Phase 1: OAT ──────────────────────────────────────────────────────────────

def _build_oat_tasks() -> list[dict]:
    tasks = []
    datasets = list(_make_datasets().keys())

    # Baseline rows (param="baseline", value="baseline")
    for ds in datasets:
        for seed in range(N_SEEDS_OAT):
            for fold in range(N_FOLDS_OAT):
                tasks.append(dict(
                    phase="oat", dataset=ds, seed=seed, fold=fold,
                    n_folds=N_FOLDS_OAT,
                    param="baseline", value="baseline",
                    params=dict(BASELINE),
                ))

    # OAT rows
    for param, values in OAT_SWEEP.items():
        for val in values:
            for ds in datasets:
                for seed in range(N_SEEDS_OAT):
                    for fold in range(N_FOLDS_OAT):
                        p = dict(BASELINE)
                        p[param] = val
                        tasks.append(dict(
                            phase="oat", dataset=ds, seed=seed, fold=fold,
                            n_folds=N_FOLDS_OAT,
                            param=param, value=val,
                            params=p,
                        ))

    return tasks


def _oat_key_fields() -> list[str]:
    return ["phase", "dataset", "param", "value", "seed", "fold"]


def run_oat(pool: mp.Pool) -> None:
    print("\n── Phase 1: OAT ──────────────────────────────────────────────────")
    tasks = _build_oat_tasks()
    print(f"  Total tasks: {len(tasks)}  "
          f"(datasets={len(_make_datasets())}  "
          f"seeds={N_SEEDS_OAT}  folds={N_FOLDS_OAT})")
    _run_pool(pool, tasks, CSV_OAT, OAT_FIELDS, _oat_key_fields())
    _print_oat_summary()


def _print_oat_summary() -> None:
    if not os.path.exists(CSV_OAT):
        print("  No OAT CSV found.")
        return

    rows: list[dict] = []
    with open(CSV_OAT, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    # Compute per-dataset baseline mean
    base: dict[str, list[float]] = {}
    for r in rows:
        if r["param"] == "baseline" and r["status"] == "ok":
            base.setdefault(r["dataset"], []).append(float(r["val_nmae"]))
    base_mean = {ds: float(np.mean(v)) for ds, v in base.items()}

    # Per param: importance = mean |Δnmae vs baseline| across all datasets/seeds/folds
    importance: dict[str, float] = {}
    best_val:   dict[str, tuple] = {}  # param -> (best_value, mean_nmae)

    for param in OAT_SWEEP:
        param_rows = [r for r in rows if r["param"] == param and r["status"] == "ok"]
        if not param_rows:
            continue

        # Group by value, compute mean delta
        val_deltas: dict[str, list[float]] = {}
        val_nmae:   dict[str, list[float]] = {}
        for r in param_rows:
            v    = r["value"]
            ds   = r["dataset"]
            nmae = float(r["val_nmae"])
            delta = nmae - base_mean.get(ds, nmae)
            val_deltas.setdefault(v, []).append(abs(delta))
            val_nmae.setdefault(v, []).append(nmae)

        importance[param] = float(np.mean([
            np.mean(dlist) for dlist in val_deltas.values()
        ]))

        best_v   = min(val_nmae, key=lambda v: np.mean(val_nmae[v]))
        best_val[param] = (best_v, float(np.mean(val_nmae[best_v])))

    print("\n  OAT IMPORTANCE (mean |Δnmae| vs baseline):")
    print(f"  {'Parameter':<20}  {'Importance':>12}  {'Best value':>18}  {'Best nmae':>10}")
    print(f"  {'-'*65}")
    for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bv, bn = best_val.get(param, ("?", float("nan")))
        print(f"  {param:<20}  {imp:>12.5f}  {str(bv):>18}  {bn:>10.5f}")

    print(f"\n  Baseline mean nmae: "
          f"{np.mean(list(base_mean.values())):.5f}")

    # Per-parameter value table
    print("\n  PER-PARAMETER VALUE TABLE (mean nmae across all datasets):")
    for param, values in OAT_SWEEP.items():
        print(f"\n  {param}:")
        print(f"    {'Value':<22}  {'Mean NMAE':>10}  {'vs baseline':>12}")
        base_all = float(np.mean(list(base_mean.values())))
        for val in (["baseline"] + [str(v) for v in values]):
            vrows = [r for r in rows
                     if r["param"] == ("baseline" if val == "baseline" else param)
                     and (val == "baseline" or r["value"] == val)
                     and r["status"] == "ok"]
            if not vrows:
                continue
            m = float(np.mean([float(r["val_nmae"]) for r in vrows]))
            vs = m - base_all
            tag = " *" if vs < -0.0005 else ""
            print(f"    {val:<22}  {m:>10.5f}  {vs:>+12.5f}{tag}")


# ── Phase 2: Pairwise ─────────────────────────────────────────────────────────

def _top_params(n: int = 2) -> list[str]:
    """Read OAT CSV and return top-n params by importance."""
    if not os.path.exists(CSV_OAT):
        raise FileNotFoundError(f"OAT CSV not found: {CSV_OAT}. Run phase 1 first.")

    rows: list[dict] = []
    with open(CSV_OAT, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    base: dict[str, list[float]] = {}
    for r in rows:
        if r["param"] == "baseline" and r["status"] == "ok":
            base.setdefault(r["dataset"], []).append(float(r["val_nmae"]))
    base_mean = {ds: float(np.mean(v)) for ds, v in base.items()}

    importance: dict[str, float] = {}
    for param in OAT_SWEEP:
        param_rows = [r for r in rows if r["param"] == param and r["status"] == "ok"]
        if not param_rows:
            continue
        val_deltas: dict[str, list[float]] = {}
        for r in param_rows:
            v  = r["value"]
            ds = r["dataset"]
            delta = float(r["val_nmae"]) - base_mean.get(ds, float(r["val_nmae"]))
            val_deltas.setdefault(v, []).append(abs(delta))
        importance[param] = float(np.mean([
            np.mean(dlist) for dlist in val_deltas.values()
        ]))

    ranked = sorted(importance, key=lambda p: -importance[p])
    selected = ranked[:n]
    print(f"  Top-{n} params by OAT importance: {selected}")
    return selected


def _build_pair_tasks(p1: str, p2: str) -> list[dict]:
    tasks = []
    v1s = OAT_SWEEP[p1]
    v2s = OAT_SWEEP[p2]
    datasets = list(_make_datasets().keys())

    for val1 in v1s:
        for val2 in v2s:
            for ds in datasets:
                for seed in range(N_SEEDS_PAIR):
                    for fold in range(N_FOLDS_PAIR):
                        p = dict(BASELINE)
                        p[p1] = val1
                        p[p2] = val2
                        tasks.append(dict(
                            phase="pairwise", dataset=ds,
                            seed=seed, fold=fold, n_folds=N_FOLDS_PAIR,
                            param=p1, value=val1,
                            param1=p1, val1=val1,
                            param2=p2, val2=val2,
                            params=p,
                        ))
    return tasks


def _pair_key_fields() -> list[str]:
    return ["phase", "dataset", "param1", "val1", "param2", "val2", "seed", "fold"]


def run_pairwise(pool: mp.Pool) -> None:
    print("\n── Phase 2: Pairwise ─────────────────────────────────────────────")
    top2 = _top_params(n=2)
    p1, p2 = top2[0], top2[1]

    tasks = _build_pair_tasks(p1, p2)
    n_combos = len(OAT_SWEEP[p1]) * len(OAT_SWEEP[p2])
    print(f"  {p1} × {p2}:  {len(OAT_SWEEP[p1])} × {len(OAT_SWEEP[p2])} = {n_combos} combos")
    print(f"  Total tasks: {len(tasks)}  "
          f"(datasets={len(_make_datasets())}  "
          f"seeds={N_SEEDS_PAIR}  folds={N_FOLDS_PAIR})")
    _run_pool(pool, tasks, CSV_PAIR, PAIR_FIELDS, _pair_key_fields())
    _print_pair_summary(p1, p2)


def _print_pair_summary(p1: str, p2: str) -> None:
    if not os.path.exists(CSV_PAIR):
        print("  No pairwise CSV found.")
        return

    rows: list[dict] = []
    with open(CSV_PAIR, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        return

    # Group by (val1, val2) → mean nmae across all datasets/seeds/folds
    grid: dict[tuple, list[float]] = {}
    for r in ok_rows:
        key = (r["val1"], r["val2"])
        grid.setdefault(key, []).append(float(r["val_nmae"]))

    grid_mean = {k: float(np.mean(v)) for k, v in grid.items()}
    best_k    = min(grid_mean, key=grid_mean.get)

    v1s = [str(v) for v in OAT_SWEEP[p1]]
    v2s = [str(v) for v in OAT_SWEEP[p2]]

    print(f"\n  PAIRWISE GRID: mean nmae  ({p1} rows × {p2} cols)")
    header = f"  {'':>22}" + "".join(f"  {v2:>10}" for v2 in v2s)
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for v1 in v1s:
        row_vals = []
        for v2 in v2s:
            m = grid_mean.get((v1, v2), float("nan"))
            tag = "*" if (v1, v2) == best_k else " "
            row_vals.append(f"{m:>9.5f}{tag}")
        print(f"  {p1}={v1:<18}" + "  ".join(row_vals))

    best_nmae = grid_mean[best_k]
    print(f"\n  Best: {p1}={best_k[0]}, {p2}={best_k[1]}  →  mean nmae={best_nmae:.5f}")

    # Also show marginal improvement from varying each param alone
    print(f"\n  Marginal effect of {p1} (averaged over {p2}):")
    for v1 in v1s:
        vals = [grid_mean.get((v1, v2), float("nan")) for v2 in v2s]
        finite = [v for v in vals if not np.isnan(v)]
        m = float(np.mean(finite)) if finite else float("nan")
        print(f"    {p1}={v1:<15}  mean nmae={m:.5f}")

    print(f"\n  Marginal effect of {p2} (averaged over {p1}):")
    for v2 in v2s:
        vals = [grid_mean.get((v1, v2), float("nan")) for v1 in v1s]
        finite = [v for v in vals if not np.isnan(v)]
        m = float(np.mean(finite)) if finite else float("nan")
        print(f"    {p2}={v2:<15}  mean nmae={m:.5f}")


# ── Summary-only mode ─────────────────────────────────────────────────────────

def run_summary() -> None:
    print("── OAT Summary ────────────────────────────────────────────────────")
    _print_oat_summary()
    print("\n── Pairwise Summary ───────────────────────────────────────────────")
    if os.path.exists(CSV_PAIR):
        rows: list[dict] = []
        with open(CSV_PAIR, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ok = [r for r in rows if r["status"] == "ok"]
        if ok:
            p1 = ok[0].get("param1", "")
            p2 = ok[0].get("param2", "")
            _print_pair_summary(p1, p2)
        else:
            print("  No completed pairwise rows.")
    else:
        print("  No pairwise CSV found.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",   type=int, default=0,
                        help="1=OAT, 2=pairwise, 0=both")
    parser.add_argument("--summary", action="store_true",
                        help="Print summaries from saved CSVs and exit")
    parser.add_argument("--jobs",    type=int, default=0,
                        help="Parallel workers (0 = cpu_count)")
    args = parser.parse_args()

    if args.summary:
        run_summary()
        return

    n_jobs = mp.cpu_count() if args.jobs <= 0 else args.jobs
    print(f"PyramidHART meta-regression  |  workers={n_jobs}")
    print(f"Baseline: {BASELINE}")

    # Set BLAS threads to 1 to avoid nested parallelism
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_var, "1")

    ctx = mp.get_context("spawn")
    with ctx.Pool(n_jobs, initializer=_worker_init) as pool:
        if args.phase in (0, 1):
            run_oat(pool)
        if args.phase in (0, 2):
            run_pairwise(pool)


if __name__ == "__main__":
    main()
