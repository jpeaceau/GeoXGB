"""
GeoXGB Regression Meta-Analysis v2
====================================
Noise-aware OAT + pairwise + residual geometry analysis.
See strategy.md for full design rationale.

Usage
-----
  python meta_reg.py                   # all phases
  python meta_reg.py --phase 1         # noise sweep only
  python meta_reg.py --phase 2         # OAT primary
  python meta_reg.py --phase 3         # auto-edge extension (after phase 2)
  python meta_reg.py --phase 4         # OAT secondary
  python meta_reg.py --phase 5         # pairwise (reads phases 2-4)
  python meta_reg.py --phase 6         # residual geometry (reads phase 5)
  python meta_reg.py --summary         # print all summaries from saved CSVs
  python meta_reg.py --jobs N          # parallel workers (default: cpu_count)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
from sklearn.datasets import make_friedman1, make_friedman2, make_regression
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV = {
    1: os.path.join(RESULTS_DIR, "phase1_noise.csv"),
    2: os.path.join(RESULTS_DIR, "phase2_oat_primary.csv"),
    3: os.path.join(RESULTS_DIR, "phase3_oat_edge.csv"),
    4: os.path.join(RESULTS_DIR, "phase4_oat_secondary.csv"),
    5: os.path.join(RESULTS_DIR, "phase5_pairwise.csv"),
    6: os.path.join(RESULTS_DIR, "phase6_residual.csv"),
}

# ── Baseline (C++ GeoXGBConfig defaults, n_rounds=3000 for fairness) ─────────

BASELINE: dict = dict(
    n_rounds            = 3000,
    learning_rate       = 0.2,
    max_depth           = 3,
    min_samples_leaf    = 5,
    reduce_ratio        = 0.7,
    expand_ratio        = 0.0,
    y_weight            = 0.5,
    refit_interval      = 20,
    auto_noise          = True,
    noise_guard         = True,
    refit_noise_floor   = 0.05,
    auto_expand         = True,
    min_train_samples   = 5000,
    bandwidth           = "auto",
    variance_weighted   = True,
    hvrt_min_samples_leaf = -1,
    n_partitions        = -1,
    n_bins              = 64,
    random_state        = 42,
)

# ── Sweep definitions ─────────────────────────────────────────────────────────

PRIMARY_SWEEP: dict[str, list] = dict(
    learning_rate  = [0.02, 0.05, 0.1, 0.2, 0.3],
    max_depth      = [2, 3, 4, 5],
    reduce_ratio   = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    y_weight       = [0.2, 0.3, 0.5, 0.7, 0.8],
    refit_interval = [0, 5, 10, 20, 50],
    auto_noise     = [True, False],
)

SECONDARY_SWEEP: dict[str, list] = dict(
    min_samples_leaf   = [3, 5, 10, 20],
    variance_weighted  = [True, False],
    n_bins             = [32, 64, 128],
    n_partitions       = [-1, 30, 60, 100],
    expand_ratio       = [0.0, 0.1, 0.2],
    noise_guard        = [True, False],
)

NOISE_LEVELS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

N_SEEDS       = 10   # seeds per OAT config
N_FOLDS       = 4    # folds for OAT
N_SEEDS_PAIR  = 5    # seeds for pairwise
N_FOLDS_PAIR  = 3    # folds for pairwise
N_SEEDS_NOISE = 10   # seeds for noise sweep
N_FOLDS_NOISE = 5
PAIRWISE_TOP_N = 3   # top-N params for pairwise coupling

# ── CSV field definitions ──────────────────────────────────────────────────────

OAT_FIELDS = [
    "phase", "dataset", "n_samples", "n_features", "noise_sigma", "r2_ceiling",
    "param", "value", "seed", "fold",
    "val_r2", "val_r2_adj", "val_rmse", "val_mae",
    "val_nmae", "val_skill_adj", "val_mad_skill",
    "train_time_s", "convergence_round", "last_noise_mod",
    "config_json", "status",
]

PAIR_FIELDS = [
    "dataset", "n_samples", "n_features", "noise_sigma", "r2_ceiling",
    "param1", "val1", "param2", "val2", "seed", "fold",
    "val_r2", "val_r2_adj", "val_rmse", "val_mae",
    "val_nmae", "val_skill_adj", "val_mad_skill",
    "train_time_s", "config_json", "status",
]

NOISE_FIELDS = [
    "model", "noise_sigma", "r2_ceiling",
    "dataset", "n_samples", "n_features", "seed", "fold",
    "val_r2", "val_r2_adj", "val_rmse", "train_time_s", "status",
]

RESID_FIELDS = [
    "dataset", "seed", "partition_id", "n_real",
    "mean_resid", "std_resid", "skew_resid", "abs_mean_resid",
    "mean_abs_z", "T_value", "frac_in_cone", "cone_degenerate",
    "status",
]

# ── Dataset generation ────────────────────────────────────────────────────────

def _r2_ceiling(noise_sigma: float, y: np.ndarray) -> float:
    """R²_ceiling = 1 - σ² / Var(y).  For σ=0, returns 1.0."""
    if noise_sigma <= 0.0:
        return 1.0
    var_y = float(np.var(y))
    if var_y < 1e-12:
        return 1.0
    ceil = 1.0 - noise_sigma ** 2 / var_y
    return max(ceil, 1e-6)  # clamp away from zero/negative


def _make_datasets() -> dict:
    """
    Returns dict of name -> (X, y, noise_sigma, r2_ceiling).
    All y are z-score normalised for comparable RMSE.
    r2_ceiling is computed on raw y BEFORE normalisation: ceil = 1 - σ²/Var(y_raw).
    Normalisation scales signal and noise equally, so the ceiling is the same
    but must be computed using raw units — after normalisation Var(y_norm)=1 and
    σ is still in raw units, making 1-σ²/1 incorrect.
    """
    rs = 0  # fixed seed for dataset generation

    X1, y1 = make_friedman1(n_samples=1_000, n_features=10, noise=1.0, random_state=rs)
    X2, y2 = make_friedman2(n_samples=1_000, noise=0.0, random_state=rs)
    X3, y3 = make_regression(n_samples=2_000, n_features=30, n_informative=8,
                              noise=0.5, random_state=rs)
    X4, y4 = make_regression(n_samples=5_000, n_features=20, n_informative=15,
                              noise=1.0, random_state=rs)

    raw = {
        "friedman1":  (X1, y1, 1.0),
        "friedman2":  (X2, y2, 0.0),
        "reg_sparse": (X3, y3, 0.5),
        "reg_large":  (X4, y4, 1.0),
    }

    datasets = {}
    for name, (X, y, sigma) in raw.items():
        ceil   = _r2_ceiling(sigma, y)          # compute BEFORE normalisation
        y_norm = (y - y.mean()) / (y.std() + 1e-10)
        datasets[name] = (X, y_norm, sigma, ceil)
    return datasets


def _make_noise_datasets(sigma: float) -> tuple:
    """friedman1 with a specific noise level (for Phase 1 sweep)."""
    rs = 0
    X, y = make_friedman1(n_samples=1_000, n_features=10, noise=sigma, random_state=rs)
    ceil   = _r2_ceiling(sigma, y)              # compute BEFORE normalisation
    y_norm = (y - y.mean()) / (y.std() + 1e-10)
    return X, y_norm, ceil

# ── Worker infrastructure ─────────────────────────────────────────────────────

_DATASETS: dict = {}   # populated in each worker process by _worker_init


def _worker_init() -> None:
    """Called once per worker process; caches datasets to avoid per-task pickling."""
    global _DATASETS
    warnings.filterwarnings("ignore")
    _DATASETS = _make_datasets()


def _worker(task: dict) -> dict:
    """
    Module-level worker for multiprocessing.Pool.imap_unordered.
    task keys: phase, dataset, seed, fold, n_folds, params,
               param, value, param2, val2 (optional), noise_sigma, r2_ceiling
    """
    import warnings as _w
    _w.filterwarnings("ignore")

    from sklearn.metrics import r2_score as _r2
    from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor

    ds_name  = task["dataset"]
    X, y, _, _ = _DATASETS[ds_name]
    n_samples, n_features = X.shape

    seed    = task["seed"]
    fold    = task["fold"]
    n_folds = task["n_folds"]
    noise_sigma = task.get("noise_sigma", 0.0)
    r2_ceil     = task.get("r2_ceiling", 1.0)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fi, (tr, val) in enumerate(kf.split(X)):
        if fi == fold:
            train_idx, val_idx = tr, val
            break

    t0 = time.perf_counter()
    try:
        cfg   = make_cpp_config(**task["params"])
        model = CppGeoXGBRegressor(cfg)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[val_idx])
        y_val  = y[val_idx]

        val_r2   = float(_r2(y_val, y_pred))
        val_r2_adj = val_r2 / r2_ceil if r2_ceil > 1e-6 else float("nan")
        val_rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
        val_mae  = float(np.mean(np.abs(y_pred - y_val)))
        conv_r   = int(model.convergence_round())
        noise_m  = float(model.last_noise_modulation())

        # Noise-corrected NMAE skill: primary deployment metric.
        # Datasets are z-score normalised so std(y_train) ≈ 1.0.
        # nmae_base: naive MAE (predict training mean) / std(y_train).
        # nmae_floor: irreducible MAE in normalised units.
        #   noise_sigma is in RAW units; after normalisation
        #   sigma_norm = sqrt(1 - r2_ceiling)  [exact for Gaussian noise].
        #   MAE_floor_norm = sigma_norm * sqrt(2/pi).
        _std_tr    = float(np.std(y[train_idx])) + 1e-9
        _mae_base  = float(np.mean(np.abs(y_val - y[train_idx].mean())))
        val_nmae   = val_mae / _std_tr
        _nmae_base = _mae_base / _std_tr
        _r2_ceil   = task.get("r2_ceiling", 1.0)
        _sigma_norm = float(np.sqrt(max(1.0 - _r2_ceil, 0.0)))
        _nmae_floor = _sigma_norm * np.sqrt(2.0 / np.pi) / _std_tr
        _denom = _nmae_base - _nmae_floor
        val_skill_adj = (float((_nmae_base - val_nmae) / (_denom + 1e-9))
                         if _denom > 1e-6 else float("nan"))

        # MAD-skill (noise-floor corrected): compare against predict-median baseline.
        # Structurally identical to val_skill_adj but uses the median predictor
        # instead of the mean predictor as the naive baseline.
        #
        #   val_mad_skill = (MAE_median_pred - MAE_model) / (MAE_median_pred - MAE_floor)
        #
        # MAE_floor (irreducible) is the same absolute floor used in val_skill_adj:
        #   MAE_floor = sigma_noise * sqrt(2/pi)  where sigma_noise = sigma_norm * std(y)
        #
        # Why the median predictor?
        #   median(y_train) is the Bayes-optimal constant predictor under L1 loss.
        #   The mean predictor minimises MSE; using it as baseline for an MAE metric
        #   is inconsistent.  On skewed / heavy-tailed targets the median predictor
        #   is substantially harder to beat, giving a more honest skill score.
        #
        # Range: (-inf, 1].  1 = perfect; 0 = same as predict-median; <0 = worse.
        _med_tr       = float(np.median(y[train_idx]))
        _mad_base     = float(np.mean(np.abs(y_val - _med_tr)))
        _mae_floor    = _sigma_norm * float(np.sqrt(2.0 / np.pi)) * _std_tr
        _mad_denom    = _mad_base - _mae_floor
        val_mad_skill = (float((_mad_base - val_mae) / (_mad_denom + 1e-9))
                         if _mad_denom > 1e-6 else float("nan"))
        status   = "ok"
    except Exception as exc:
        val_r2 = val_r2_adj = val_rmse = val_mae = float("nan")
        val_nmae = val_skill_adj = val_mad_skill = float("nan")
        conv_r = -1
        noise_m = float("nan")
        status = repr(exc)[:160]

    train_t = time.perf_counter() - t0

    row = dict(
        phase              = task.get("phase", ""),
        dataset            = ds_name,
        n_samples          = n_samples,
        n_features         = n_features,
        noise_sigma        = noise_sigma,
        r2_ceiling         = r2_ceil,
        param              = task.get("param", ""),
        value              = str(task.get("value", "")),
        param2             = task.get("param2", ""),
        val2               = str(task.get("val2", "")),
        seed               = seed,
        fold               = fold,
        val_r2             = val_r2,
        val_r2_adj         = val_r2_adj,
        val_rmse           = val_rmse,
        val_mae            = val_mae,
        val_nmae           = val_nmae,
        val_skill_adj      = val_skill_adj,
        val_mad_skill      = val_mad_skill,
        train_time_s       = train_t,
        convergence_round  = conv_r,
        last_noise_mod     = noise_m,
        config_json        = json.dumps({k: str(v) for k, v in task["params"].items()}),
        status             = status,
    )
    return row


def _worker_noise(task: dict) -> dict:
    """Worker for Phase 1 noise sweep (GeoXGB vs XGBoost)."""
    import warnings as _w
    _w.filterwarnings("ignore")

    import numpy as np
    from sklearn.metrics import r2_score as _r2

    model_name  = task["model"]
    sigma       = task["sigma"]
    seed        = task["seed"]
    fold        = task["fold"]
    n_folds     = task["n_folds"]

    # Regenerate dataset with this specific sigma
    X, y, r2_ceil = _make_noise_datasets(sigma)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fi, (tr, val) in enumerate(kf.split(X)):
        if fi == fold:
            train_idx, val_idx = tr, val
            break

    t0 = time.perf_counter()
    try:
        if model_name == "geoxgb":
            from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor
            params = dict(BASELINE)
            params["random_state"] = seed
            cfg   = make_cpp_config(**params)
            model = CppGeoXGBRegressor(cfg)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
        elif model_name == "xgboost":
            import xgboost as xgb
            model = xgb.XGBRegressor(
                n_estimators=3000, learning_rate=0.2, max_depth=3,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                random_state=seed, n_jobs=1, verbosity=0,
                early_stopping_rounds=None,
            )
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        y_val    = y[val_idx]
        val_r2   = float(_r2(y_val, y_pred))
        val_r2_adj = val_r2 / r2_ceil if r2_ceil > 1e-6 else float("nan")
        val_rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
        status   = "ok"
    except Exception as exc:
        val_r2 = val_r2_adj = val_rmse = float("nan")
        status  = repr(exc)[:160]

    return dict(
        model       = model_name,
        noise_sigma = sigma,
        r2_ceiling  = r2_ceil,
        dataset     = "friedman1",
        n_samples   = 1000,
        n_features  = 10,
        seed        = seed,
        fold        = fold,
        val_r2      = val_r2,
        val_r2_adj  = val_r2_adj,
        val_rmse    = val_rmse,
        train_time_s = time.perf_counter() - t0,
        status      = status,
    )

# ── CSV utilities ─────────────────────────────────────────────────────────────

def _load_done(csv_path: str, key_fields: list[str]) -> set[tuple]:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {
            tuple(r.get(k, "") for k in key_fields)
            for r in csv.DictReader(f)
        }


def _append_rows(csv_path: str, rows: list[dict], fields: list[str]) -> None:
    write_header = not os.path.exists(csv_path)
    mode = "a" if not write_header else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _run_pool(pool: mp.Pool, tasks: list[dict], worker_fn,
              label: str, fields: list[str], csv_path: str,
              key_fields: list[str]) -> list[dict]:
    """Submit tasks to pool, collect results with progress, save incrementally."""
    done_keys = _load_done(csv_path, key_fields)
    pending = [t for t in tasks
               if tuple(str(t.get(k, "")) for k in key_fields) not in done_keys]

    if not pending:
        print(f"  {label}: all {len(tasks)} configs cached, skipping.")
        return []

    print(f"  {label}: {len(pending)}/{len(tasks)} configs to run "
          f"({len(tasks) - len(pending)} cached)...")

    results = []
    t0 = time.perf_counter()
    batch = []
    BATCH_SIZE = max(1, len(pending) // 50)  # flush every ~2%

    for i, row in enumerate(pool.imap_unordered(worker_fn, pending), 1):
        results.append(row)
        batch.append(row)
        if len(batch) >= BATCH_SIZE or i == len(pending):
            _append_rows(csv_path, batch, fields)
            batch.clear()
        if i % max(1, len(pending) // 20) == 0 or i == len(pending):
            elapsed = time.perf_counter() - t0
            eta = elapsed / i * (len(pending) - i) if i < len(pending) else 0
            print(f"    {i:>5}/{len(pending)}  {elapsed:>6.0f}s  ETA {eta:>5.0f}s")

    ok  = sum(1 for r in results if r.get("status") == "ok")
    err = len(results) - ok
    print(f"  {label}: done. {ok} ok, {err} errors.")
    return results

# ── Phase 1: Noise robustness sweep ──────────────────────────────────────────

def run_phase1(pool: mp.Pool) -> None:
    print("\n" + "=" * 70)
    print("  Phase 1 — Noise robustness sweep (GeoXGB vs XGBoost)")
    print("=" * 70)

    tasks = []
    for sigma in NOISE_LEVELS:
        for model_name in ["geoxgb", "xgboost"]:
            for seed in range(N_SEEDS_NOISE):
                for fold in range(N_FOLDS_NOISE):
                    tasks.append(dict(
                        model=model_name, sigma=sigma, seed=seed,
                        fold=fold, n_folds=N_FOLDS_NOISE,
                    ))

    _run_pool(pool, tasks, _worker_noise, "Phase 1 noise sweep",
              NOISE_FIELDS, CSV[1],
              key_fields=["model", "noise_sigma", "seed", "fold"])

# ── Phase 2 / 4: OAT runner (shared) ─────────────────────────────────────────

def _build_oat_tasks(sweep: dict, phase_name: str,
                     datasets: dict) -> list[dict]:
    """Build task list for an OAT sweep over given params."""
    tasks = []
    for param, values in sweep.items():
        for value in values:
            params = dict(BASELINE)
            params[param] = value
            for ds_name, (_, _, sigma, ceil) in datasets.items():
                for seed in range(N_SEEDS):
                    for fold in range(N_FOLDS):
                        tasks.append(dict(
                            phase       = phase_name,
                            dataset     = ds_name,
                            noise_sigma = sigma,
                            r2_ceiling  = ceil,
                            param       = param,
                            value       = value,
                            param2      = "",
                            val2        = "",
                            seed        = seed,
                            fold        = fold,
                            n_folds     = N_FOLDS,
                            params      = {k: v for k, v in params.items()
                                           if k != "random_state"},
                        ))
    # Inject per-task seed into params
    for t in tasks:
        t["params"]["random_state"] = t["seed"]
    return tasks


def run_phase2(pool: mp.Pool, datasets: dict) -> None:
    print("\n" + "=" * 70)
    print("  Phase 2 — OAT Primary (6 parameters)")
    print("=" * 70)
    tasks = _build_oat_tasks(PRIMARY_SWEEP, "oat_primary", datasets)
    _run_pool(pool, tasks, _worker, "Phase 2 OAT primary",
              OAT_FIELDS, CSV[2],
              key_fields=["phase", "dataset", "param", "value", "seed", "fold"])


def run_phase4(pool: mp.Pool, datasets: dict) -> None:
    print("\n" + "=" * 70)
    print("  Phase 4 — OAT Secondary (6 parameters)")
    print("=" * 70)
    tasks = _build_oat_tasks(SECONDARY_SWEEP, "oat_secondary", datasets)
    _run_pool(pool, tasks, _worker, "Phase 4 OAT secondary",
              OAT_FIELDS, CSV[4],
              key_fields=["phase", "dataset", "param", "value", "seed", "fold"])

# ── Phase 3: Auto-edge extension ──────────────────────────────────────────────

def _best_value_per_param(csv_path: str,
                          metric: str = "val_skill_adj") -> dict[str, object]:
    """
    Read an OAT CSV, compute mean <metric> per (param, value) across all
    datasets and seeds, return {param: best_value}.

    metric choices:
      "val_skill_adj"  -- NMAE-skill (noise-floor corrected, mean-predictor baseline)
      "val_mad_skill"  -- MAD-skill  (median-predictor baseline, tail-robust)
    """
    if not os.path.exists(csv_path):
        return {}

    cell: dict[tuple, list] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            try:
                v = float(r[metric])
                if not np.isnan(v):
                    # Try to parse value back to original type
                    raw = r["value"]
                    try:
                        val = int(raw)
                    except ValueError:
                        try:
                            val = float(raw)
                        except ValueError:
                            val = raw  # string (True/False/etc)
                    cell[(r["param"], raw)].append(v)
            except (ValueError, KeyError):
                pass

    # Aggregate mean per (param, value_str)
    param_means: dict[str, dict] = defaultdict(dict)
    for (param, val_str), scores in cell.items():
        param_means[param][val_str] = float(np.mean(scores))

    best: dict[str, object] = {}
    for param, val_dict in param_means.items():
        best_str = max(val_dict, key=val_dict.__getitem__)
        # Reconstruct type
        try:
            best[param] = int(best_str)
        except ValueError:
            try:
                best[param] = float(best_str)
            except ValueError:
                best[param] = best_str
    return best


def _edge_extensions(param: str, best_val, sweep_vals: list) -> list:
    """
    If best_val is at the boundary of sweep_vals, return extra values to test.
    Only applies to numeric (int/float) parameters.
    """
    if not isinstance(best_val, (int, float)) or isinstance(best_val, bool):
        return []

    numeric = sorted(v for v in sweep_vals
                     if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not numeric or best_val not in numeric:
        return []

    extras = []
    is_int = isinstance(best_val, int) and all(isinstance(v, int) for v in numeric)

    if best_val == numeric[0]:  # at lower boundary → extend downward
        if is_int:
            step = numeric[1] - numeric[0] if len(numeric) > 1 else 1
            for k in [1, 2]:
                new = best_val - k * step
                if new >= 0 and new not in numeric:
                    extras.append(new)
        else:
            for divisor in [2.0, 4.0]:
                new = round(best_val / divisor, 6)
                if new > 0 and new not in numeric:
                    extras.append(new)

    elif best_val == numeric[-1]:  # at upper boundary → extend upward
        if is_int:
            step = numeric[-1] - numeric[-2] if len(numeric) > 1 else 1
            for k in [1, 2]:
                new = best_val + k * step
                if new not in numeric:
                    extras.append(new)
        else:
            for mult in [1.5, 2.0]:
                new = round(best_val * mult, 6)
                if new not in numeric:
                    extras.append(new)

    # Clamp params with natural bounds
    if param == "reduce_ratio":
        extras = [v for v in extras if 0.0 < v <= 1.0]
    if param == "y_weight":
        extras = [v for v in extras if 0.0 <= v <= 1.0]
    if param == "refit_interval":
        extras = [v for v in extras if v >= 0]
    if param == "max_depth":
        extras = [v for v in extras if v >= 1]
    if param == "learning_rate":
        extras = [v for v in extras if v > 0]

    return extras


def run_phase3(pool: mp.Pool, datasets: dict) -> None:
    print("\n" + "=" * 70)
    print("  Phase 3 — Auto-edge extension")
    print("=" * 70)

    best_vals = _best_value_per_param(CSV[2])
    if not best_vals:
        print("  No Phase 2 results found — skipping edge extension.")
        return

    ext_sweep: dict[str, list] = {}
    for param, sweep_vals in PRIMARY_SWEEP.items():
        best = best_vals.get(param)
        if best is None:
            continue
        extras = _edge_extensions(param, best, sweep_vals)
        if extras:
            print(f"  {param}: best={best}  extending with {extras}")
            ext_sweep[param] = extras

    if not ext_sweep:
        print("  No parameters hit boundary — no edge extensions needed.")
        return

    tasks = _build_oat_tasks(ext_sweep, "oat_edge", datasets)
    _run_pool(pool, tasks, _worker, "Phase 3 edge extension",
              OAT_FIELDS, CSV[3],
              key_fields=["phase", "dataset", "param", "value", "seed", "fold"])

# ── Phase 5: Pairwise ─────────────────────────────────────────────────────────

def _top_params_by_importance(n: int = PAIRWISE_TOP_N,
                              metric: str = "val_skill_adj") -> list[str]:
    """
    Combine Phase 2, 3, 4 OAT results.  Return top-n params by mean
    |delta-metric| from baseline across all datasets and seeds.
    Only primary params are candidates (pairwise coupling is for primary params).
    """
    # Collect mean metric per (param, value_str) from phases 2 + 3
    cell: dict[tuple, list] = defaultdict(list)
    for csv_p in [CSV[2], CSV[3]]:
        if not os.path.exists(csv_p):
            continue
        with open(csv_p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                try:
                    v = float(r[metric])
                    if not np.isnan(v):
                        cell[(r["param"], r["value"])].append(v)
                except (ValueError, KeyError):
                    pass

    # Baseline mean per param (value == baseline value)
    base_mean: dict[str, float] = {}
    for param, bval in BASELINE.items():
        key = (param, str(bval))
        scores = cell.get(key, [])
        if scores:
            base_mean[param] = float(np.mean(scores))

    # Importance = mean |delta| across all values for this param
    importance: dict[str, float] = {}
    for (param, val_str), scores in cell.items():
        if param not in PRIMARY_SWEEP:
            continue
        base = base_mean.get(param)
        if base is None:
            continue
        delta = abs(float(np.mean(scores)) - base)
        if param not in importance:
            importance[param] = 0.0
        importance[param] = max(importance[param], delta)

    ranked = sorted(importance.items(), key=lambda x: -x[1])
    top = [p for p, _ in ranked[:n]]
    print(f"  Top-{n} params for pairwise: {top}")
    for p, imp in ranked[:n]:
        print(f"    {p:<20}  importance={imp:.5f}")
    return top


def _collect_sweep_vals(param: str) -> list:
    """Collect all tested values for a param (primary sweep + edge extensions)."""
    base = list(PRIMARY_SWEEP.get(param, []))
    if not os.path.exists(CSV[3]):
        return base
    extra = set()
    with open(CSV[3], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("param") == param and r.get("status") == "ok":
                raw = r["value"]
                try:
                    extra.add(int(raw))
                except ValueError:
                    try:
                        extra.add(float(raw))
                    except ValueError:
                        extra.add(raw)
    all_vals = list(base)
    for v in extra:
        if v not in all_vals:
            all_vals.append(v)
    return all_vals


def run_phase5(pool: mp.Pool, datasets: dict) -> None:
    print("\n" + "=" * 70)
    print("  Phase 5 — Pairwise (top OAT parameters)")
    print("=" * 70)

    top_params = _top_params_by_importance(PAIRWISE_TOP_N)
    if len(top_params) < 2:
        print("  Not enough params for pairwise — skipping.")
        return

    from itertools import combinations
    pairs = list(combinations(top_params, 2))
    print(f"  Running {len(pairs)} pairs...")

    tasks = []
    for p1, p2 in pairs:
        vals1 = _collect_sweep_vals(p1)
        vals2 = _collect_sweep_vals(p2)
        for v1 in vals1:
            for v2 in vals2:
                params = dict(BASELINE)
                params[p1] = v1
                params[p2] = v2
                for ds_name, (_, _, sigma, ceil) in datasets.items():
                    for seed in range(N_SEEDS_PAIR):
                        for fold in range(N_FOLDS_PAIR):
                            t = dict(
                                phase       = "pairwise",
                                dataset     = ds_name,
                                noise_sigma = sigma,
                                r2_ceiling  = ceil,
                                param       = p1,
                                value       = v1,
                                param2      = p2,
                                val2        = v2,
                                seed        = seed,
                                fold        = fold,
                                n_folds     = N_FOLDS_PAIR,
                                params      = {k: vv for k, vv in params.items()
                                               if k != "random_state"},
                            )
                            t["params"]["random_state"] = seed
                            tasks.append(t)

    _run_pool(pool, tasks, _worker, "Phase 5 pairwise",
              OAT_FIELDS, CSV[5],
              key_fields=["phase", "dataset", "param", "value",
                          "param2", "val2", "seed", "fold"])

# ── Phase 6: Residual geometry analysis ───────────────────────────────────────

def _parse_config_json(json_str: str) -> dict:
    """
    Deserialise a config_json string (all values stored as strings) back to
    the correct Python types expected by make_cpp_config().
    Bool-valued params are stored as "True"/"False" and must be converted to
    bool, not left as strings — the C++ binding rejects string args for bool params.
    """
    raw = json.loads(json_str)
    result = dict(BASELINE)
    for k, v in raw.items():
        if v in ("True", "False"):   # bool must come first (before int attempt)
            result[k] = (v == "True")
        else:
            try:
                result[k] = int(v)
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
    return result


def _best_config_from_pairwise(metric: str = "val_skill_adj") -> dict:
    """
    Read Phase 5 pairwise CSV. Return the param combo with highest mean
    <metric> across all datasets, averaged over seeds and folds.
    Falls back to OAT Phase 2 best-per-param if Phase 5 is unavailable.
    """
    source = CSV[5] if os.path.exists(CSV[5]) else CSV[2]
    if not os.path.exists(source):
        return dict(BASELINE)

    cell: dict[str, list] = defaultdict(list)
    with open(source, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                try:
                    v = float(r[metric])
                    if not np.isnan(v):
                        cell[r.get("config_json", "")].append(v)
                except (ValueError, KeyError):
                    pass

    if not cell:
        return dict(BASELINE)

    best_json = max(cell, key=lambda k: np.mean(cell[k]))
    try:
        return _parse_config_json(best_json)
    except Exception:
        return dict(BASELINE)


def run_phase6(datasets: dict) -> None:
    print("\n" + "=" * 70)
    print("  Phase 6 — Residual geometry analysis")
    print("=" * 70)

    best_params = _best_config_from_pairwise()
    print(f"  Best config: lr={best_params.get('learning_rate')}  "
          f"depth={best_params.get('max_depth')}  "
          f"reduce={best_params.get('reduce_ratio')}  "
          f"y_w={best_params.get('y_weight')}")

    try:
        from hvrt import HVRT
        hvrt_available = True
    except ImportError:
        print("  WARNING: hvrt package not importable — geometry stats skipped.")
        hvrt_available = False

    from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor
    from scipy.stats import skew as _skew

    rows = []
    N_SEEDS_RESID = 5

    for ds_name, (X, y, sigma, r2_ceil) in datasets.items():
        n = len(y)
        for seed in range(N_SEEDS_RESID):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n)
            tr, val = idx[:int(0.8 * n)], idx[int(0.8 * n):]
            X_tr, y_tr = X[tr], y[tr]

            try:
                params = dict(best_params)
                params["random_state"] = seed
                cfg   = make_cpp_config(**params)
                model = CppGeoXGBRegressor(cfg)
                model.fit(X_tr, y_tr)
                resid = y_tr - model.predict(X_tr)

                if hvrt_available:
                    hvrt = HVRT(
                        y_weight=float(best_params.get("y_weight", 0.5)),
                        random_state=seed,
                    )
                    hvrt.fit(X_tr, y_tr)
                    pids  = np.asarray(hvrt.partition_ids_)
                    geom  = hvrt.geometry_stats()

                    # Build per-partition geometry lookup
                    part_geom: dict[int, dict] = {}
                    for pg in geom.get("partitions", []):
                        pid = int(pg.get("id", -1))
                        part_geom[pid] = pg

                    for pid in np.unique(pids):
                        mask   = pids == pid
                        r_p    = resid[mask]
                        n_real = int(mask.sum())
                        pg     = part_geom.get(int(pid), {})

                        rows.append(dict(
                            dataset         = ds_name,
                            seed            = seed,
                            partition_id    = int(pid),
                            n_real          = n_real,
                            mean_resid      = float(np.mean(r_p)),
                            std_resid       = float(np.std(r_p)),
                            skew_resid      = float(_skew(r_p)) if n_real > 2 else float("nan"),
                            abs_mean_resid  = float(abs(np.mean(r_p))),
                            mean_abs_z      = float(pg.get("mean_abs_z", float("nan"))),
                            T_value         = float(pg.get("E_T", float("nan"))),
                            frac_in_cone    = float(pg.get("frac_in_cone", float("nan"))),
                            cone_degenerate = str(pg.get("cone_degenerate", "unknown")),
                            status          = "ok",
                        ))
                else:
                    # No HVRT available: record aggregate residual stats only
                    rows.append(dict(
                        dataset         = ds_name,
                        seed            = seed,
                        partition_id    = -1,
                        n_real          = len(resid),
                        mean_resid      = float(np.mean(resid)),
                        std_resid       = float(np.std(resid)),
                        skew_resid      = float(_skew(resid)),
                        abs_mean_resid  = float(abs(np.mean(resid))),
                        mean_abs_z      = float("nan"),
                        T_value         = float("nan"),
                        frac_in_cone    = float("nan"),
                        cone_degenerate = "unavailable",
                        status          = "ok",
                    ))

            except Exception as exc:
                rows.append(dict(
                    dataset=ds_name, seed=seed, partition_id=-1, n_real=0,
                    mean_resid=float("nan"), std_resid=float("nan"),
                    skew_resid=float("nan"), abs_mean_resid=float("nan"),
                    mean_abs_z=float("nan"), T_value=float("nan"),
                    frac_in_cone=float("nan"), cone_degenerate="error",
                    status=repr(exc)[:160],
                ))

            print(f"    {ds_name}  seed={seed}  ok")

    _append_rows(CSV[6], rows, RESID_FIELDS)
    print(f"  Phase 6 complete → {CSV[6]}")

# ── Summary ───────────────────────────────────────────────────────────────────

def print_noise_summary() -> None:
    if not os.path.exists(CSV[1]):
        print("  Phase 1 not run yet.")
        return
    print("\n" + "=" * 70)
    print("  Phase 1 — Noise robustness summary")
    print("=" * 70)

    cell: dict[tuple, list] = defaultdict(list)
    with open(CSV[1], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            try:
                v = float(r["val_r2_adj"])  # Phase 1 CSV only has val_r2_adj
                if not np.isnan(v):
                    cell[(r["model"], r["noise_sigma"])].append(v)
            except (ValueError, KeyError):
                pass

    sigmas = sorted(set(float(s) for _, s in cell.keys()))
    print(f"\n  {'sigma':>6}  {'ceiling':>8}  {'GeoXGB R²_adj':>16}  {'XGBoost R²_adj':>16}")
    print("  " + "-" * 52)
    for sigma in sigmas:
        geo  = cell.get(("geoxgb",   str(sigma)), [])
        xgb_ = cell.get(("xgboost",  str(sigma)), [])
        # also try float-formatted key
        sigma_str = f"{sigma}"
        geo  = geo  or cell.get(("geoxgb",  sigma_str), [])
        xgb_ = xgb_ or cell.get(("xgboost", sigma_str), [])
        gmean = f"{np.mean(geo):.4f}±{np.std(geo):.4f}" if geo else "n/a"
        xmean = f"{np.mean(xgb_):.4f}±{np.std(xgb_):.4f}" if xgb_ else "n/a"
        print(f"  {sigma:>6.1f}                   {gmean:>16}  {xmean:>16}")


def _print_oat_table(csv_path: str, title: str, sweep: dict) -> None:
    if not os.path.exists(csv_path):
        print(f"  {title}: not run yet.")
        return
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    cell: dict[tuple, list] = defaultdict(list)
    ds_set: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            try:
                v = float(r["val_skill_adj"])
                if not np.isnan(v):
                    cell[(r["param"], r["value"], r["dataset"])].append(v)
                    ds_set.add(r["dataset"])
            except (ValueError, KeyError):
                pass

    datasets = sorted(ds_set)
    base_mean: dict[tuple, float] = {}
    for param in sweep:
        bval = str(BASELINE.get(param, ""))
        for ds in datasets:
            scores = cell.get((param, bval, ds), [])
            if scores:
                base_mean[(param, ds)] = float(np.mean(scores))

    importance: dict[str, float] = {}
    for param, values in sweep.items():
        deltas = []
        for val in values:
            for ds in datasets:
                scores = cell.get((param, str(val), ds), [])
                base   = base_mean.get((param, ds))
                if scores and base is not None:
                    deltas.append(abs(float(np.mean(scores)) - base))
        importance[param] = float(np.mean(deltas)) if deltas else 0.0

    ranked = sorted(importance.items(), key=lambda x: -x[1])

    hdr = f"  {'rank':>4}  {'parameter':<22}  {'importance':>10}"
    for ds in datasets:
        hdr += f"  {ds[:10]:>12}"
    print(hdr)
    print("  " + "-" * len(hdr))

    for rank, (param, imp) in enumerate(ranked, 1):
        bval = str(BASELINE.get(param, ""))
        row  = f"  {rank:>4d}  {param:<22}  {imp:>10.5f}"
        for ds in datasets:
            base   = base_mean.get((param, ds), float("nan"))
            scores = []
            best_v, best_m = None, -999.0
            for val in (list(sweep.get(param, [])) +
                        [bval]):  # include baseline
                s = cell.get((param, str(val), ds), [])
                if s:
                    m = float(np.mean(s))
                    if m > best_m:
                        best_m, best_v = m, val
            if best_v is not None:
                delta = best_m - base if not np.isnan(base) else float("nan")
                row += f"  {best_m:>6.4f}({delta:>+.4f})"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    # Best value per param
    print(f"\n  Best values (mean R²_adj across all datasets):")
    for param, _ in ranked:
        all_vals = list(sweep.get(param, []))
        param_cell: dict[str, list] = defaultdict(list)
        for (p, val_str, ds), scores in cell.items():
            if p == param:
                param_cell[val_str].extend(scores)
        if not param_cell:
            continue
        best_val_str = max(param_cell, key=lambda k: np.mean(param_cell[k]))
        best_mean = float(np.mean(param_cell[best_val_str]))
        base_scores = param_cell.get(str(BASELINE.get(param, "")), [])
        base_m = float(np.mean(base_scores)) if base_scores else float("nan")
        print(f"    {param:<22}  best={best_val_str:<10}  "
              f"R²_adj={best_mean:.4f}  (baseline={base_m:.4f}  "
              f"delta={best_mean - base_m:+.4f})")


def print_oat_summary() -> None:
    _print_oat_table(CSV[2], "Phase 2 OAT Primary Summary", PRIMARY_SWEEP)


def print_edge_summary() -> None:
    if not os.path.exists(CSV[3]):
        return
    print("\n" + "=" * 70)
    print("  Phase 3 — Edge Extension Summary")
    print("=" * 70)

    # Load Phase 2 baseline per param (baseline value for each param across all datasets)
    p2_base: dict[str, list] = defaultdict(list)
    if os.path.exists(CSV[2]):
        with open(CSV[2], newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    try:
                        bval = str(BASELINE.get(r["param"], ""))
                        if r["value"] == bval:
                            v = float(r["val_skill_adj"])
                            if not np.isnan(v):
                                p2_base[r["param"]].append(v)
                    except (ValueError, KeyError):
                        pass

    cell: dict[tuple, list] = defaultdict(list)
    with open(CSV[3], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                try:
                    v = float(r["val_skill_adj"])
                    if not np.isnan(v):
                        cell[(r["param"], r["value"])].append(v)
                except (ValueError, KeyError):
                    pass

    if not cell:
        print("  No valid edge extension results.")
        return

    params = sorted(set(p for p, _ in cell))
    print(f"  {'parameter':<22}  {'value':<8}  {'R²_adj':>8}  {'vs phase2 baseline':>20}")
    print("  " + "-" * 65)
    for param in params:
        base_m = float(np.mean(p2_base[param])) if p2_base.get(param) else float("nan")
        vals = sorted(set(v for p, v in cell if p == param),
                      key=lambda x: float(x) if x.replace(".", "").lstrip("-").isdigit() else x)
        for val in vals:
            scores = cell[(param, val)]
            m = float(np.mean(scores))
            delta = m - base_m if not np.isnan(base_m) else float("nan")
            marker = " ◄ better" if delta > 0.001 else ""
            print(f"  {param:<22}  {val:<8}  {m:>8.5f}  {delta:>+8.5f}{marker}")
    print(f"\n  Phase 2 baseline: {np.mean(list(p2_base.values())[0]) if p2_base else 'n/a':.5f}"
          if p2_base else "")


def print_secondary_summary() -> None:
    _print_oat_table(CSV[4], "Phase 4 OAT Secondary Summary", SECONDARY_SWEEP)


def print_pairwise_summary() -> None:
    if not os.path.exists(CSV[5]):
        print("  Phase 5 not run yet.")
        return
    print("\n" + "=" * 70)
    print("  Phase 5 — Pairwise interaction summary")
    print("=" * 70)

    # param2/val2 not stored as CSV columns (OAT_FIELDS used); recover from config_json
    cell: dict[tuple, list] = defaultdict(list)
    with open(CSV[5], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            try:
                v = float(r["val_skill_adj"])
                if np.isnan(v):
                    continue
                p1  = r["param"]
                v1  = r["value"]
                # recover second param from config_json diff against baseline
                p2, v2 = "", ""
                try:
                    cfg = json.loads(r.get("config_json", "{}"))
                    for k, vv in cfg.items():
                        if k == "random_state" or k == p1:
                            continue
                        if str(BASELINE.get(k, "")) != str(vv):
                            p2, v2 = k, vv
                            break
                except Exception:
                    pass
                cell[(p1, v1, p2, v2)].append(v)
            except (ValueError, KeyError):
                pass

    if not cell:
        print("  No valid pairwise results.")
        return

    # Best combo overall
    best_key = max(cell, key=lambda k: np.mean(cell[k]))
    best_mean = float(np.mean(cell[best_key]))
    print(f"\n  Best combo: {best_key[0]}={best_key[1]}, "
          f"{best_key[2]}={best_key[3]}  "
          f"R²_adj={best_mean:.4f}  (n={len(cell[best_key])} evals)")


def print_residual_summary() -> None:
    if not os.path.exists(CSV[6]):
        print("  Phase 6 not run yet.")
        return
    print("\n" + "=" * 70)
    print("  Phase 6 — Residual geometry summary")
    print("=" * 70)

    from scipy.stats import spearmanr

    rows = []
    with open(CSV[6], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok" or r.get("partition_id") == "-1":
                continue
            try:
                rows.append({
                    "dataset":        r["dataset"],
                    "n_real":         int(r["n_real"]),
                    "abs_mean_resid": float(r["abs_mean_resid"]),
                    "std_resid":      float(r["std_resid"]),
                    "T_value":        float(r["T_value"]),
                    "frac_in_cone":   float(r["frac_in_cone"]),
                    "mean_abs_z":     float(r["mean_abs_z"]),
                    "cone_degenerate": r["cone_degenerate"],
                })
            except (ValueError, KeyError):
                pass

    if not rows:
        print("  No valid residual rows.")
        return

    datasets = sorted(set(r["dataset"] for r in rows))
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds
                   and all(not np.isnan(r[k]) for k in
                           ["abs_mean_resid", "T_value", "n_real"])]
        if len(ds_rows) < 5:
            continue
        abs_r  = np.array([r["abs_mean_resid"] for r in ds_rows])
        n_real = np.array([r["n_real"]         for r in ds_rows])
        T      = np.array([r["T_value"]        for r in ds_rows])
        fic    = np.array([r["frac_in_cone"]   for r in ds_rows])

        r_n, _ = spearmanr(abs_r, n_real)
        r_T, _ = spearmanr(abs_r, T)
        r_f, _ = spearmanr(abs_r, fic)
        n_degen = sum(1 for r in ds_rows if r["cone_degenerate"] == "True")

        print(f"\n  {ds}  ({len(ds_rows)} partitions)")
        print(f"    Spearman |resid| vs n_real      : {r_n:+.3f}  "
              f"{'(small partitions have more error)' if r_n < -0.3 else ''}")
        print(f"    Spearman |resid| vs T_value     : {r_T:+.3f}  "
              f"{'(geometrically diverse partitions have more error)' if r_T > 0.3 else ''}")
        print(f"    Spearman |resid| vs frac_in_cone: {r_f:+.3f}")
        print(f"    Degenerate cones: {n_degen}/{len(ds_rows)}")

# ── Two-set recommendation ────────────────────────────────────────────────────

def _oat_best_config(metric: str = "val_skill_adj") -> dict:
    """
    Assemble best per-parameter config from OAT phases 2+3+4 using <metric>.
    For each parameter the value with the highest mean <metric> across all
    datasets and seeds is selected.  Later phases override earlier ones.
    """
    config = dict(BASELINE)
    for csv_path in [CSV[2], CSV[3], CSV[4]]:
        bests = _best_value_per_param(csv_path, metric)
        config.update(bests)
    return config


def print_two_set_summary() -> None:
    """
    Print the two recommended GeoXGB parameter sets side-by-side:

      Set 1 (NMAE-optimal):  maximise val_skill_adj
        = noise-floor-corrected NMAE skill
        = (nmae_baseline - nmae_model) / (nmae_baseline - nmae_floor)
        where nmae_baseline = MAE_mean_predictor / std(y).
        Sensitive to scale via std(y); penalises large errors on outlier-rich
        targets less than MAD-skill does.

      Set 2 (MAD-skill-optimal):  maximise val_mad_skill
        = (MAE_median_pred - MAE_model) / (MAE_median_pred - MAE_floor)
        where MAE_median_pred = mean(|y_val - median(y_train)|)
        and   MAE_floor = sigma_noise * sqrt(2/pi)  [same irreducible floor].
        Baseline is the Bayes-optimal constant L1 predictor (predict-median).
        Robust to heavy tails: MAE_median_pred is unaffected by outliers
        that inflate std(y).  Noise-floor corrected identically to Set 1.

    When to use each set:
      NMAE-optimal  --  target is approximately symmetric (synthetic benchmarks,
                        z-score normalised data).  std(y) is a fair scale.
      MAD-optimal   --  target is skewed or heavy-tailed (house prices, income,
                        biological assays).  std(y) is inflated by extremes;
                        the median predictor is the right baseline.

    The two sets are derived from the highest-metric config across OAT phases
    2-4, with pairwise phase 5 used if available.
    """
    print("\n" + "=" * 72)
    print("  TWO RECOMMENDED PARAMETER SETS")
    print("=" * 72)
    print()
    print("  Metric definitions:")
    print("  NMAE-skill : (MAE_mean_pred - MAE) / (MAE_mean_pred - MAE_floor)  [noise-floor corrected]")
    print("               Baseline = predict training mean.  Floor = sigma*sqrt(2/pi).")
    print("               Sensitive to outlier scale (outliers inflate std and MAE_mean_pred).")
    print("  MAD-skill  : (MAE_median_pred - MAE) / (MAE_median_pred - MAE_floor)  [noise-floor corrected]")
    print("               Baseline = predict training median (Bayes-optimal L1 constant).")
    print("               Same irreducible floor.  Robust to skewed / heavy-tailed targets.")
    print()

    # Prefer pairwise results; fall back to OAT-assembled config
    if os.path.exists(CSV[5]):
        nmae_cfg = _best_config_from_pairwise("val_skill_adj")
        mad_cfg  = _best_config_from_pairwise("val_mad_skill")
        source_label = "Phase 5 pairwise"
    else:
        nmae_cfg = _oat_best_config("val_skill_adj")
        mad_cfg  = _oat_best_config("val_mad_skill")
        source_label = "OAT phases 2-4 (phase 5 not yet run)"

    print(f"  Source: {source_label}")
    print()

    all_params = sorted(set(list(PRIMARY_SWEEP.keys()) + list(SECONDARY_SWEEP.keys())))

    hdr = (f"  {'Parameter':<24}  {'Baseline':>12}  "
           f"{'Set1 NMAE':>12}  {'Set2 MAD':>12}  {'Diff?'}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    diffs: list[str] = []
    for param in all_params:
        base_val = BASELINE.get(param, "—")
        nmae_val = nmae_cfg.get(param, base_val)
        mad_val  = mad_cfg.get(param, base_val)
        differ   = str(nmae_val) != str(mad_val)
        mark     = " <--" if differ else ""
        if differ:
            diffs.append(param)
        print(f"  {param:<24}  {str(base_val):>12}  "
              f"{str(nmae_val):>12}  {str(mad_val):>12}  {mark}")

    print()
    if diffs:
        print(f"  Parameters where sets diverge: {diffs}")
        print()
        print("  Interpretation:")
        print("    Divergence means the optimal value depends on whether errors")
        print("    are measured against std(y) vs the median predictor.")
        print("    -> Use Set 1 when std(y) is a fair scale (symmetric targets).")
        print("    -> Use Set 2 when target has skew / heavy tails (robust scale).")
    else:
        print("  Both sets agree on all parameters.")
        print("  This is expected for symmetric synthetic datasets (mean ~ median,")
        print("  MAD ~ 0.67*std).  Divergence typically appears with skewed real-world")
        print("  targets such as house prices, survival times, or count data.")

    # Per-metric baseline vs best-config mean (from OAT phase 2, if available)
    print()
    print("  Mean metric improvement over BASELINE config (OAT phase 2):")
    for metric, label, cfg in [
        ("val_skill_adj", "NMAE-skill", nmae_cfg),
        ("val_mad_skill",  "MAD-skill",  mad_cfg),
    ]:
        if not os.path.exists(CSV[2]):
            print(f"    {label}: phase 2 not run yet.")
            continue
        base_scores: list[float] = []
        best_scores: list[float] = []
        with open(CSV[2], newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                try:
                    v = float(r[metric])
                    if np.isnan(v):
                        continue
                    bval = str(BASELINE.get(r["param"], ""))
                    if r["value"] == bval:
                        base_scores.append(v)
                    cfg_val = cfg.get(r["param"])
                    if cfg_val is not None and r["value"] == str(cfg_val):
                        best_scores.append(v)
                except (ValueError, KeyError):
                    pass
        if base_scores and best_scores:
            bm = float(np.mean(base_scores))
            fm = float(np.mean(best_scores))
            print(f"    {label:<12}: baseline={bm:.4f}  optimized≈{fm:.4f}  "
                  f"delta={fm - bm:+.4f}")
        else:
            print(f"    {label:<12}: insufficient data (re-run phase 2 to populate "
                  f"{metric}).")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GeoXGB Regression Meta-Analysis v2"
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Run a single phase only")
    parser.add_argument("--summary", action="store_true",
                        help="Print all summaries from saved CSVs and exit")
    parser.add_argument("--jobs", type=int, default=0,
                        help="Workers (0 = cpu_count)")
    args = parser.parse_args()

    n_cpu    = mp.cpu_count()
    n_jobs   = n_cpu if args.jobs <= 0 else args.jobs
    run_all  = args.phase is None and not args.summary

    # Limit BLAS threads to 1 per worker process.
    # With 'spawn' context each worker fully reinitialises Python; OpenBLAS would
    # otherwise claim all cores, turning n_jobs workers into n_jobs×n_cpu threads
    # and exhausting virtual memory on large-core machines.
    # Must be set BEFORE the Pool is created — spawned workers inherit env.
    import os as _os
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        _os.environ.setdefault(_var, "1")

    print("\nGeoXGB Regression Meta-Analysis v2")
    print(f"  CPUs: {n_cpu}  workers: {n_jobs}")
    print(f"  Results dir: {RESULTS_DIR}")

    if args.summary:
        print_noise_summary()
        print_oat_summary()
        print_edge_summary()
        print_secondary_summary()
        print_pairwise_summary()
        print_residual_summary()
        print_two_set_summary()
        return

    datasets = _make_datasets()
    for name, (X, y, sigma, ceil) in datasets.items():
        print(f"  {name:<12}  n={X.shape[0]}  d={X.shape[1]}  "
              f"sigma={sigma}  R²_ceil={ceil:.4f}")

    t_global = time.perf_counter()

    ctx = mp.get_context("spawn")  # Windows-safe
    with ctx.Pool(n_jobs, initializer=_worker_init) as pool:

        if run_all or args.phase == 1:
            run_phase1(pool)

        if run_all or args.phase == 2:
            run_phase2(pool, datasets)

        if run_all or args.phase == 3:
            run_phase3(pool, datasets)

        if run_all or args.phase == 4:
            run_phase4(pool, datasets)

        if run_all or args.phase == 5:
            run_phase5(pool, datasets)

    # Phase 6 runs in main process (requires direct model introspection)
    if run_all or args.phase == 6:
        run_phase6(datasets)

    total = time.perf_counter() - t_global
    print(f"\nTotal wall time: {total / 3600:.2f} h ({total / 60:.1f} min)")

    print_noise_summary()
    print_oat_summary()
    print_edge_summary()
    print_secondary_summary()
    print_pairwise_summary()
    print_residual_summary()
    print_two_set_summary()


if __name__ == "__main__":
    main()
