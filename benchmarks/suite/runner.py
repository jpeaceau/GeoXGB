"""
Main benchmark runner.

Default mode: 3-fold CV per dataset, evaluate both models on each fold.
HPO mode: 3 seeds × 80/20 train/test splits, run HPO on train, evaluate on test.

All results are written to CSV with full traceability (category, dataset, model,
seed/fold, all metrics, timing, and any errors).
"""
from __future__ import annotations

import logging
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .config import SEEDS, N_FOLDS, RESULTS_DIR, MODELS_DIR, hpo_trials_for_n
from .datasets import get_datasets, list_categories, DATASETS
from .metrics import compute_metrics
from .models import get_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "benchmark.log", mode="a"),
    ],
)
log = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Single dataset + model evaluation
# ---------------------------------------------------------------------------

def _evaluate_fold(model, X_train, y_train, X_test, y_test, task: str):
    """Fit model on train, predict on test, return metrics + timing."""
    model.fit(X_train, y_train)
    fit_time = model.fit_time_

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    y_proba = None
    if task != "regression":
        try:
            proba = model.predict_proba(X_test)
            y_proba = proba[:, 1] if task == "binary" else proba
        except Exception:
            y_proba = None

    metrics = compute_metrics(task, y_test, y_pred, y_proba)
    metrics["fit_time_s"] = fit_time
    metrics["predict_time_s"] = predict_time
    return metrics


# ---------------------------------------------------------------------------
# Default mode: 5-fold CV
# ---------------------------------------------------------------------------

def run_default_single(ds: dict, save_models: bool = False) -> list[dict]:
    """Run N-fold CV for one dataset, both models. Returns list of row dicts."""
    log.info("DEFAULT | %s [%s] — loading data", ds["name"], ds["category"])
    X, y = ds["loader"]()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, d = X.shape
    log.info("  n=%d, d=%d, task=%s", n, d, ds["task"])

    task = ds["task"]
    seed = SEEDS[0]
    models = get_models("default", task, random_state=seed)

    if task == "regression":
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        splits = list(kf.split(X))
    else:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        splits = list(kf.split(X, y))

    rows = []
    for model in models:
        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            try:
                metrics = _evaluate_fold(
                    model, X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], task,
                )
                row = {
                    "category": ds["category"],
                    "dataset": ds["name"],
                    "task": task,
                    "model": model.name,
                    "fold": fold_idx,
                    "seed": SEEDS[0],
                    "n_samples": n,
                    "n_features": d,
                    **metrics,
                    "error": "",
                }
                log.info("  %s fold=%d  %s",
                         model.name, fold_idx,
                         "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                   if isinstance(v, float)))
                if save_models:
                    model_path = (MODELS_DIR / "default" / ds["category"]
                                  / f"{ds['name']}_{model.name}_fold{fold_idx}.pkl")
                    try:
                        model.save(model_path)
                    except Exception as e:
                        log.warning("  Failed to save model: %s", e)

            except Exception as e:
                tb = traceback.format_exc()
                log.error("  FAILED %s fold=%d: %s\n%s", model.name, fold_idx, e, tb)
                row = {
                    "category": ds["category"],
                    "dataset": ds["name"],
                    "task": task,
                    "model": model.name,
                    "fold": fold_idx,
                    "seed": SEEDS[0],
                    "n_samples": n,
                    "n_features": d,
                    "error": str(e),
                }
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# HPO mode: 5 seeds × train/test split
# ---------------------------------------------------------------------------

def run_hpo_single(ds: dict, save_models: bool = False) -> list[dict]:
    """Run HPO for one dataset across all seeds. Returns list of row dicts."""
    log.info("HPO | %s [%s] — loading data", ds["name"], ds["category"])
    X, y = ds["loader"]()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, d = X.shape
    task = ds["task"]
    n_trials = hpo_trials_for_n(n)
    log.info("  n=%d, d=%d, task=%s, hpo_trials=%d", n, d, task, n_trials)

    rows = []
    for seed in SEEDS:
        models = get_models("hpo", task, n_trials=n_trials, random_state=seed)

        # Stratified split for classification
        if task == "regression":
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=seed,
            )
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y,
            )

        for model in models:
            try:
                metrics = _evaluate_fold(model, X_tr, y_tr, X_te, y_te, task)
                row = {
                    "category": ds["category"],
                    "dataset": ds["name"],
                    "task": task,
                    "model": model.name,
                    "fold": -1,
                    "seed": seed,
                    "n_samples": n,
                    "n_features": d,
                    **metrics,
                    "error": "",
                }
                log.info("  %s seed=%d  %s",
                         model.name, seed,
                         "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                   if isinstance(v, float)))

                if save_models:
                    model_path = (MODELS_DIR / "hpo" / ds["category"]
                                  / f"{ds['name']}_{model.name}_seed{seed}.pkl")
                    try:
                        model.save(model_path)
                    except Exception as e:
                        log.warning("  Failed to save model: %s", e)

                # Save HPO best params if available
                if hasattr(model, "best_params_") and model.best_params_ is not None:
                    row["hpo_best_params"] = str(model.best_params_)
                    row["hpo_cv_score"] = getattr(model, "best_score_",
                                                   getattr(model, "optimizer_", None)
                                                   and model.optimizer_.best_score_)
                elif hasattr(model, "optimizer_") and model.optimizer_ is not None:
                    row["hpo_best_params"] = str(model.optimizer_.best_params_)
                    row["hpo_cv_score"] = model.optimizer_.best_score_

                # Extended HPO evaluation: same best params but
                # n_rounds=5000, convergence_tol=None (no early stopping)
                if (hasattr(model, "extended_model_")
                        and model.extended_model_ is not None):
                    try:
                        ext_model = model.extended_model_
                        t0 = time.perf_counter()
                        ext_pred = ext_model.predict(X_te)
                        ext_predict_time = time.perf_counter() - t0

                        ext_proba = None
                        if task != "regression":
                            try:
                                p = ext_model.predict_proba(X_te)
                                ext_proba = p[:, 1] if task == "binary" else p
                            except Exception:
                                ext_proba = None

                        ext_metrics = compute_metrics(task, y_te, ext_pred, ext_proba)
                        ext_metrics["fit_time_s"] = model.extended_fit_time_
                        ext_metrics["predict_time_s"] = ext_predict_time

                        ext_row = {
                            "category": ds["category"],
                            "dataset": ds["name"],
                            "task": task,
                            "model": "geoxgb_hpo_extended",
                            "fold": -1,
                            "seed": seed,
                            "n_samples": n,
                            "n_features": d,
                            **ext_metrics,
                            "error": "",
                            "hpo_best_params": row.get("hpo_best_params", ""),
                            "hpo_cv_score": row.get("hpo_cv_score", ""),
                        }
                        log.info("  geoxgb_hpo_extended seed=%d  %s",
                                 seed,
                                 "  ".join(f"{k}={v:.4f}"
                                           for k, v in ext_metrics.items()
                                           if isinstance(v, float)))
                        rows.append(ext_row)
                    except Exception as ext_e:
                        log.warning("  Extended HPO eval failed: %s", ext_e)

            except Exception as e:
                tb = traceback.format_exc()
                log.error("  FAILED %s seed=%d: %s\n%s", model.name, seed, e, tb)
                row = {
                    "category": ds["category"],
                    "dataset": ds["name"],
                    "task": task,
                    "model": model.name,
                    "fold": -1,
                    "seed": seed,
                    "n_samples": n,
                    "n_features": d,
                    "error": str(e),
                }
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _run_dataset_task(ds_name: str, m: str, save_models: bool,
                      threads_per_worker: int | None = None) -> list[dict]:
    """Worker function for parallel execution. Runs one dataset in one mode."""
    import os
    if threads_per_worker is not None:
        # Divide cores evenly: limit OpenMP (GeoXGB) and XGBoost thread pools
        os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
        os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
        # Store for model wrappers to pick up
        os.environ["GEOXGB_BENCH_NJOBS"] = str(threads_per_worker)
    # Look up dataset by name in this process (avoids lambda pickling issues)
    matches = [d for d in DATASETS if d["name"] == ds_name]
    if not matches:
        raise ValueError(f"Dataset not found: {ds_name}")
    ds = matches[0]
    t0 = time.perf_counter()
    if m == "default":
        rows = run_default_single(ds, save_models=save_models)
    else:
        rows = run_hpo_single(ds, save_models=save_models)
    for r in rows:
        r["mode"] = m
    elapsed = time.perf_counter() - t0
    log.info("  %s [%s] completed in %.1fs", ds["name"], m, elapsed)
    return rows


def run(mode: str = "default",
        category: str | None = None,
        dataset_name: str | None = None,
        save_models: bool = False,
        output: str | None = None,
        workers: int = 1):
    """
    Run the full benchmark suite.

    Parameters
    ----------
    mode : 'default' | 'hpo' | 'all'
    category : optional filter to a single category
    dataset_name : optional filter to a single dataset
    save_models : if True, serialize fitted models for later inspection
    output : optional output CSV path (default: results/<mode>_results.csv)
    workers : int, parallel dataset workers (1 = sequential)
    """
    datasets = get_datasets(category, dataset_name)
    if not datasets:
        log.error("No datasets matched filters: category=%s, dataset=%s",
                  category, dataset_name)
        return

    n_datasets = len(datasets)
    categories = sorted(set(d["category"] for d in datasets))
    log.info("=" * 70)
    log.info("BENCHMARK SUITE — mode=%s, %d datasets across %d categories, workers=%d",
             mode, n_datasets, len(categories), workers)
    log.info("Categories: %s", ", ".join(categories))
    log.info("Save models: %s", save_models)
    log.info("=" * 70)

    all_rows = []
    modes = ["default", "hpo"] if mode == "all" else [mode]

    # Single output CSV for all results (mode is stored per-row)
    if output:
        out_path = Path(output)
    else:
        out_path = RESULTS_DIR / "results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing CSV if present (append new rows)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        all_rows = existing.to_dict("records")
        log.info("Resuming from %s (%d existing rows)", out_path.name, len(all_rows))

    # Compute threads per worker to avoid oversubscription.
    # Each worker gets total_cores / workers threads for OpenMP + XGBoost.
    import os
    total_cores = os.cpu_count() or 4
    threads_per_worker = max(1, total_cores // max(workers, 1))
    log.info("CPU cores: %d, threads per worker: %d", total_cores, threads_per_worker)

    def _flush_rows(rows: list[dict]):
        """Append new rows to the global CSV after each dataset completes."""
        all_rows.extend(rows)
        df = pd.DataFrame(all_rows)
        df.to_csv(out_path, index=False)

    if workers <= 1:
        # Sequential — give this process all cores
        os.environ["OMP_NUM_THREADS"] = str(total_cores)
        os.environ["GEOXGB_BENCH_NJOBS"] = str(total_cores)
        for m in modes:
            log.info("-" * 70)
            log.info("Starting mode: %s", m)
            log.info("-" * 70)
            for i, ds in enumerate(datasets, 1):
                log.info("[%d/%d] %s", i, n_datasets, ds["name"])
                rows = _run_dataset_task(ds["name"], m, save_models)
                _flush_rows(rows)
                log.info("  Results saved (%d total rows in %s)", len(all_rows), out_path.name)
    else:
        # Parallel across datasets — divide cores evenly
        tasks = [(ds["name"], m) for m in modes for ds in datasets]
        log.info("Submitting %d tasks to %d workers (%d threads each)",
                 len(tasks), workers, threads_per_worker)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_dataset_task, name, m, save_models,
                            threads_per_worker): (name, m)
                for name, m in tasks
            }
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                name, m = futures[future]
                try:
                    rows = future.result()
                    _flush_rows(rows)
                    log.info("[%d/%d] %s (%s) done — %d total rows saved",
                             done_count, len(tasks), name, m, len(all_rows))
                except Exception as e:
                    log.error("[%d/%d] %s (%s) FAILED: %s",
                              done_count, len(tasks), name, m, e)

    log.info("Final results: %s (%d rows)", out_path, len(all_rows))

    # Print summary
    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        _print_summary(df)
    else:
        log.warning("No results to summarize.")
    return df


def _print_summary(df: pd.DataFrame):
    """Print a concise summary table to stdout."""
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)

    errors = df[df["error"].fillna("").str.len() > 0]
    if len(errors) > 0:
        log.warning("%d errors encountered:", len(errors))
        for _, row in errors.iterrows():
            log.warning("  %s / %s / %s: %s",
                        row["dataset"], row["model"], row.get("seed", "?"),
                        row["error"][:120])

    # Aggregate by dataset × model
    metric_cols = ["r2", "mae", "rmse", "mse", "auc", "f1", "accuracy",
                   "precision", "recall", "fit_time_s", "predict_time_s"]
    available = [c for c in metric_cols if c in df.columns]

    valid = df[df["error"].fillna("").str.len() == 0]
    if len(valid) == 0:
        log.warning("No successful runs to summarize.")
        return

    summary = (valid.groupby(["mode", "category", "dataset", "model"])[available]
               .mean().round(4).reset_index())
    summary_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Summary written to %s", summary_path)

    # Print to console
    print("\n" + summary.to_string(index=False))
