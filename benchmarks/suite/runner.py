"""
Main benchmark runner.

Default mode: 5-fold CV per dataset, evaluate both models on each fold.
HPO mode: 5 random 80/20 train/test splits, run HPO on train, evaluate on test.

All results are written to CSV with full traceability (category, dataset, model,
seed/fold, all metrics, timing, and any errors).
"""
from __future__ import annotations

import logging
import sys
import time
import traceback
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
    """Run 5-fold CV for one dataset, both models. Returns list of row dicts."""
    log.info("DEFAULT | %s [%s] — loading data", ds["name"], ds["category"])
    X, y = ds["loader"]()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, d = X.shape
    log.info("  n=%d, d=%d, task=%s", n, d, ds["task"])

    task = ds["task"]
    models = get_models("default", task)

    if task == "regression":
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEEDS[0])
        splits = list(kf.split(X))
    else:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEEDS[0])
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
    """Run HPO for one dataset across 5 seeds. Returns list of row dicts."""
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
        models = get_models("hpo", task, n_trials=n_trials)

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

def run(mode: str = "default",
        category: str | None = None,
        dataset_name: str | None = None,
        save_models: bool = False,
        output: str | None = None):
    """
    Run the full benchmark suite.

    Parameters
    ----------
    mode : 'default' | 'hpo' | 'all'
    category : optional filter to a single category
    dataset_name : optional filter to a single dataset
    save_models : if True, serialize fitted models for later inspection
    output : optional output CSV path (default: results/<mode>_results.csv)
    """
    datasets = get_datasets(category, dataset_name)
    if not datasets:
        log.error("No datasets matched filters: category=%s, dataset=%s",
                  category, dataset_name)
        return

    n_datasets = len(datasets)
    categories = sorted(set(d["category"] for d in datasets))
    log.info("=" * 70)
    log.info("BENCHMARK SUITE — mode=%s, %d datasets across %d categories",
             mode, n_datasets, len(categories))
    log.info("Categories: %s", ", ".join(categories))
    log.info("Save models: %s", save_models)
    log.info("=" * 70)

    all_rows = []
    modes = ["default", "hpo"] if mode == "all" else [mode]

    for m in modes:
        log.info("-" * 70)
        log.info("Starting mode: %s", m)
        log.info("-" * 70)

        for i, ds in enumerate(datasets, 1):
            log.info("[%d/%d] %s", i, n_datasets, ds["name"])
            t0 = time.perf_counter()

            if m == "default":
                rows = run_default_single(ds, save_models=save_models)
            else:
                rows = run_hpo_single(ds, save_models=save_models)

            # Tag rows with mode
            for r in rows:
                r["mode"] = m

            all_rows.extend(rows)
            elapsed = time.perf_counter() - t0
            log.info("  Completed in %.1fs", elapsed)

    # Write results
    df = pd.DataFrame(all_rows)
    if output:
        out_path = Path(output)
    else:
        out_path = RESULTS_DIR / f"{mode}_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Results written to %s (%d rows)", out_path, len(df))

    # Print summary
    _print_summary(df)
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
