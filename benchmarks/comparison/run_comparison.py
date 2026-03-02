"""
GeoXGB vs XGBoost vs LightGBM — Real-World Regression Benchmark
================================================================
Tests GeoXGB's performance and robustness on 8 diverse real-world datasets.

Evaluation axes
---------------
  1. I.i.d. accuracy   — mean R² ± std across 3 seeds × 5 folds
  2. Robustness        — ΔR² when Gaussian noise (σ=0.2×, 0.5× feat std)
                         is added to test features at inference time
  3. GeoXGB geometry   — Spearman(z-NN distance, |residual|) per fold:
                         positive ρ means z-proximity predicts prediction quality

Model configs (fixed, no early stopping, no per-dataset tuning):
  GeoXGB   n_rounds=1000  lr=0.05  max_depth=2  (CppGeoXGBRegressor)
  XGBoost  n_estimators=1000  max_depth=6  lr=0.05  (matched budget)
  LightGBM n_estimators=1000  num_leaves=63  lr=0.05  (matched budget)

Runtime estimate: ~30–90 min depending on CPUs and dataset sizes.
  --fast flag: use n_rounds=300 / n_estimators=300 + 1 seed for quick runs

Usage
-----
    cd benchmarks/comparison
    python run_comparison.py
    python run_comparison.py --jobs 8
    python run_comparison.py --fast
    python run_comparison.py --summary          # reprint from saved CSV
"""
from __future__ import annotations

import argparse
import csv
import io
import multiprocessing
import os
import sys
import time
import warnings

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ── Paths ────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.join(_HERE, "..", "..")
RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, "comparison.csv")

sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _HERE)

# ── Model configs ────────────────────────────────────────────────────────────

_GEOXGB_BASE = dict(
    max_depth=2, min_samples_leaf=5,
    reduce_ratio=0.7, expand_ratio=0.1, y_weight=0.5, refit_interval=5,
    auto_noise=False, noise_guard=False, refit_noise_floor=0.0,
    auto_expand=True, min_train_samples=5000, bandwidth="auto",
    variance_weighted=True, hvrt_min_samples_leaf=-1, n_partitions=-1, n_bins=64,
)

_XGB_BASE = dict(
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, reg_alpha=0.0, reg_lambda=1.0,
    tree_method="hist", n_jobs=1, verbosity=0,
)

_LGB_BASE = dict(
    max_depth=6, num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=10, n_jobs=1, verbose=-1,
)

NOISE_FRACS = [0.2, 0.5]   # multiplier × per-feature std
SEEDS       = [0, 1, 2]
N_FOLDS     = 5

# ── CSV schema ───────────────────────────────────────────────────────────────

FIELDNAMES = [
    "model", "dataset", "n", "d",
    "seed", "fold",
    "r2", "r2_n02", "r2_n05",
    "z_nn_spearman", "z_nn_p",
    "train_s", "status",
]

# ── Per-fold worker ──────────────────────────────────────────────────────────

def _eval_fold(
    model_tag:  str,
    X_tr:       np.ndarray,
    y_tr:       np.ndarray,
    X_te:       np.ndarray,
    y_te:       np.ndarray,
    seed:       int,
    feat_stds:  np.ndarray,   # per-feature std from training data
    n_rounds:   int,
    lr:         float,
) -> dict:
    """Evaluate one (model, fold, seed) combination. Thread-safe."""
    import warnings as _w
    _w.filterwarnings("ignore")

    t0 = time.perf_counter()
    try:
        # ── Fit ──────────────────────────────────────────────────────────────
        if model_tag == "GeoXGB":
            from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBRegressor
            cfg = make_cpp_config(
                **_GEOXGB_BASE,
                n_rounds=n_rounds,
                learning_rate=lr,
                random_state=seed,
            )
            model = CppGeoXGBRegressor(cfg)
            model.fit(X_tr, y_tr)

        elif model_tag == "XGBoost":
            from xgboost import XGBRegressor
            model = XGBRegressor(
                **_XGB_BASE,
                n_estimators=n_rounds,
                learning_rate=lr,
                random_state=seed,
            )
            model.fit(X_tr, y_tr)

        elif model_tag == "LightGBM":
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                **_LGB_BASE,
                n_estimators=n_rounds,
                learning_rate=lr,
                random_state=seed,
            )
            model.fit(X_tr, y_tr)

        else:
            raise ValueError(f"Unknown model_tag={model_tag!r}")

        # ── Baseline R² ──────────────────────────────────────────────────────
        yhat   = model.predict(X_te)
        r2_b   = float(r2_score(y_te, yhat))
        errors = y_te - yhat

        # ── Robustness: add Gaussian noise to test features ───────────────────
        rng_n = np.random.default_rng(seed + 9999)
        r2_n02 = r2_n05 = float("nan")
        for frac in NOISE_FRACS:
            noise  = frac * feat_stds * rng_n.standard_normal(X_te.shape)
            yhat_n = model.predict(X_te + noise)
            val    = float(r2_score(y_te, yhat_n))
            if frac == 0.2:
                r2_n02 = val
            else:
                r2_n05 = val

        # ── GeoXGB geometry: z-NN distance Spearman ───────────────────────────
        # Fit a lightweight Python HVRT on training data to get z-coordinates.
        # (CppGeoXGBRegressor does not expose z-space; HVRT fit is fast.)
        z_rho = z_p = float("nan")
        if model_tag == "GeoXGB":
            try:
                from hvrt import HVRT
                from scipy.spatial import cKDTree
                from scipy.stats import spearmanr

                hvrt_m = HVRT(y_weight=0.5, random_state=seed)
                hvrt_m.fit(X_tr, y_tr)
                X_tr_z = np.asarray(hvrt_m.X_z_)
                X_te_z = np.asarray(hvrt_m._to_z(X_te))
                kdt    = cKDTree(X_tr_z)
                dists, _ = kdt.query(X_te_z, k=1, workers=1)
                rho, p   = spearmanr(dists, np.abs(errors))
                z_rho, z_p = float(rho), float(p)
            except Exception:
                pass   # geometry unavailable — leave as nan

        status = "ok"

    except Exception as exc:   # noqa: BLE001
        r2_b = r2_n02 = r2_n05 = z_rho = z_p = float("nan")
        status = repr(exc)[:140]

    return dict(
        r2=r2_b, r2_n02=r2_n02, r2_n05=r2_n05,
        z_nn_spearman=z_rho, z_nn_p=z_p,
        train_s=time.perf_counter() - t0,
        status=status,
    )


# ── Job builder ───────────────────────────────────────────────────────────────

def _build_jobs(datasets, seeds, n_folds, n_rounds, lr, done):
    """Return list of (args_for_worker, meta) pairs for all pending jobs."""
    model_tags = ["GeoXGB", "XGBoost", "LightGBM"]
    jobs = []
    for ds_name, (X, y, _desc) in datasets.items():
        feat_stds = X.std(axis=0) + 1e-10
        for seed in seeds:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold_i, (tr_idx, te_idx) in enumerate(kf.split(X)):
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_te, y_te = X[te_idx], y[te_idx]
                for tag in model_tags:
                    key = (tag, ds_name, str(seed), str(fold_i))
                    if key in done:
                        continue
                    jobs.append(dict(
                        model_tag=tag,
                        X_tr=X_tr, y_tr=y_tr,
                        X_te=X_te, y_te=y_te,
                        seed=seed,
                        feat_stds=feat_stds,
                        n_rounds=n_rounds,
                        lr=lr,
                        # meta
                        _ds_name=ds_name,
                        _n=len(y),
                        _d=X.shape[1],
                        _fold=fold_i,
                    ))
    return jobs


def _worker(job: dict) -> dict:
    meta = {k[1:]: v for k, v in job.items() if k.startswith("_")}
    result = _eval_fold(
        model_tag=job["model_tag"],
        X_tr=job["X_tr"], y_tr=job["y_tr"],
        X_te=job["X_te"], y_te=job["y_te"],
        seed=job["seed"],
        feat_stds=job["feat_stds"],
        n_rounds=job["n_rounds"],
        lr=job["lr"],
    )
    return dict(
        model=job["model_tag"],
        dataset=meta["ds_name"],
        n=meta["n"],
        d=meta["d"],
        seed=job["seed"],
        fold=meta["fold"],
        **result,
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def run(n_jobs: int = -1, fast: bool = False, n_rounds: int = 0) -> None:
    from datasets import load_datasets  # noqa: E402

    if n_rounds == 0:
        n_rounds = 300 if fast else 1000
    lr       = 0.05
    seeds    = [0] if fast else SEEDS
    n_folds  = 3  if fast else N_FOLDS

    print("\n" + "=" * 68)
    print("GeoXGB vs XGBoost vs LightGBM — Real-World Comparison")
    print("=" * 68)
    print(f"  Config: n_rounds={n_rounds}  lr={lr}  seeds={seeds}  folds={n_folds}")
    if fast:
        print("  [fast mode — reduced folds/rounds for quick preview]")
    print()

    print("Loading datasets …")
    datasets = load_datasets(verbose=True)
    if not datasets:
        print("No datasets loaded — aborting.")
        return
    print()

    # Resume support: skip already-saved rows
    done: set[tuple] = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["model"], row["dataset"], row["seed"], row["fold"]))
        if done:
            print(f"  Resuming: {len(done)} fold-results already saved.\n")

    jobs = _build_jobs(datasets, seeds, n_folds, n_rounds, lr, done)
    total = len(jobs) + len(done)

    if not jobs:
        print("All jobs already complete — printing summary from CSV.\n")
    else:
        n_workers = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        n_workers = min(max(1, n_workers), len(jobs))
        print(f"Running {len(jobs)} jobs on {n_workers} workers  "
              f"({len(done)}/{total} already saved) …\n")

        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_worker)(j) for j in jobs
        )

        write_header = not os.path.exists(OUT_CSV) or len(done) == 0
        with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in FIELDNAMES})

        errors_list = [r for r in results if r.get("status") != "ok"]
        if errors_list:
            print(f"\n  {len(errors_list)} fold(s) failed:")
            for e in errors_list[:5]:
                print(f"    {e['model']} / {e['dataset']} / "
                      f"seed {e['seed']} fold {e['fold']}: {e['status']}")

    print_summary()


# ── Summary printer ───────────────────────────────────────────────────────────

def _load_csv() -> list[dict]:
    if not os.path.exists(OUT_CSV):
        print(f"No results file found at {OUT_CSV}")
        return []
    with open(OUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("status") == "ok"]


def _agg(vals: list[float]) -> tuple[float, float]:
    """(mean, std) ignoring NaN."""
    v = [x for x in vals if x == x]  # filter NaN
    if not v:
        return float("nan"), float("nan")
    n = len(v)
    mu = sum(v) / n
    std = (sum((x - mu) ** 2 for x in v) / max(n - 1, 1)) ** 0.5
    return mu, std


def print_summary() -> None:
    rows = _load_csv()
    if not rows:
        return

    from collections import defaultdict
    # r2[model][dataset]
    r2:    dict = defaultdict(lambda: defaultdict(list))
    r2n02: dict = defaultdict(lambda: defaultdict(list))
    r2n05: dict = defaultdict(lambda: defaultdict(list))
    zrho:  dict = defaultdict(lambda: defaultdict(list))

    for r in rows:
        m, ds = r["model"], r["dataset"]
        def _f(k):
            try:
                return float(r[k])
            except (ValueError, KeyError):
                return float("nan")
        r2[m][ds].append(_f("r2"))
        r2n02[m][ds].append(_f("r2_n02"))
        r2n05[m][ds].append(_f("r2_n05"))
        z = _f("z_nn_spearman")
        if z == z:   # not NaN
            zrho[m][ds].append(z)

    models   = ["GeoXGB", "XGBoost", "LightGBM"]
    datasets = sorted({r["dataset"] for r in rows})
    cw       = 18   # column width

    def _cell(mu, sd):
        if mu != mu:
            return "---"
        return f"{mu:.3f}({sd:.3f})"

    def _dcell(vals_m, vals_ref):
        mu_m,   _ = _agg(vals_m)
        mu_ref, _ = _agg(vals_ref)
        if mu_m != mu_m or mu_ref != mu_ref:
            return "---"
        d = mu_m - mu_ref
        return f"{d:+.3f}"

    # ── Table 1: I.i.d. R² ────────────────────────────────────────────────
    sep = "=" * (20 + cw * len(datasets) + 6)
    print()
    print(sep)
    print("  TABLE 1 — I.i.d. R²   mean(std) over folds")
    print(sep)
    hdr = f"  {'model':<18}" + "".join(f"{d:>{cw}}" for d in datasets) + f"  {'avg':>6}"
    print(hdr)
    print("  " + "-" * (18 + cw * len(datasets) + 6))

    for m in models:
        cells = []
        means = []
        for ds in datasets:
            mu, sd = _agg(r2[m][ds])
            cells.append(_cell(mu, sd))
            if mu == mu:
                means.append(mu)
        avg = f"{sum(means)/len(means):.3f}" if means else "---"
        print(f"  {m:<18}" + "".join(f"{c:>{cw}}" for c in cells) + f"  {avg:>6}")

    # ── Table 2: Robustness ΔR² ───────────────────────────────────────────
    for noise_lab, noise_dict in [("σ=0.2×std", r2n02), ("σ=0.5×std", r2n05)]:
        print()
        print(sep)
        print(f"  TABLE 2 — Robustness ΔR²  (noisy − clean)   noise: {noise_lab}")
        print(sep)
        print(hdr)
        print("  " + "-" * (18 + cw * len(datasets) + 6))
        for m in models:
            cells = []
            deltas = []
            for ds in datasets:
                base_vals  = r2[m][ds]
                noisy_vals = noise_dict[m][ds]
                if base_vals and noisy_vals:
                    mu_b, _ = _agg(base_vals)
                    mu_n, _ = _agg(noisy_vals)
                    if mu_b == mu_b and mu_n == mu_n:
                        d = mu_n - mu_b
                        cells.append(f"{d:>+.3f}")
                        deltas.append(d)
                    else:
                        cells.append("---")
                else:
                    cells.append("---")
            avg = f"{sum(deltas)/len(deltas):>+.3f}" if deltas else "---"
            print(f"  {m:<18}" + "".join(f"{c:>{cw}}" for c in cells) + f"  {avg:>6}")

    # ── Table 3: ΔR² GeoXGB vs XGBoost ───────────────────────────────────
    print()
    print(sep)
    print("  TABLE 3 — ΔR² vs XGBoost  (positive = GeoXGB/LGB wins)")
    print(sep)
    print(hdr)
    print("  " + "-" * (18 + cw * len(datasets) + 6))
    for m in ["GeoXGB", "LightGBM"]:
        cells = []
        deltas = []
        for ds in datasets:
            cells.append(_dcell(r2[m][ds], r2["XGBoost"][ds]))
            mu_m, _  = _agg(r2[m][ds])
            mu_xg, _ = _agg(r2["XGBoost"][ds])
            if mu_m == mu_m and mu_xg == mu_xg:
                deltas.append(mu_m - mu_xg)
        avg = f"{sum(deltas)/len(deltas):>+.3f}" if deltas else "---"
        lbl = f"{m} − XGBoost"
        print(f"  {lbl:<18}" + "".join(f"{c:>{cw}}" for c in cells) + f"  {avg:>6}")

    # ── Table 4: GeoXGB geometry quality ─────────────────────────────────
    geo_rows = [(m, ds) for m in ["GeoXGB"] for ds in datasets if zrho[m][ds]]
    if geo_rows:
        print()
        print(sep)
        print("  TABLE 4 — GeoXGB geometry quality")
        print("  Spearman(z-NN distance, |residual|):  ρ > 0 → z-space predicts errors")
        print(sep)
        print(f"  {'dataset':<22}  {'mean ρ':>8}  {'std ρ':>8}  {'n_folds':>8}  interpretation")
        print("  " + "-" * 70)
        for ds in datasets:
            vals = zrho["GeoXGB"][ds]
            if not vals:
                continue
            mu, sd = _agg(vals)
            interp = ("geometry useful" if mu > 0.15
                      else "weak signal" if mu > 0.05
                      else "no signal")
            print(f"  {ds:<22}  {mu:>8.3f}  {sd:>8.3f}  {len(vals):>8d}  {interp}")

    # ── Table 5: Training time ────────────────────────────────────────────
    times: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            times[r["model"]][r["dataset"]].append(float(r["train_s"]))
        except (ValueError, KeyError):
            pass

    print()
    print(sep)
    print("  TABLE 5 — Mean training time per fold (seconds)")
    print(sep)
    print(hdr)
    print("  " + "-" * (18 + cw * len(datasets) + 6))
    for m in models:
        cells = []
        all_t = []
        for ds in datasets:
            vals = times[m][ds]
            if vals:
                mu = sum(vals) / len(vals)
                cells.append(f"{mu:>{cw}.1f}")
                all_t.append(mu)
            else:
                cells.append(f"{'---':>{cw}}")
        avg = f"{sum(all_t)/len(all_t):.1f}" if all_t else "---"
        print(f"  {m:<18}" + "".join(cells) + f"  {avg:>6}")

    print()
    print(f"  Results: {OUT_CSV}")
    print()


# ── Interpretation helper ─────────────────────────────────────────────────────

def interpret(verbose: bool = True) -> None:
    """Print a short narrative interpretation after all tables."""
    rows = _load_csv()
    if not rows:
        return

    from collections import defaultdict
    r2:    dict = defaultdict(lambda: defaultdict(list))
    r2n05: dict = defaultdict(lambda: defaultdict(list))
    zrho:  dict = defaultdict(lambda: defaultdict(list))

    for r in rows:
        m, ds = r["model"], r["dataset"]
        def _f(k):
            try:
                return float(r[k])
            except (ValueError, KeyError):
                return float("nan")
        r2[m][ds].append(_f("r2"))
        r2n05[m][ds].append(_f("r2_n05"))
        z = _f("z_nn_spearman")
        if z == z:
            zrho[m][ds].append(z)

    datasets = sorted({r["dataset"] for r in rows})

    def _mean_over_ds(d, model):
        vals = [_agg(d[model][ds])[0] for ds in datasets]
        valid = [v for v in vals if v == v]
        return sum(valid) / len(valid) if valid else float("nan")

    geo_mu, _ = _agg([x for ds in datasets for x in zrho["GeoXGB"][ds]])

    mu_geo = _mean_over_ds(r2, "GeoXGB")
    mu_xgb = _mean_over_ds(r2, "XGBoost")
    delta_geo_xgb = mu_geo - mu_xgb if (mu_geo == mu_geo and mu_xgb == mu_xgb) else float("nan")

    mu_geo_n5  = _mean_over_ds(r2n05, "GeoXGB")
    mu_xgb_n5  = _mean_over_ds(r2n05, "XGBoost")
    delta_rob_geo = mu_geo_n5 - mu_geo if (mu_geo_n5 == mu_geo_n5 and mu_geo == mu_geo) else float("nan")
    delta_rob_xgb = mu_xgb_n5 - mu_xgb if (mu_xgb_n5 == mu_xgb_n5 and mu_xgb == mu_xgb) else float("nan")

    print("=" * 68)
    print("  INTERPRETATION")
    print("=" * 68)
    print(f"  Accuracy:    GeoXGB vs XGBoost  Δmean R² = {delta_geo_xgb:+.4f}")
    print(f"  Robustness:  GeoXGB ΔR² at σ=0.5  avg = {delta_rob_geo:+.4f}")
    print(f"               XGBoost ΔR² at σ=0.5 avg = {delta_rob_xgb:+.4f}")
    rob_diff = delta_rob_geo - delta_rob_xgb
    print(f"               Robustness advantage (GeoXGB − XGBoost) = {rob_diff:+.4f}")
    print(f"               {'GeoXGB more robust' if rob_diff > 0 else 'XGBoost more robust'}"
          f" under distribution shift")
    print(f"  Geometry:    mean Spearman(z-NN dist, |error|) = {geo_mu:.3f}")
    print(f"               {'z-space reliably signals prediction uncertainty' if geo_mu > 0.1 else 'weak geometry signal on these datasets'}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GeoXGB vs XGBoost vs LightGBM real-world regression benchmark"
    )
    parser.add_argument("--jobs",    type=int, default=-1,
                        help="Parallel workers (-1 = all CPUs)")
    parser.add_argument("--fast",    action="store_true",
                        help="Quick preview: 300 rounds, 1 seed, 3 folds")
    parser.add_argument("--rounds",  type=int, default=0,
                        help="Override n_rounds for all models (0 = use default)")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary from saved CSV without re-running")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        interpret()
    else:
        run(n_jobs=args.jobs, fast=args.fast, n_rounds=args.rounds)
        interpret()
