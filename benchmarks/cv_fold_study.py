"""
cv=1 vs cv=3 HPO ranking study
================================
Tests whether a single hold-out split (cv=1) ranks Optuna trials identically
to 3-fold CV across multiple datasets and tasks.

If Kendall-tau rank correlation is >= 0.9 consistently, cv=1 is sufficient
and reduces optimizer cost by 3x.

Methodology
-----------
1. For each dataset, run one Optuna study with cv=3, recording per-trial score.
2. Re-score the SAME parameter sets with a single 80/20 hold-out (independent
   split). Both cv=3 and cv=1 use IDENTICAL parameter draws from the same study.
3. Compute Kendall-tau and Spearman rank correlation between the two score vectors.
4. Report top-k agreement: do cv=1 and cv=3 agree on the best-3 params?

NOTE: convergence_tol is intentionally DISABLED for this study.
convergence_tol stops training based on training-set loss, which converges at
different rounds depending on training set size (cv=3 uses 2/3 of data, cv=1
uses 80%). Including it would confound splitting strategy with effective n_rounds,
biasing the rank correlation. We want to isolate: does cv=1 rank trials the same?

Datasets (regression + classification mix):
  diabetes         -- n=442,  d=10, regression  (sklearn)
  california       -- n=20640, d=8, regression  (sklearn, subsample 4000)
  friedman1        -- n=2000,  d=10, regression  (sklearn synthetic)
  breast_cancer    -- n=569,  d=30, classification
  wine             -- n=178,  d=13, multiclass

Usage
-----
    python benchmarks/cv_fold_study.py [--n-trials N] [--seed S]

Takes ~5-15 min with default n_trials=30 (convergence_tol disabled = full rounds).
"""

import argparse
import sys
import time
import warnings
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.datasets import (
    load_diabetes, load_breast_cancer, load_wine,
    fetch_california_housing, make_friedman1
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, roc_auc_score

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from geoxgb import GeoXGBRegressor, GeoXGBClassifier


def _p(*a, **k):
    print(*a, **k, flush=True)


# --- search space (same as GeoXGBOptimizer) ----------------------------------
_SPACE_REG = {
    "n_rounds":       [500, 1000, 1500, 2000],
    "learning_rate":  [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05],
    "max_depth":      [2, 3, 4, 5],
    "reduce_ratio":   [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "refit_interval": [10, 20, 50, 100, 200],
    "y_weight":       [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
}

_SPACE_CLF = {
    "n_rounds":       [500, 1000, 1500, 2000],
    "learning_rate":  [0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05],
    "max_depth":      [2, 3, 4, 5],
    "reduce_ratio":   [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "refit_interval": [10, 20, 50, 100, 200],
    "y_weight":       [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "class_weight":   [None, "balanced"],
}

# convergence_tol intentionally excluded (see module docstring for rationale)
_TRIAL_FIXED = {"random_state": 42}

_DEFAULTS_REG = {
    "n_rounds": 1000, "learning_rate": 0.02, "max_depth": 3,
    "reduce_ratio": 0.8, "refit_interval": 50, "y_weight": 0.25,
}
_DEFAULTS_CLF = {**_DEFAULTS_REG, "class_weight": None}


def r2_scorer(m, X, y):
    return float(r2_score(y, m.predict(X)))


def auc_scorer(m, X, y):
    p = m.predict_proba(X)
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))


# --- study runner -------------------------------------------------------------

def run_study(X, y, task, n_trials, seed):
    """
    Run an Optuna study with cv=3 and simultaneously score every trial's
    parameters on an independent single 80/20 hold-out (cv=1).

    Returns (study, cv3_scores, cv1_scores) -- all three arrays have
    length n_trials, one entry per trial in study completion order.
    """
    space       = _SPACE_CLF if task == "classification" else _SPACE_REG
    defaults    = _DEFAULTS_CLF if task == "classification" else _DEFAULTS_REG
    model_cls   = GeoXGBClassifier if task == "classification" else GeoXGBRegressor
    scorer      = auc_scorer if task == "classification" else r2_scorer

    # Fixed cv=3 splits
    if task == "classification":
        kf     = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        splits = list(kf.split(X, y))
    else:
        kf     = KFold(n_splits=3, shuffle=True, random_state=seed)
        splits = list(kf.split(X))

    # Fixed cv=1 hold-out -- independent of cv=3 splits, different random seed
    if task == "classification":
        X_tr1, X_val1, y_tr1, y_val1 = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed + 99)
    else:
        X_tr1, X_val1, y_tr1, y_val1 = train_test_split(
            X, y, test_size=0.2, random_state=seed + 99)

    all_cv3 = []   # mean 3-fold score per trial
    all_cv1 = []   # single hold-out score per trial (same params)

    def objective(trial):
        params = {
            name: trial.suggest_categorical(name, choices)
            for name, choices in space.items()
        }
        run = {**params, **_TRIAL_FIXED}
        fold_scores = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr, val in splits:
                m = model_cls(**run)
                m.fit(X[tr], y[tr])
                fold_scores.append(scorer(m, X[val], y[val]))
            cv3_score = float(np.mean(fold_scores))
            all_cv3.append(cv3_score)

            # cv=1: same params, independent split
            m1 = model_cls(**run)
            m1.fit(X_tr1, y_tr1)
            cv1_score = scorer(m1, X_val1, y_val1)
            all_cv1.append(cv1_score)

        return cv3_score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({name: defaults[name] for name in space})
    study.optimize(objective, n_trials=n_trials)

    return study, np.array(all_cv3), np.array(all_cv1)


# --- analysis ----------------------------------------------------------------

def analyse(name, cv3_scores, cv1_scores, top_k=3):
    tau, p_tau = kendalltau(cv3_scores, cv1_scores)
    rho, p_rho = spearmanr(cv3_scores, cv1_scores)

    n = len(cv3_scores)
    rank3 = np.argsort(-cv3_scores)[:top_k]
    rank1 = np.argsort(-cv1_scores)[:top_k]
    overlap = len(set(rank3) & set(rank1))

    best3 = int(np.argmax(cv3_scores))
    best1 = int(np.argmax(cv1_scores))
    best_agree = (best3 == best1)

    # Where does cv3's best trial rank in cv1's ordering?
    order1 = np.argsort(-cv1_scores)
    cv3_best_rank_in_cv1 = int(np.where(order1 == best3)[0][0]) + 1

    return {
        "dataset":                name,
        "n_trials":               n,
        "kendall_tau":            float(tau),
        "p_tau":                  float(p_tau),
        "spearman_r":             float(rho),
        "p_rho":                  float(p_rho),
        "top3_overlap":           overlap,
        "best1_agree":            best_agree,
        "cv3_best_score":         float(cv3_scores[best3]),
        "cv1_rank_of_cv3_best":   cv3_best_rank_in_cv1,
    }


# --- datasets ----------------------------------------------------------------

def load_datasets(seed):
    ds = []

    d = load_diabetes()
    ds.append(("diabetes", d.data, d.target, "regression"))

    X_ca_full, y_ca = fetch_california_housing(return_X_y=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_ca_full), size=4000, replace=False)
    ds.append(("california_4k", X_ca_full[idx], y_ca[idx], "regression"))

    X_f, y_f = make_friedman1(n_samples=2000, noise=1.0, random_state=seed)
    ds.append(("friedman1", X_f, y_f, "regression"))

    bc = load_breast_cancer()
    ds.append(("breast_cancer", bc.data, bc.target, "classification"))

    w = load_wine()
    ds.append(("wine", w.data, w.target, "classification"))

    return ds


# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials",   type=int, default=30)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--partitioner", type=str, default="pyramid_hart",
                    choices=["pyramid_hart", "hvrt", "hart", "fasthvrt"],
                    help="GeoXGB partitioner to use across all trials")
    ap.add_argument("--min-n", type=int, default=0,
                    help="Skip datasets with fewer than this many samples")
    args = ap.parse_args()

    # Inject partitioner into the fixed trial params so all models use it
    _TRIAL_FIXED["partitioner"] = args.partitioner

    datasets = [(n, X, y, t) for n, X, y, t in load_datasets(args.seed)
                if len(X) >= args.min_n]

    _p(f"\n{'='*72}")
    min_n_str = f", min_n={args.min_n}" if args.min_n > 0 else ""
    _p(f"cv=1 vs cv=3 HPO ranking study  --  {args.n_trials} trials, "
       f"seed={args.seed}, partitioner={args.partitioner}{min_n_str}")
    _p(f"NOTE: convergence_tol disabled to avoid training-set-size confound.")
    _p(f"{'='*72}")
    _p(f"  {'Dataset':<20}  {'tau':>6}  {'rho':>6}  {'top3':>5}  {'best1':>6}  "
       f"{'cv3-best rank in cv1':>22}")
    _p(f"  {'-'*72}")

    results = []
    for name, X, y, task in datasets:
        t0 = time.time()
        _p(f"  Running {name} ({task}, n={len(X)}, d={X.shape[1]}) ...", end=" ")
        _, cv3, cv1 = run_study(X, y, task, args.n_trials, args.seed)
        elapsed = time.time() - t0
        r = analyse(name, cv3, cv1)
        results.append(r)
        agree_str = "yes" if r["best1_agree"] else "NO"
        _p(f"done ({elapsed:.0f}s)")
        _p(f"  {'':20}  tau={r['kendall_tau']:+.3f}  rho={r['spearman_r']:+.3f}  "
           f"top3={r['top3_overlap']}/3  best1={agree_str}  "
           f"cv3-best is rank-{r['cv1_rank_of_cv3_best']:d} in cv1")

    _p(f"\n{'='*72}")
    _p("SUMMARY")
    _p(f"{'='*72}")
    _p(f"  {'Dataset':<20}  {'Kendall-tau':>12}  {'Spearman-rho':>13}  {'top3/3':>7}  {'best1':>6}")
    _p(f"  {'-'*65}")
    tau_all = []
    rho_all = []
    for r in results:
        agree_str = "yes" if r["best1_agree"] else "NO"
        _p(f"  {r['dataset']:<20}  {r['kendall_tau']:>+12.3f}  {r['spearman_r']:>+13.3f}  "
           f"{r['top3_overlap']:>5}/3  {agree_str:>6}")
        tau_all.append(r["kendall_tau"])
        rho_all.append(r["spearman_r"])

    _p(f"  {'-'*65}")
    _p(f"  {'Mean':<20}  {np.mean(tau_all):>+12.3f}  {np.mean(rho_all):>+13.3f}")

    _p(f"\nInterpretation:")
    mean_tau = np.mean(tau_all)
    if mean_tau >= 0.85:
        _p(f"  Mean tau={mean_tau:.3f} >= 0.85 -- cv=1 is a valid substitute for cv=3.")
        _p(f"  Optimizer default cv can safely be reduced to 1, cutting trial cost by 3x.")
    elif mean_tau >= 0.70:
        _p(f"  Mean tau={mean_tau:.3f} -- reasonable but imperfect; cv=1 finds near-best params.")
        _p(f"  Conservative choice: keep cv=2 (half cost) or cv=1 with more trials.")
    else:
        _p(f"  Mean tau={mean_tau:.3f} < 0.70 -- cv=3 (or higher) meaningfully improves ranking.")
        _p(f"  Keep cv=3 as default.")

    _p(f"\nDone.")
    return results


if __name__ == "__main__":
    main()
