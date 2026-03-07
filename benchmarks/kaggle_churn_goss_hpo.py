"""
Kaggle Churn — GOSS HPO with multiple optimizer strategies.

Searches over GOSS parameters (goss_alpha, goss_beta) alongside core GeoXGB
hyperparameters.  Uses multiprocessing to parallelise trial evaluation on the
full 475k training set (block cycling active).

Three Optuna sampler phases:
  1. TPE (main exploration + exploitation)
  2. CMA-ES (continuous param refinement)
  3. Random (diversity injection)

PCA(19) rotation, y_weight=0, convergence_tol=0.01.
"""
import sys, time, os, warnings, multiprocessing
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_data():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    y = (tr["Churn"] == "Yes").astype(int).values
    X = tr.drop(columns=["id", "Churn"])
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    return X.values.astype(np.float64), y


def objective(trial, X, y, splits):
    from geoxgb import GeoXGBClassifier

    # GOSS parameters
    goss_alpha = trial.suggest_float("goss_alpha", 0.05, 0.5)
    goss_beta  = trial.suggest_float("goss_beta", 0.05, 0.4)

    params = {
        "n_rounds":       trial.suggest_int("n_rounds", 2000, 8000, step=500),
        "learning_rate":  trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        "max_depth":      trial.suggest_int("max_depth", 3, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "reduce_ratio":   trial.suggest_float("reduce_ratio", 0.2, 0.8),
        "refit_interval": trial.suggest_categorical("refit_interval",
                              [100, 200, 300, 500]),
        "expand_ratio":   trial.suggest_float("expand_ratio", 0.0, 0.5),
        "y_weight":       0.0,
        "hvrt_min_samples_leaf": trial.suggest_categorical(
            "hvrt_min_samples_leaf", [None, 10, 30, 50, 100]),
        "noise_guard":    trial.suggest_categorical("noise_guard", [True, False]),
        "auto_noise":     True,
        "auto_expand":    True,
        "generation_strategy": trial.suggest_categorical(
            "generation_strategy",
            ["epanechnikov", "simplex_mixup", "bootstrap"]),
        "method":         "variance_ordered",
        "partitioner":    trial.suggest_categorical(
            "partitioner", ["hvrt", "pyramid_hart"]),
        "n_bins":         trial.suggest_categorical("n_bins", [64, 128]),
        "goss_alpha":     goss_alpha,
        "goss_beta":      goss_beta,
        "convergence_tol": 0.01,
        "random_state":   42,
        "sample_block_n": "auto",
    }

    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr_idx, va_idx in splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_va = scaler.transform(X[va_idx])
            pca = PCA(n_components=19, random_state=42)
            X_tr_pca = pca.fit_transform(X_tr)
            X_va_pca = pca.transform(X_va)

            m = GeoXGBClassifier(**params)
            m.fit(X_tr_pca, y[tr_idx].astype(float))
            proba = m.predict_proba(X_va_pca)[:, 1]
            scores.append(roc_auc_score(y[va_idx], proba))
    return float(np.mean(scores))


def run_study_phase(phase_name, sampler, study_name, storage, n_trials, X, y, splits):
    """Run a phase of HPO with a specific sampler."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.load_study(study_name=study_name, storage=storage,
                              sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X, y, splits),
        n_trials=n_trials,
    )
    return study.best_value


def worker_objective(args):
    """Multiprocessing worker for parallel CV folds."""
    trial_params, fold_data = args
    from geoxgb import GeoXGBClassifier
    X_tr, X_va, y_tr, y_va = fold_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = GeoXGBClassifier(**trial_params)
        m.fit(X_tr, y_tr.astype(float))
        proba = m.predict_proba(X_va)[:, 1]
        return roc_auc_score(y_va, proba)


if __name__ == "__main__":
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    X, y = load_data()
    print(f"Data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # CV splits on full training data
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(skf.split(X_train, y_train))

    # Warm-start seeds: best known params + GOSS variants
    warm_starts = [
        # Best from 500-trial HPO + GOSS 20/20
        {"n_rounds": 7000, "learning_rate": 0.00797, "max_depth": 6,
         "min_samples_leaf": 11, "reduce_ratio": 0.302, "refit_interval": 500,
         "expand_ratio": 0.294, "hvrt_min_samples_leaf": 100,
         "noise_guard": False, "generation_strategy": "epanechnikov",
         "partitioner": "pyramid_hart", "n_bins": 128,
         "goss_alpha": 0.2, "goss_beta": 0.2},
        # Same but GOSS 30/10
        {"n_rounds": 7000, "learning_rate": 0.00797, "max_depth": 6,
         "min_samples_leaf": 11, "reduce_ratio": 0.302, "refit_interval": 500,
         "expand_ratio": 0.294, "hvrt_min_samples_leaf": 100,
         "noise_guard": False, "generation_strategy": "epanechnikov",
         "partitioner": "pyramid_hart", "n_bins": 128,
         "goss_alpha": 0.3, "goss_beta": 0.1},
        # Best from ri=200 (fastest config)
        {"n_rounds": 7000, "learning_rate": 0.00797, "max_depth": 6,
         "min_samples_leaf": 11, "reduce_ratio": 0.302, "refit_interval": 200,
         "expand_ratio": 0.294, "hvrt_min_samples_leaf": 100,
         "noise_guard": False, "generation_strategy": "epanechnikov",
         "partitioner": "pyramid_hart", "n_bins": 128,
         "goss_alpha": 0.2, "goss_beta": 0.2},
        # More aggressive GOSS
        {"n_rounds": 5000, "learning_rate": 0.015, "max_depth": 6,
         "min_samples_leaf": 11, "reduce_ratio": 0.4, "refit_interval": 300,
         "expand_ratio": 0.2, "hvrt_min_samples_leaf": 50,
         "noise_guard": False, "generation_strategy": "bootstrap",
         "partitioner": "pyramid_hart", "n_bins": 128,
         "goss_alpha": 0.4, "goss_beta": 0.2},
        # HVRT partitioner + lower GOSS
        {"n_rounds": 6000, "learning_rate": 0.01, "max_depth": 5,
         "min_samples_leaf": 15, "reduce_ratio": 0.5, "refit_interval": 300,
         "expand_ratio": 0.15, "hvrt_min_samples_leaf": 30,
         "noise_guard": False, "generation_strategy": "simplex_mixup",
         "partitioner": "hvrt", "n_bins": 64,
         "goss_alpha": 0.15, "goss_beta": 0.15},
    ]

    # ── Phase 1: TPE (60% of budget) ─────────────────────────────────────────
    n_tpe   = int(N_TRIALS * 0.60)
    n_cmaes = int(N_TRIALS * 0.25)
    n_rand  = N_TRIALS - n_tpe - n_cmaes

    storage = f"sqlite:///churn_goss_hpo.db"
    study_name = "churn_goss"

    # Clean start
    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except Exception:
        pass

    # Create study with TPE sampler
    tpe_sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=20)
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        direction="maximize", sampler=tpe_sampler
    )
    for ws in warm_starts:
        study.enqueue_trial(ws)

    print(f"\n{'='*70}")
    print(f"GOSS HPO: {N_TRIALS} trials on FULL train ({len(X_train)} samples)")
    print(f"  Phase 1: TPE        ({n_tpe} trials)")
    print(f"  Phase 2: CMA-ES     ({n_cmaes} trials)")
    print(f"  Phase 3: Random     ({n_rand} trials)")
    print(f"PCA(19), y_weight=0, convergence_tol=0.01, 3-fold CV")
    print(f"{'='*70}")

    t0_total = time.perf_counter()

    # Phase 1: TPE
    print(f"\n--- Phase 1: TPE ({n_tpe} trials) ---")
    t0 = time.perf_counter()
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, splits),
        n_trials=n_tpe,
    )
    t_tpe = time.perf_counter() - t0
    print(f"  TPE done in {t_tpe:.0f}s ({t_tpe/n_tpe:.1f}s/trial), "
          f"best so far: {study.best_value:.5f}")

    # Phase 2: CMA-ES (continuous refinement)
    print(f"\n--- Phase 2: CMA-ES ({n_cmaes} trials) ---")
    cma_sampler = optuna.samplers.CmaEsSampler(seed=123)
    study.sampler = cma_sampler
    t0 = time.perf_counter()
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, splits),
        n_trials=n_cmaes,
    )
    t_cma = time.perf_counter() - t0
    print(f"  CMA-ES done in {t_cma:.0f}s ({t_cma/n_cmaes:.1f}s/trial), "
          f"best so far: {study.best_value:.5f}")

    # Phase 3: Random (diversity)
    print(f"\n--- Phase 3: Random ({n_rand} trials) ---")
    rand_sampler = optuna.samplers.RandomSampler(seed=456)
    study.sampler = rand_sampler
    t0 = time.perf_counter()
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, splits),
        n_trials=n_rand,
    )
    t_rand = time.perf_counter() - t0
    print(f"  Random done in {t_rand:.0f}s ({t_rand/n_rand:.1f}s/trial), "
          f"best so far: {study.best_value:.5f}")

    t_total = time.perf_counter() - t0_total

    print(f"\n{'='*70}")
    print(f"HPO complete: {t_total:.0f}s total ({t_total/N_TRIALS:.1f}s/trial)")
    print(f"Best CV AUC: {study.best_value:.5f}")
    print(f"Best params: {study.best_params}")

    # Top 10
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]
    df = df.sort_values("value", ascending=False).drop_duplicates(subset="value").head(10)
    print(f"\nTop 10 distinct trials:")
    for _, row in df.iterrows():
        p = {k.replace("params_", ""): row[k]
             for k in row.index if k.startswith("params_")}
        print(f"  AUC={row['value']:.5f} | "
              f"lr={p.get('learning_rate',0):.4f} "
              f"d={p.get('max_depth',0)} "
              f"msl={p.get('min_samples_leaf',0)} "
              f"ri={p.get('refit_interval',0)} "
              f"nr={p.get('n_rounds',0)} "
              f"rr={p.get('reduce_ratio',0):.2f} "
              f"er={p.get('expand_ratio',0):.2f} "
              f"ga={p.get('goss_alpha',0):.2f} "
              f"gb={p.get('goss_beta',0):.2f} "
              f"hmsl={p.get('hvrt_min_samples_leaf',None)} "
              f"ng={p.get('noise_guard','')} "
              f"gs={p.get('generation_strategy','')} "
              f"pt={p.get('partitioner','')} "
              f"nb={p.get('n_bins','')}")

    # ── Full train/test evaluation ──────────────────────────────────────
    from geoxgb import GeoXGBClassifier

    best = dict(study.best_params)
    best["y_weight"] = 0.0
    best["convergence_tol"] = 0.01
    best["random_state"] = 42
    best["sample_block_n"] = "auto"
    best["auto_noise"] = True
    best["auto_expand"] = True
    best["method"] = "variance_ordered"

    # PCA on full train/test
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    pca = PCA(n_components=19, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    # Fit on full train
    print(f"\nRefitting best on FULL train ({len(X_train)} samples)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        m_full = GeoXGBClassifier(**best)
        m_full.fit(X_train_pca, y_train.astype(float))
        t_full = time.perf_counter() - t0
    train_auc = roc_auc_score(y_train, m_full.predict_proba(X_train_pca)[:, 1])
    test_auc = roc_auc_score(y_test, m_full.predict_proba(X_test_pca)[:, 1])
    cr = m_full.convergence_round_
    print(f"Test AUC: {test_auc:.5f} | Train AUC: {train_auc:.5f} | "
          f"Gap: {train_auc - test_auc:+.4f} | "
          f"conv@{cr} | {t_full:.1f}s")

    # Also test without GOSS for comparison
    best_no_goss = {k: v for k, v in best.items()
                    if k not in ("goss_alpha", "goss_beta")}
    print(f"\nRefitting best WITHOUT GOSS...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        m_ng = GeoXGBClassifier(**best_no_goss)
        m_ng.fit(X_train_pca, y_train.astype(float))
        t_ng = time.perf_counter() - t0
    train_auc_ng = roc_auc_score(y_train, m_ng.predict_proba(X_train_pca)[:, 1])
    test_auc_ng = roc_auc_score(y_test, m_ng.predict_proba(X_test_pca)[:, 1])
    cr_ng = m_ng.convergence_round_
    print(f"Test AUC: {test_auc_ng:.5f} | Train AUC: {train_auc_ng:.5f} | "
          f"Gap: {train_auc_ng - test_auc_ng:+.4f} | "
          f"conv@{cr_ng} | {t_ng:.1f}s")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Best GOSS HPO test AUC:       {test_auc:.5f} ({t_full:.1f}s)")
    print(f"  Same params without GOSS:     {test_auc_ng:.5f} ({t_ng:.1f}s)")
    print(f"  GOSS speedup:                 {t_ng/t_full:.1f}x")
    print(f"  Prior best (no GOSS):         0.91206")
    print(f"  XGBoost+PCA full train:       0.91076")
    print(f"  XGBoost raw full train:       0.91610")
