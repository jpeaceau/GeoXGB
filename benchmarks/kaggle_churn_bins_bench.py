"""
Benchmark: HVRT partition tree bin count (16 vs 32 vs 64) on Kaggle Churn.

Currently HVRT uses 16 bins for the partition tree. GBT weak learners use 64.
Test if increasing HVRT bins improves partition quality.

Uses best HPO params from hpo3 as baseline.
"""
import sys, time, os, warnings
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_data():
    tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    y = (tr["Churn"] == "Yes").astype(int).values
    X = tr.drop(columns=["id", "Churn"])
    cat_cols = X.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    return X.values.astype(np.float64), y


if __name__ == "__main__":
    from geoxgb import GeoXGBClassifier

    X, y = load_data()
    print(f"Data: n={len(X)}, d={X.shape[1]}, churn_rate={y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    MAX_SAMPLES = 10_000
    X_hpo, _, y_hpo, _ = train_test_split(
        X_train, y_train, train_size=MAX_SAMPLES,
        stratify=y_train, random_state=42
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(skf.split(X_hpo, y_hpo))

    # Best params from hpo3
    best_params = dict(
        n_rounds=5000,
        learning_rate=0.046,
        max_depth=7,
        min_samples_leaf=20,
        reduce_ratio=0.54,
        refit_interval=300,
        expand_ratio=0.475,
        y_weight=0.31,
        hvrt_min_samples_leaf=30,
        noise_guard=False,
        auto_noise=False,
        auto_expand=True,
        generation_strategy="laplace",
        method="orthant_stratified",
        variance_weighted=True,
        convergence_tol=0.01,
        random_state=42,
        sample_block_n=None,  # disable block cycling for clean comparison
    )

    # The HVRT bin count is hardcoded at 16 in geoxgb_base.cpp line ~266.
    # We can't change it via config — but we CAN test different GBT n_bins
    # which is the only bin parameter exposed. For the HVRT partition tree,
    # we'd need a code change. Let's first test GBT bins impact.
    #
    # Actually, let's also test the adaptive_y_weight flag.

    print(f"\n=== Test 1: GBT n_bins (weak learner binning) ===")
    print(f"10k subsample, 3-fold CV, sample_block_n=None")
    print("=" * 60)

    # n_bins is not currently exposed in GeoXGBClassifier constructor.
    # It goes through _cpp_backend. Let's check if it's in _PARAM_NAMES.
    # ... it's not. We need to pass it via the C++ config directly.

    # Direct C++ approach for bin testing:
    from geoxgb._cpp_backend import make_cpp_config, CppGeoXGBClassifier
    from geoxgb._base import _resolve_auto_block

    for n_bins in [16, 32, 64, 128, 256]:
        scores = []
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr_idx, va_idx in splits:
                cpp_params = dict(best_params)
                cpp_params.pop("class_weight", None)
                cpp_params["n_bins"] = n_bins
                cfg = make_cpp_config(**cpp_params)
                m = CppGeoXGBClassifier(cfg)
                m.fit(X_hpo[tr_idx], y_hpo[tr_idx].astype(float))
                raw = m.predict_proba(X_hpo[va_idx])
                proba = raw[:, 1]
                scores.append(roc_auc_score(y_hpo[va_idx], proba))
        elapsed = time.perf_counter() - t0
        mean_auc = float(np.mean(scores))
        std_auc = float(np.std(scores))
        print(f"  n_bins={n_bins:3d}  CV AUC={mean_auc:.5f} +/- {std_auc:.5f}  ({elapsed:.1f}s)")

    print(f"\n=== Test 2: adaptive_y_weight on/off ===")
    print("=" * 60)

    for adaptive in [True, False]:
        scores = []
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr_idx, va_idx in splits:
                cpp_params = dict(best_params)
                cpp_params.pop("class_weight", None)
                cpp_params["adaptive_y_weight"] = adaptive
                cfg = make_cpp_config(**cpp_params)
                m = CppGeoXGBClassifier(cfg)
                m.fit(X_hpo[tr_idx], y_hpo[tr_idx].astype(float))
                raw = m.predict_proba(X_hpo[va_idx])
                proba = raw[:, 1]
                scores.append(roc_auc_score(y_hpo[va_idx], proba))
        elapsed = time.perf_counter() - t0
        mean_auc = float(np.mean(scores))
        std_auc = float(np.std(scores))
        print(f"  adaptive_y_weight={str(adaptive):5s}  CV AUC={mean_auc:.5f} +/- {std_auc:.5f}  ({elapsed:.1f}s)")

    # Also test on the standard benchmarks (diabetes, friedman1)
    print(f"\n=== Test 3: adaptive_y_weight on standard datasets ===")
    print("=" * 60)

    from sklearn.datasets import load_diabetes, make_friedman1
    from geoxgb._cpp_backend import CppGeoXGBRegressor as _CppReg
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    for name, get_data in [
        ("diabetes", lambda: load_diabetes(return_X_y=True)),
        ("friedman1", lambda: make_friedman1(n_samples=1000, noise=1.0, random_state=42)),
    ]:
        X_d, y_d = get_data()
        X_d = np.asarray(X_d, dtype=np.float64)
        y_d = np.asarray(y_d, dtype=np.float64)
        for adaptive in [True, False]:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            for tr_i, va_i in kf.split(X_d):
                cfg = make_cpp_config(
                    n_rounds=500, learning_rate=0.02, max_depth=3,
                    refit_interval=50, random_state=42,
                    adaptive_y_weight=adaptive, sample_block_n=-1,
                )
                m = _CppReg(cfg)
                m.fit(X_d[tr_i], y_d[tr_i])
                p = m.predict(X_d[va_i])
                cv_scores.append(r2_score(y_d[va_i], p))
            print(f"  {name:12s}  adaptive={str(adaptive):5s}  R2={np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
