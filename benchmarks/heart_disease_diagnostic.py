"""
GeoXGB -- Heart Disease: Regressor vs Classifier Diagnostic
============================================================

Investigates why GeoXGBRegressor outperforms GeoXGBClassifier on the
Heart Disease dataset (n=270) even at higher round counts.

Compares two conditions to isolate the confound:
  A) auto_expand=True  (default -- 216 real -> ~5000 synthetic samples)
  B) auto_expand=False (no synthetic expansion -- 151 real samples only)

This separates the gradient-compression effect from the synthetic-target
reconstruction effect, which differ between reg and clf on synthetic samples.

Also checks clf_noise trajectory to diagnose gradient compression severity.

Usage
-----
    python benchmarks/heart_disease_diagnostic.py
"""

from __future__ import annotations

import multiprocessing
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RANDOM_STATE      = 42
LEARNING_RATE     = 0.2
REFIT_INTERVAL    = 20
MAX_DEPTH         = 4
N_FOLDS           = 5
ROUND_CHECKPOINTS = [100, 300, 500, 1000]

GEO_FIXED = dict(
    learning_rate    = LEARNING_RATE,
    refit_interval   = REFIT_INTERVAL,
    max_depth        = MAX_DEPTH,
    min_samples_leaf = 5,
    reduce_ratio     = 0.7,
    auto_noise       = True,
    cache_geometry   = False,
    random_state     = RANDOM_STATE,
)

_SEP  = "=" * 72
_SEP2 = "-" * 72


def _section(title):
    print(f"\n{_SEP}\n  {title}\n{_SEP}")


def _subsection(title):
    print(f"\n{_SEP2}\n  {title}\n{_SEP2}")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(model_type, n_rounds, auto_expand, X, y, tr, te):
    import warnings as _w; _w.filterwarnings("ignore")
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from geoxgb import GeoXGBClassifier, GeoXGBRegressor

    params = dict(**GEO_FIXED, n_rounds=n_rounds, auto_expand=auto_expand)

    if model_type == "reg":
        m = GeoXGBRegressor(**params)
        m.fit(X[tr], y[tr].astype(float))
        scores = m.predict(X[te])
    else:
        m = GeoXGBClassifier(**params)
        m.fit(X[tr], y[tr])
        scores = m.predict_proba(X[te])[:, 1]

    auc        = float(roc_auc_score(y[te], scores))
    mean_noise = float(np.mean([e["noise_modulation"] for e in m._resample_history]))
    n_train    = m._train_n_resampled
    return model_type, n_rounds, auto_expand, auc, mean_noise, n_train


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cpu_count = multiprocessing.cpu_count()

    df = pd.read_csv("data/Heart_Disease_Prediction.csv")
    X  = df.drop(columns=["Heart Disease"]).values.astype(np.float64)
    y  = LabelEncoder().fit_transform(df["Heart Disease"])

    _section("GeoXGB -- Heart Disease: Regressor vs Classifier Diagnostic")
    counts = np.bincount(y)
    print(
        f"\n  Dataset       : Heart Disease (UCI)"
        f"\n  Samples       : {len(y)}  (train fold ~{int(len(y)*0.8)})"
        f"\n  Features      : {X.shape[1]}"
        f"\n  Class balance : Absence={counts[0]}  Presence={counts[1]}"
        f"\n  Conditions    : auto_expand=True (default) vs auto_expand=False"
        f"\n  Checkpoints   : {ROUND_CHECKPOINTS}"
        f"\n  CV folds      : {N_FOLDS}"
    )

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))

    jobs     = []
    job_keys = []
    for auto_expand in (True, False):
        for n_rounds in ROUND_CHECKPOINTS:
            for model_type in ("reg", "clf"):
                for fi, (tr, te) in enumerate(splits):
                    jobs.append((model_type, n_rounds, auto_expand, X, y, tr, te))
                    job_keys.append((auto_expand, model_type, n_rounds, fi))

    n_jobs_total = len(jobs)
    print(f"\n  Launching {n_jobs_total} jobs across {cpu_count} workers...")

    import time
    t0  = time.perf_counter()
    raw = Parallel(n_jobs=cpu_count, prefer="processes", verbose=1)(
        delayed(_worker)(*j) for j in jobs
    )
    wall = time.perf_counter() - t0
    print(f"\n  Done. Wall time: {wall:.1f}s")

    # Aggregate
    scores  = defaultdict(list)
    noises  = defaultdict(list)
    n_train = defaultdict(list)

    for (ae, mt, nr, _fi), (_, _, _, auc, noise, ntrain) in zip(job_keys, raw):
        scores[(ae, mt, nr)].append(auc)
        noises[(ae, mt, nr)].append(noise)
        n_train[(ae, mt, nr)].append(ntrain)

    # -----------------------------------------------------------------------
    # Per-condition tables
    # -----------------------------------------------------------------------
    for auto_expand in (True, False):
        label = "WITH auto_expand (default)" if auto_expand else "WITHOUT auto_expand"
        _section(f"AUC Comparison -- {label}")

        ntrain_ex = int(np.mean(n_train[(auto_expand, "reg", ROUND_CHECKPOINTS[0])]))
        print(f"\n  Avg training set size per round: ~{ntrain_ex} samples")
        print(
            f"\n  {'rounds':>7s}  {'reg AUC':>9s}  {'clf AUC':>9s}  "
            f"{'delta':>8s}  {'reg noise':>9s}  {'clf noise':>9s}  winner"
        )
        print(
            f"  {'-'*7}  {'-'*9}  {'-'*9}  "
            f"{'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}"
        )
        for nr in ROUND_CHECKPOINTS:
            ra = float(np.mean(scores[(auto_expand, "reg", nr)]))
            ca = float(np.mean(scores[(auto_expand, "clf", nr)]))
            delta  = ra - ca
            rn = float(np.mean(noises[(auto_expand, "reg", nr)]))
            cn = float(np.mean(noises[(auto_expand, "clf", nr)]))
            winner = "reg  <--" if delta > 0.0005 else (
                     "clf  <--" if delta < -0.0005 else "tied")
            print(
                f"  {nr:>7d}  {ra:>9.5f}  {ca:>9.5f}  "
                f"  {delta:>+7.5f}  {rn:>9.4f}  {cn:>9.4f}  {winner}"
            )

    # -----------------------------------------------------------------------
    # Cross-condition comparison at n_rounds=1000
    # -----------------------------------------------------------------------
    _section("Summary at n_rounds=1000")
    print(
        f"\n  {'condition':<30s}  {'reg':>9s}  {'clf':>9s}  "
        f"{'delta':>8s}  winner"
    )
    print(
        f"  {'-'*30}  {'-'*9}  {'-'*9}  "
        f"{'-'*8}  {'-'*8}"
    )
    for ae, label in [(True, "auto_expand=True (default)"), (False, "auto_expand=False")]:
        ra = float(np.mean(scores[(ae, "reg", 1000)]))
        ca = float(np.mean(scores[(ae, "clf", 1000)]))
        delta  = ra - ca
        winner = "reg" if delta > 0.0005 else ("clf" if delta < -0.0005 else "tied")
        print(
            f"  {label:<30s}  {ra:>9.5f}  {ca:>9.5f}  "
            f"  {delta:>+7.5f}  {winner}"
        )

    # -----------------------------------------------------------------------
    # Noise trajectory (auto_expand=True only)
    # -----------------------------------------------------------------------
    _section("Noise Modulation Trajectory  (auto_expand=True)")
    print(
        f"\n  {'rounds':>7s}  {'reg noise':>10s}  {'clf noise':>10s}  "
        f"{'clf - reg':>10s}  note"
    )
    print(f"  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*30}")
    for nr in ROUND_CHECKPOINTS:
        rn = float(np.mean(noises[(True, "reg", nr)]))
        cn = float(np.mean(noises[(True, "clf", nr)]))
        diff = cn - rn
        note = ("clf sees weaker signal" if diff < -0.05 else
                ("clf sees stronger signal" if diff > 0.05 else "comparable"))
        print(f"  {nr:>7d}  {rn:>10.4f}  {cn:>10.4f}  {diff:>+10.4f}  {note}")

    print(
        f"\n  Key question: if auto_expand=False narrows or reverses the gap,"
        f"\n  the cause is synthetic-target reconstruction differences, not"
        f"\n  purely gradient compression.  If the gap persists without"
        f"\n  expansion, gradient compression is the primary driver."
        f"\n"
    )


if __name__ == "__main__":
    main()
