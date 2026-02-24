"""
Tree Splitter Dimensionality Benchmark
=======================================

Tests whether the benefit of tree_splitter='random' vs 'best' degrades
with feature dimensionality, and identifies the crossover point for an
'auto' mode.

Hypothesis: random splits have probability (n_informative / n_features) of
landing on a relevant feature per node. As d grows with fixed n_informative,
this probability falls. Below some threshold dimension, random splits act as
beneficial regularisation (lower inter-tree correlation, faster fitting).
Above it, random splits are mostly noise and optimal search is needed.

Two sweeps:
  Regression:     make_friedman1(n_features=d) — always 5 informative
  Classification: make_classification(n_features=d, n_informative=6) — fixed 6 informative

d values: 5, 8, 10, 15, 20, 25, 30, 40
n=1000, 5 seeds, 500 rounds (halved for speed; ratio effects dominate).
"""
import warnings, time
import numpy as np
from sklearn.datasets import make_classification, make_friedman1
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import geoxgb._base as _base_mod

from geoxgb import GeoXGBClassifier, GeoXGBRegressor

warnings.filterwarnings("ignore")

SEEDS  = [0, 1, 2, 3, 4]
N      = 1000
ROUNDS = 500
D_VALUES = [5, 8, 10, 15, 20, 25, 30, 40]
COL_W  = 7


def _run(task, d, splitter, seed):
    _orig = DecisionTreeRegressor
    # Patch splitter into DecisionTreeRegressor via _base_mod
    _orig_dtr = _base_mod.DecisionTreeRegressor
    class _PatchedDTR(_orig_dtr):
        def __init__(self, **kw):
            kw['splitter'] = splitter
            super().__init__(**kw)
    _base_mod.DecisionTreeRegressor = _PatchedDTR

    try:
        if task == 'reg':
            n_feat = max(d, 5)   # friedman1 requires >= 5
            X, y = make_friedman1(n_samples=N, n_features=n_feat, noise=1.0,
                                  random_state=seed)
        else:
            n_info = min(6, d - 1)
            n_red  = min(2, d - n_info - 1)
            X, y = make_classification(n_samples=N, n_features=d,
                                       n_informative=n_info, n_redundant=n_red,
                                       random_state=seed)
        X = X.astype(float)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2,
            stratify=y if task == 'clf' else None,
            random_state=seed,
        )
        cls = GeoXGBClassifier if task == 'clf' else GeoXGBRegressor
        m = cls(n_rounds=ROUNDS, random_state=seed)
        m.fit(Xtr, ytr)
        if task == 'clf':
            return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        return r2_score(yte, m.predict(Xte))
    finally:
        _base_mod.DecisionTreeRegressor = _orig_dtr


for task, label, metric in [('reg', 'Regression (Friedman #1)', 'R2 '),
                              ('clf', 'Classification (6 informative)', 'AUC')]:
    print(f"{'='*70}")
    print(f"  {label}  [{metric}]")
    print(f"{'='*70}")

    header = f"  {'Splitter':<10}"
    for d in D_VALUES:
        header += f"  {'d='+str(d):>{COL_W}}"
    print(header)
    print("  " + "-" * (10 + len(D_VALUES) * (COL_W + 2) + 4))

    results = {}
    for splitter in ['best', 'random']:
        row = f"  {splitter:<10}"
        results[splitter] = {}
        for d in D_VALUES:
            scores = [_run(task, d, splitter, s) for s in SEEDS]
            mean = np.mean(scores)
            results[splitter][d] = mean
            row += f"  {mean:>{COL_W}.4f}"
        print(row)

    # Delta row
    delta_row = f"  {'delta':<10}"
    crossover = None
    for d in D_VALUES:
        delta = results['random'][d] - results['best'][d]
        delta_row += f"  {delta:>+{COL_W}.4f}"
        if delta < 0 and crossover is None:
            crossover = d
    print(delta_row)

    if crossover:
        print(f"\n  Crossover: random loses to best at d={crossover}")
        print(f"  Suggested auto threshold: d < {crossover} -> random, d >= {crossover} -> best")
    else:
        print(f"\n  No crossover found: random wins across all d values tested")
    print()

print("=== DONE ===")
