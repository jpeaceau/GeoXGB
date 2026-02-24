"""
Diabetes Dataset — GeoXGB Degradation Investigation
=====================================================

Observed: GeoXGB peaks at ~50 rounds then degrades monotonically.
Hypothesis: degenerate HVRT partitions + aggressive auto_expand flooding
training set with low-quality synthetic samples at this n.

This script diagnoses:
  1. Noise modulation and partition structure across training
  2. n_reduced / n_expanded progression
  3. Round-by-round R2: default vs auto_expand=False vs explicit hvrt_min_samples_leaf
  4. Whether the degradation is partition-collapse or noise-misclassification
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")

from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from geoxgb import GeoXGBRegressor

X, y = load_diabetes(return_X_y=True)
X = X.astype(float)
print(f"Diabetes: n={len(X)}, d={X.shape[1]}, y range [{y.min():.0f}, {y.max():.0f}]")
print()

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train n={len(Xtr)}, Test n={len(Xte)}")
print()

# ── 1. Partition / noise diagnostics at default settings ─────────────────────
print("=== Diagnostic: partition structure (default settings, 100 rounds) ===")
m = GeoXGBRegressor(n_rounds=100, random_state=42)
m.fit(Xtr, ytr)

for entry in m.partition_trace():
    print(f"  round={entry['round']:>4}  noise_mod={entry['noise_modulation']:.3f}"
          f"  n_reduced={entry['n_reduced']:>5}  n_expanded={entry['n_expanded']:>5}"
          f"  n_partitions={len(entry['partitions']):>3}"
          f"  total_training={entry['n_samples']:>5}")
print()

# ── 2. R2 at checkpoint rounds ────────────────────────────────────────────────
print("=== R2 by round: 4 configurations ===")
configs = [
    ("default",                  dict()),
    ("auto_expand=False",        dict(auto_expand=False)),
    ("hvrt_msl=10",              dict(hvrt_min_samples_leaf=10)),
    ("hvrt_msl=10,noexp",        dict(hvrt_min_samples_leaf=10, auto_expand=False)),
]
checkpoints = [10, 25, 50, 100, 200, 500, 1000]
COL = 9

header = f"  {'Config':<24}" + "".join(f"  {f'r={r}':>{COL}}" for r in checkpoints)
print(header)
print("  " + "-" * (24 + len(checkpoints) * (COL + 2) + 2))

for label, kw in configs:
    row = f"  {label:<24}"
    for r in checkpoints:
        m2 = GeoXGBRegressor(n_rounds=r, random_state=42, **kw)
        m2.fit(Xtr, ytr)
        score = r2_score(yte, m2.predict(Xte))
        row += f"  {score:>{COL}.4f}"
    print(row)
print()

# ── 3. Partition structure with hvrt_min_samples_leaf=10 ─────────────────────
print("=== Diagnostic: partition structure (hvrt_msl=10, 100 rounds) ===")
m3 = GeoXGBRegressor(n_rounds=100, hvrt_min_samples_leaf=10, random_state=42)
m3.fit(Xtr, ytr)
for entry in m3.partition_trace():
    print(f"  round={entry['round']:>4}  noise_mod={entry['noise_modulation']:.3f}"
          f"  n_reduced={entry['n_reduced']:>5}  n_expanded={entry['n_expanded']:>5}"
          f"  n_partitions={len(entry['partitions']):>3}"
          f"  total_training={entry['n_samples']:>5}")
print()

# ── 4. XGBoost baseline ───────────────────────────────────────────────────────
print("=== XGBoost baseline (R2 at checkpoints) ===")
from xgboost import XGBRegressor
row = f"  {'xgboost':<24}"
for r in checkpoints:
    xm = XGBRegressor(n_estimators=r, learning_rate=0.1, max_depth=4,
                      tree_method='hist', verbosity=0, random_state=42)
    xm.fit(Xtr, ytr)
    row += f"  {r2_score(yte, xm.predict(Xte)):>{COL}.4f}"
print(row)
print()

print("=== DONE ===")
