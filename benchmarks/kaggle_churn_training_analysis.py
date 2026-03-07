"""
Kaggle Churn — Training data analysis after HVRT reduce/expand.

Examines what happens to the training data after GeoXGB's resampling:
1. How many unique values survive per feature after FPS reduction?
2. What do the synthetic (expanded) samples look like?
3. How does the GBT training set differ from original?
4. Compare training predictions distribution to true labels
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from geoxgb import GeoXGBClassifier

DATA_DIR = "data"

# -- Load ----------------------------------------------------------------
tr = pd.read_csv(f"{DATA_DIR}/train.csv")
y = (tr["Churn"] == "Yes").astype(int).values
X_df = tr.drop(columns=["id", "Churn"])
feat_names = list(X_df.columns)
cat_cols = X_df.select_dtypes("object").columns.tolist()
for col in cat_cols:
    X_df[col] = LabelEncoder().fit_transform(X_df[col])
X = X_df.values.astype(np.float64)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
tr_idx, va_idx = next(iter(skf.split(X, y)))

# -- Train with msl=10 (best from sweep) --------------------------------
geo = GeoXGBClassifier(
    n_rounds=1000, max_depth=6, learning_rate=0.10,
    class_weight="balanced", random_state=42,
    hvrt_min_samples_leaf=10,
)
geo.fit(X[tr_idx], y[tr_idx])
cpp = getattr(geo, '_cpp_model', None) or getattr(geo, '_mc_cpp_model', None)

p_geo = geo.predict_proba(X[va_idx])[:, 1]
print(f"AUC: {roc_auc_score(y[va_idx], p_geo):.6f}")

if cpp is not None:
    # Training data after HVRT processing
    X_z_train = np.asarray(cpp.X_z())
    part_ids_train = np.asarray(cpp.partition_ids())
    train_preds = np.asarray(cpp.train_predictions())

    print(f"\nTraining data shape: {X_z_train.shape}")
    print(f"Original train shape: {X[tr_idx].shape}")
    print(f"Compression ratio: {X_z_train.shape[0] / X[tr_idx].shape[0]:.4f}")
    print(f"Unique partitions in training: {len(np.unique(part_ids_train))}")

    # The training set IS the reduced+expanded set
    n_train = X_z_train.shape[0]
    print(f"\nTraining set size: {n_train}")

    # Z-score distribution per feature
    print("\nZ-score distribution per feature:")
    print(f"  {'Feature':<20} {'mean':>8} {'std':>8} {'min':>8} {'max':>8} {'unique':>8}")
    for i, fn in enumerate(feat_names):
        z = X_z_train[:, i]
        print(f"  {fn:<20} {z.mean():>8.3f} {z.std():>8.3f} "
              f"{z.min():>8.3f} {z.max():>8.3f} {len(np.unique(z)):>8}")

    # Training predictions distribution
    print(f"\nTraining predictions:")
    print(f"  mean={train_preds.mean():.4f}  std={train_preds.std():.4f}")
    print(f"  min={train_preds.min():.4f}  max={train_preds.max():.4f}")
    for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        print(f"  p{int(q*100):>2}={np.quantile(train_preds, q):.4f}")

    # Per-partition training prediction stats
    print("\nPer-partition training predictions (sorted by size):")
    unique_parts = np.unique(part_ids_train)
    part_stats = []
    for pid in unique_parts:
        mask = part_ids_train == pid
        n = mask.sum()
        mean_pred = train_preds[mask].mean()
        std_pred = train_preds[mask].std()
        part_stats.append((n, pid, mean_pred, std_pred))

    part_stats.sort(reverse=True)
    for n, pid, mp, sp in part_stats[:15]:
        print(f"  part={pid:>3} n={n:>6} mean_pred={mp:>+8.4f} std_pred={sp:>7.4f}")

    # Convergence info
    print(f"\nn_trees: {geo.n_trees}")
    print(f"convergence_round: {cpp.convergence_round()}")
    print(f"n_init_reduced: {cpp.n_init_reduced()}")

    # Noise estimate
    ne = geo.noise_estimate()
    print(f"noise_estimate: {ne:.4f}")

    # Sample provenance
    prov = geo.sample_provenance()
    print(f"sample_provenance: {prov}")
