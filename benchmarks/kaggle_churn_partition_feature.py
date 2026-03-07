"""
Test: add HVRT partition_id as a feature to GBT training.

Hypothesis: partition_id encodes geometric neighborhood, which concentrates
minority-class samples. A single GBT split on partition_id can access
multi-feature interactions that depth-3 trees can't find individually.
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/train.csv")
y_all = (df["Churn"] == "Yes").astype(int).values.astype(np.float64)
feat_cols = [c for c in df.columns if c not in ["id", "Churn"]]
for c in feat_cols:
    if df[c].dtype == object:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
X_all = df[feat_cols].values.astype(np.float64)

print(f"Dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features")
print(f"Churn rate: {y_all.mean():.4f}")

# ── First, verify the hypothesis: do partitions concentrate minority class? ──
from geoxgb import GeoXGBClassifier

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
tr_idx, va_idx = next(iter(skf.split(X_all, y_all)))
X_train, X_val = X_all[tr_idx], X_all[va_idx]
y_train, y_val = y_all[tr_idx], y_all[va_idx]

clf = GeoXGBClassifier(n_rounds=1000, learning_rate=0.02, max_depth=3,
                        refit_interval=50, random_state=42)
clf.fit(X_train, y_train)

# Get partition assignments for training data
part_ids_train = np.asarray(clf._cpp_model.partition_ids())
n_block = len(part_ids_train)

# The partition_ids are for the current block (last refit), not full training set.
# We need to map full training indices to partitions via apply()
part_ids_full = np.asarray(clf._cpp_model.apply(X_train))  # (n_train,)

print(f"\nPartition distribution (full training set via apply()):")
unique_parts = np.unique(part_ids_full)
print(f"  {len(unique_parts)} unique partitions")

# Check churn rate per partition
print(f"\nChurn rate by partition (top 20 by minority enrichment):")
part_stats = []
for p in unique_parts:
    mask = part_ids_full == p
    n = mask.sum()
    churn_rate = y_train[mask].mean()
    part_stats.append((p, n, churn_rate))

part_stats.sort(key=lambda x: x[2], reverse=True)
overall_churn = y_train.mean()
print(f"  Overall churn rate: {overall_churn:.4f}")
for p, n, cr in part_stats[:20]:
    enrichment = cr / overall_churn if overall_churn > 0 else 0
    print(f"  Partition {p:4d}: n={n:7d} churn_rate={cr:.4f} enrichment={enrichment:.2f}x")

print(f"\nPartitions with churn_rate > 2x overall:")
enriched = [(p, n, cr) for p, n, cr, in part_stats if cr > 2 * overall_churn]
print(f"  {len(enriched)} partitions, covering {sum(n for _, n, _ in enriched)} samples")

# ── Now test: XGBoost with partition_id as extra feature ─────────────────────
print(f"\n{'='*70}")
print(f"XGBOOST WITH PARTITION_ID FEATURE")
print(f"{'='*70}")

from xgboost import XGBClassifier

# Baseline: XGBoost without partition_id
xgb_base = XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=3,
                           random_state=42, eval_metric='logloss')
xgb_base.fit(X_train, y_train)
base_auc = roc_auc_score(y_val, xgb_base.predict_proba(X_val)[:, 1])
print(f"XGBoost baseline AUC: {base_auc:.4f}")

# With partition_id feature
part_ids_val = np.asarray(clf._cpp_model.apply(X_val))
X_train_aug = np.column_stack([X_train, part_ids_full.astype(np.float64)])
X_val_aug = np.column_stack([X_val, part_ids_val.astype(np.float64)])

xgb_aug = XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=3,
                          random_state=42, eval_metric='logloss')
xgb_aug.fit(X_train_aug, y_train)
aug_auc = roc_auc_score(y_val, xgb_aug.predict_proba(X_val_aug)[:, 1])
print(f"XGBoost + partition_id AUC: {aug_auc:.4f} (delta={aug_auc - base_auc:+.4f})")

# Check how much XGBoost uses the partition_id feature
fimp = xgb_aug.feature_importances_
part_imp = fimp[-1]
print(f"Partition_id feature importance: {part_imp:.4f} (rank {(fimp > part_imp).sum() + 1}/{len(fimp)})")

# ── Also test: GeoXGB baseline vs "what if GBT had partition info" ───────────
print(f"\n{'='*70}")
print(f"GEOXGB BASELINE vs PARTITION-AUGMENTED COMPARISON")
print(f"{'='*70}")
geo_proba = clf.predict_proba(X_val)[:, 1]
geo_auc = roc_auc_score(y_val, geo_proba)
print(f"GeoXGB baseline AUC: {geo_auc:.4f}")

# Per-contract-type breakdown with partition_id feature
for contract_val, name in [(0, 'Month-to-month'), (1, 'One year'), (2, 'Two year')]:
    mask = X_val[:, feat_cols.index('Contract')] == contract_val
    n_pos = int(y_val[mask].sum())
    if n_pos > 0 and (1 - y_val[mask]).sum() > 0:
        g = roc_auc_score(y_val[mask], geo_proba[mask])
        xb = roc_auc_score(y_val[mask], xgb_base.predict_proba(X_val[mask])[:, 1])
        xa = roc_auc_score(y_val[mask], xgb_aug.predict_proba(X_val_aug[mask])[:, 1])
        print(f"  {name:20s}: GeoXGB={g:.4f} XGB={xb:.4f} XGB+part={xa:.4f} (part_delta={xa-xb:+.4f})")

# ── Deeper: what's the partition_id capturing? ───────────────────────────────
print(f"\n{'='*70}")
print(f"PARTITION CHARACTERIZATION")
print(f"{'='*70}")
# For the 3 most minority-enriched partitions, show feature profiles
for p, n, cr in part_stats[:5]:
    mask = part_ids_full == p
    print(f"\nPartition {p} (n={n}, churn={cr:.4f}, enrichment={cr/overall_churn:.1f}x):")
    for i, feat in enumerate(feat_cols):
        p_mean = X_train[mask, i].mean()
        all_mean = X_train[:, i].mean()
        if abs(p_mean - all_mean) > 0.3 * X_train[:, i].std():
            print(f"  {feat:25s}: partition={p_mean:.3f} vs overall={all_mean:.3f} ({p_mean - all_mean:+.3f})")

print("\nDone.")
