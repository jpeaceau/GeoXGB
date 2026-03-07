"""
Analyze local cooperation matrices for the 100 worst GeoXGB predictions
on the Kaggle churn dataset.

Questions:
1. Does Contract dominate because it genuinely has the most signal, or are
   other features locally informative but ignored by GBT?
2. What does the geometry (z-space cooperation) look like for misclassified samples?
3. Are there features with strong local cooperation that get zero boosting importance?
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ── Load data (Kaggle competition format) ────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

y_all = (df["Churn"] == "Yes").astype(int).values.astype(np.float64)
drop_cols = ["id", "Churn"]
feat_cols = [c for c in df.columns if c not in drop_cols]

# Label-encode categoricals
les = {}
for c in feat_cols:
    if df[c].dtype == object:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        les[c] = le

X_all = df[feat_cols].values.astype(np.float64)

print(f"Dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features")
print(f"Features: {feat_cols}")
print(f"Churn rate: {y_all.mean():.3f}")

# Use a single stratified fold for analysis (same as CV fold 0)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
train_idx, test_idx = next(iter(skf.split(X_all, y_all)))
X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]

# ── Fit GeoXGB ───────────────────────────────────────────────────────────────
from geoxgb import GeoXGBClassifier

clf = GeoXGBClassifier(
    n_rounds=1000,
    learning_rate=0.02,
    max_depth=3,
    refit_interval=50,
    random_state=42,
)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"\nGeoXGB test AUC: {auc:.4f}")

# ── Identify 100 worst predictions ──────────────────────────────────────────
# "Worst" = highest absolute error between predicted probability and true label
errors = np.abs(y_test - proba)
worst_idx = np.argsort(errors)[-100:][::-1]  # descending by error

print(f"\n{'='*70}")
print(f"100 WORST PREDICTIONS")
print(f"{'='*70}")
print(f"Error range: [{errors[worst_idx[-1]]:.4f}, {errors[worst_idx[0]]:.4f}]")
print(f"True label distribution: {y_test[worst_idx].mean():.2f} churn rate")
print(f"Mean predicted proba: {proba[worst_idx].mean():.4f}")

# How many are FP vs FN?
preds = (proba > 0.5).astype(int)
fp_mask = (preds[worst_idx] == 1) & (y_test[worst_idx] == 0)
fn_mask = (preds[worst_idx] == 0) & (y_test[worst_idx] == 1)
print(f"False positives: {fp_mask.sum()}, False negatives: {fn_mask.sum()}")
print(f"Confident wrong (error > 0.8): {(errors[worst_idx] > 0.8).sum()}")

# ── Per-feature analysis for worst predictions ──────────────────────────────
print(f"\n{'='*70}")
print(f"FEATURE VALUE DISTRIBUTION: WORST 100 vs OVERALL TEST SET")
print(f"{'='*70}")
for i, feat in enumerate(feat_cols):
    worst_vals = X_test[worst_idx, i]
    all_vals = X_test[:, i]
    print(f"  {feat:25s}  worst: mean={worst_vals.mean():.3f} std={worst_vals.std():.3f}"
          f"  |  all: mean={all_vals.mean():.3f} std={all_vals.std():.3f}"
          f"  |  diff={worst_vals.mean() - all_vals.mean():+.3f}")

# ── Local cooperation matrices for worst 100 ────────────────────────────────
print(f"\n{'='*70}")
print(f"LOCAL COOPERATION MATRICES (100 worst predictions)")
print(f"{'='*70}")

coop_result = clf.cooperation_matrix(X_test[worst_idx])
coop = coop_result['matrices']  # (100, d, d)
d = coop.shape[1]

# Average cooperation across worst 100
avg_coop = np.mean(coop, axis=0)  # (d, d)

print(f"\nAverage cooperation matrix (top off-diagonal pairs):")
pairs = []
for i in range(d):
    for j in range(i+1, d):
        pairs.append((feat_cols[i], feat_cols[j], avg_coop[i, j]))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for a, b, v in pairs[:20]:
    print(f"  {a:20s} x {b:20s} = {v:+.4f}")

# ── Compare: cooperation for BEST 100 predictions ───────────────────────────
best_idx = np.argsort(errors)[:100]  # lowest error
coop_best_result = clf.cooperation_matrix(X_test[best_idx])
coop_best = coop_best_result['matrices']
avg_coop_best = np.mean(coop_best, axis=0)

print(f"\n{'='*70}")
print(f"COOPERATION COMPARISON: WORST vs BEST 100")
print(f"{'='*70}")
print(f"{'Feature A':>20s} x {'Feature B':>20s}  | {'Worst':>8s} {'Best':>8s} {'Diff':>8s}")
print("-" * 70)
for a, b, v_worst in pairs[:20]:
    i = feat_cols.index(a)
    j = feat_cols.index(b)
    v_best = avg_coop_best[i, j]
    print(f"  {a:20s} x {b:20s}  | {v_worst:+.4f} {v_best:+.4f} {v_worst - v_best:+.4f}")

# ── Per-feature diagonal: local z-variance (self-cooperation = 1.0 always for
#    standardized z, but let's check effective spread) ────────────────────────

# ── Boosting importance vs geometric importance ──────────────────────────────
print(f"\n{'='*70}")
print(f"BOOSTING IMPORTANCE vs GEOMETRIC ACTIVITY")
print(f"{'='*70}")

fimp_raw = clf._cpp_model.feature_importances()  # raw list indexed by feature
fimp = np.asarray(fimp_raw)

# Geometric activity: mean |z-score| across worst 100 partitions
X_z_worst = clf._cpp_model.to_z(X_test[worst_idx])  # (100, d) z-scores
part_ids_worst = clf._cpp_model.apply(X_test[worst_idx])  # (100,) partition assignments

geo_activity = np.mean(np.abs(X_z_worst), axis=0)  # mean |z| per feature

# Also get cooperation strength per feature (sum of |off-diagonal| in coop matrix)
coop_strength = np.zeros(d)
for i in range(d):
    for j in range(d):
        if i != j:
            coop_strength[i] += np.mean(np.abs(coop[:, i, j]))

print(f"\n{'Feature':>25s} | {'Boost Imp':>10s} | {'Geo |z|':>10s} | {'Coop Str':>10s} | {'Status':>12s}")
print("-" * 80)
for i in np.argsort(fimp)[::-1]:
    status = ""
    if fimp[i] == 0 and (geo_activity[i] > 0.3 or coop_strength[i] > 0.5):
        status = "GEO UNUSED"
    elif fimp[i] > 0.05 and geo_activity[i] < 0.2:
        status = "BOOST ONLY"
    print(f"  {feat_cols[i]:25s} | {fimp[i]:10.4f} | {geo_activity[i]:10.4f} | {coop_strength[i]:10.4f} | {status:>12s}")

# ── Partition distribution for worst 100 ─────────────────────────────────────
print(f"\n{'='*70}")
print(f"PARTITION DISTRIBUTION (worst 100 predictions)")
print(f"{'='*70}")
unique_parts, counts = np.unique(part_ids_worst, return_counts=True)
print(f"Unique partitions: {len(unique_parts)} (out of 100 samples)")
for p, c in sorted(zip(unique_parts, counts), key=lambda x: -x[1])[:10]:
    print(f"  Partition {p:4d}: {c:3d} samples")

# ── Single-feature AUC (univariate signal) ──────────────────────────────────
print(f"\n{'='*70}")
print(f"SINGLE-FEATURE AUC (univariate signal strength)")
print(f"{'='*70}")
for i in np.argsort(fimp)[::-1]:
    try:
        uni_auc = roc_auc_score(y_test, X_test[:, i])
        uni_auc = max(uni_auc, 1 - uni_auc)  # handle inverted features
    except:
        uni_auc = 0.5
    print(f"  {feat_cols[i]:25s} | boost_imp={fimp[i]:.4f} | univariate_AUC={uni_auc:.4f}"
          f" | {'*** SIGNAL MISSED ***' if fimp[i] == 0 and uni_auc > 0.55 else ''}")

# ── Z-space profile for a few worst predictions ─────────────────────────────
print(f"\n{'='*70}")
print(f"Z-SPACE PROFILE (5 worst predictions)")
print(f"{'='*70}")
for rank, idx in enumerate(worst_idx[:5]):
    x_sample = X_test[idx:idx+1]
    z_sample = np.asarray(clf._cpp_model.to_z(x_sample))[0]  # (d,)
    part_id = int(np.asarray(clf._cpp_model.apply(x_sample))[0])
    print(f"\n--- Rank #{rank+1} worst (error={errors[idx]:.4f}, true={y_test[idx]:.0f}, pred={proba[idx]:.4f}) ---")
    print(f"  Partition: {part_id}")
    print(f"  Raw features and z-scores:")
    for i in range(d):
        print(f"    {feat_cols[i]:25s}: raw={X_test[idx, i]:8.2f}  z={z_sample[i]:+.4f}")

# ── XGBoost comparison: which features does XGB use for worst predictions? ──
print(f"\n{'='*70}")
print(f"XGBOOST COMPARISON")
print(f"{'='*70}")
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=3,
                         random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    print(f"XGBoost AUC: {xgb_auc:.4f}")

    xgb_errors = np.abs(y_test - xgb_proba)
    # Check XGB's predictions on GeoXGB's worst 100
    print(f"\nXGBoost predictions on GeoXGB's 100 worst samples:")
    print(f"  Mean proba: {xgb_proba[worst_idx].mean():.4f}")
    print(f"  XGB also wrong (error>0.5): {(xgb_errors[worst_idx] > 0.5).sum()}/100")
    print(f"  XGB confident correct (error<0.2): {(xgb_errors[worst_idx] < 0.2).sum()}/100")

    # XGB feature importance comparison
    xgb_fimp = xgb.feature_importances_
    print(f"\nXGBoost feature importances (top 10):")
    for i in np.argsort(xgb_fimp)[::-1][:10]:
        print(f"  {feat_cols[i]:25s}: {xgb_fimp[i]:.4f}  (GeoXGB: {fimp[i]:.4f})")
except ImportError:
    print("XGBoost not available")

print("\nDone.")
