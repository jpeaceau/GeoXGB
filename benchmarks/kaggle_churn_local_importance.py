"""
Kaggle Churn — Local (geometrical) importance analysis.

Examines per-partition feature importance from the HVRT geometry layer:
1. Which features does the HVRT partition tree split on? (geometrical importance)
2. Per-partition z-score variance (which features vary locally?)
3. Per-partition cooperation matrices (local feature interactions)
4. Compare geometrical vs boosting importance
5. Test effect of hvrt_min_samples_leaf on partition granularity and AUC
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

print(f"Train: {X[tr_idx].shape}, Val: {X[va_idx].shape}")
print(f"Features: {len(feat_names)}, d={X.shape[1]}")
print(f"Churn rate: train={y[tr_idx].mean():.4f} val={y[va_idx].mean():.4f}")
print()

# -- Auto-tune formula analysis ------------------------------------------
print("=" * 70)
print("AUTO-TUNE FORMULA ANALYSIS (d=19)")
print("=" * 70)

d = X.shape[1]
msl_auto = max(5, (d * 40 * 2) // 3)
print(f"  msl = max(5, (d*40*2)//3) = max(5, {(d*40*2)//3}) = {msl_auto}")

for block_n in [2000, 5000, 9400, 15000, 30000]:
    max_leaf = max(30, min(1500, 3 * block_n // (msl_auto * 2)))
    n_parts_approx = block_n // msl_auto
    print(f"  block_n={block_n:>6}: max_leaf={max_leaf:>4}, "
          f"max possible parts={n_parts_approx:>3}, "
          f"samples/partition~{block_n//max(1,n_parts_approx):>4}")

print()
print("  With msl=10:")
msl_small = 10
for block_n in [2000, 5000, 9400, 15000]:
    max_leaf = max(30, min(1500, 3 * block_n // (msl_small * 2)))
    print(f"  block_n={block_n:>6}: max_leaf={max_leaf:>4}, "
          f"samples/partition~{block_n//max_leaf:>4}")

# -- Experiment: sweep hvrt_min_samples_leaf -----------------------------
print()
print("=" * 70)
print("HVRT MSL SWEEP: effect on partitions, importances, and AUC")
print("=" * 70)

for msl in [None, 200, 100, 50, 20, 10, 5]:
    t0 = time.time()
    kw = dict(
        n_rounds=1000, max_depth=6, learning_rate=0.10,
        class_weight="balanced", random_state=42,
    )
    if msl is not None:
        kw["hvrt_min_samples_leaf"] = msl

    geo = GeoXGBClassifier(**kw)
    geo.fit(X[tr_idx], y[tr_idx])
    p_geo = geo.predict_proba(X[va_idx])[:, 1]
    auc = roc_auc_score(y[va_idx], p_geo)
    elapsed = time.time() - t0

    # Get partition info
    cpp = getattr(geo, '_cpp_model', None) or getattr(geo, '_mc_cpp_model', None)
    if cpp is not None:
        part_ids = np.asarray(cpp.apply(X[va_idx]))
        n_parts = len(np.unique(part_ids))

        # Feature importances (boosting)
        fi = geo.feature_importances(feat_names)
        n_nonzero = sum(1 for v in fi.values() if v > 0.001)

        # Z-score analysis: per-partition feature variance
        X_z = np.asarray(cpp.to_z(X[va_idx]))
        train_part_ids = np.asarray(cpp.partition_ids())
        n_train_parts = len(np.unique(train_part_ids))

        # Mean z-score std per feature across partitions
        z_var_per_feat = np.zeros(d)
        for pid in np.unique(part_ids):
            mask = part_ids == pid
            if mask.sum() > 5:
                z_var_per_feat += X_z[mask].std(axis=0)
        z_var_per_feat /= max(1, len(np.unique(part_ids)))

        label = f"msl={str(msl):>4}" if msl is not None else "msl=auto"
        print(f"\n  {label}: AUC={auc:.6f}  parts={n_parts:>3} (train={n_train_parts:>3})  "
              f"nonzero_feats={n_nonzero:>2}/19  n_train={cpp.n_train()}  ({elapsed:.1f}s)")

        # Top boosting features
        top_boost = sorted(fi.items(), key=lambda x: -x[1])[:5]
        boost_str = ", ".join(f"{k}={v:.3f}" for k, v in top_boost)
        print(f"    top boost: {boost_str}")

        # Top z-score-varying features (geometrical signal)
        z_rank = np.argsort(z_var_per_feat)[::-1]
        z_str = ", ".join(f"{feat_names[i]}={z_var_per_feat[i]:.3f}" for i in z_rank[:5])
        print(f"    top z-var: {z_str}")
    else:
        print(f"  msl={msl}: AUC={auc:.6f}  ({elapsed:.1f}s)  [no cpp model]")

# -- Deep local importance for best msl ----------------------------------
print()
print("=" * 70)
print("LOCAL IMPORTANCE: per-partition cooperation and feature analysis")
print("=" * 70)

best_msl = 10
geo = GeoXGBClassifier(
    n_rounds=1000, max_depth=6, learning_rate=0.10,
    class_weight="balanced", random_state=42,
    hvrt_min_samples_leaf=best_msl,
)
geo.fit(X[tr_idx], y[tr_idx])
cpp = getattr(geo, '_cpp_model', None) or getattr(geo, '_mc_cpp_model', None)

if cpp is not None:
    X_val = X[va_idx]
    y_val = y[va_idx]
    p_geo = geo.predict_proba(X_val)[:, 1]

    part_ids = np.asarray(cpp.apply(X_val))
    X_z = np.asarray(cpp.to_z(X_val))
    unique_parts = np.unique(part_ids)

    print(f"\n  Total validation partitions: {len(unique_parts)}")
    print(f"  AUC: {roc_auc_score(y_val, p_geo):.6f}")
    print(f"  Partition size stats: min={min(np.bincount(part_ids)[np.bincount(part_ids)>0])}"
          f"  max={max(np.bincount(part_ids))}"
          f"  median={np.median(np.bincount(part_ids)[np.bincount(part_ids)>0]):.0f}")

    # Per-partition: which features have high z-variance?
    # High z-variance = feature is geometrically discriminative in that partition
    print("\n  Per-partition feature z-variance (top 10 largest partitions):")

    part_counts = np.bincount(part_ids)
    top_parts = np.argsort(part_counts)[::-1][:10]

    for pid in top_parts:
        mask = part_ids == pid
        n_in = mask.sum()
        if n_in < 10:
            continue
        z_std = X_z[mask].std(axis=0)
        churn_rate = y_val[mask].mean()
        auc_local = "N/A"
        if y_val[mask].sum() > 3 and (1 - y_val[mask]).sum() > 3:
            auc_local = f"{roc_auc_score(y_val[mask], p_geo[mask]):.4f}"

        top3 = np.argsort(z_std)[::-1][:3]
        feat_str = ", ".join(f"{feat_names[i]}={z_std[i]:.3f}" for i in top3)
        print(f"    part={pid:>3} n={n_in:>5} churn={churn_rate:.3f} "
              f"AUC={auc_local}  top_z: {feat_str}")

    # Cooperation analysis: per-partition correlation structure
    print("\n  Cooperation matrix (top interactions per partition, largest 5 parts):")

    for pid in top_parts[:5]:
        mask = part_ids == pid
        n_in = mask.sum()
        if n_in < 20:
            continue

        z_part = X_z[mask]
        # Compute correlation matrix of z-scores
        z_centered = z_part - z_part.mean(axis=0)
        z_std = z_part.std(axis=0)
        z_std[z_std < 1e-10] = 1.0
        z_normed = z_centered / z_std

        corr = (z_normed.T @ z_normed) / max(1, n_in - 1)
        np.fill_diagonal(corr, 0)

        # Top interactions
        top_pairs = []
        for i in range(d):
            for j in range(i+1, d):
                top_pairs.append((abs(corr[i, j]), corr[i, j], i, j))
        top_pairs.sort(reverse=True)

        print(f"\n    Part {pid} (n={n_in}):")
        for _, c, i, j in top_pairs[:5]:
            print(f"      {feat_names[i]:>20} x {feat_names[j]:<20} corr={c:+.4f}")

    # Feature usage across partitions: how many features are "alive" in each?
    print("\n  Feature coverage: how many features have z-std > 0.1 per partition")
    alive_counts = []
    for pid in unique_parts:
        mask = part_ids == pid
        if mask.sum() < 5:
            continue
        z_std = X_z[mask].std(axis=0)
        alive = (z_std > 0.1).sum()
        alive_counts.append(alive)

    if alive_counts:
        alive_arr = np.array(alive_counts)
        print(f"    mean={alive_arr.mean():.1f}  median={np.median(alive_arr):.0f}  "
              f"min={alive_arr.min()}  max={alive_arr.max()}")

    # Per-feature: in what fraction of partitions is it "alive"?
    print("\n  Per-feature alive fraction (z-std > 0.1):")
    feat_alive = np.zeros(d)
    n_counted = 0
    for pid in unique_parts:
        mask = part_ids == pid
        if mask.sum() < 5:
            continue
        z_std = X_z[mask].std(axis=0)
        feat_alive += (z_std > 0.1).astype(float)
        n_counted += 1

    for i in np.argsort(feat_alive)[::-1]:
        frac = feat_alive[i] / max(1, n_counted)
        print(f"    {feat_names[i]:<20} alive in {frac:.2%} of partitions "
              f"({int(feat_alive[i])}/{n_counted})")
