"""
HVRT-Guided Feature Selection: Independent Validation
======================================================

Hypothesis: HVRT z-score variance per partition can deterministically select
feature subsets.  Different partitions activate different features, so
partition-aware feature selection across blocks would give broader feature
coverage than using all features in every tree.

This benchmark validates the hypothesis on synthetic and real data WITHOUT
modifying GeoXGB internals.  It answers three questions:

  Q1. Do HVRT partitions genuinely activate different feature subsets?
      (Measured by z-score variance per partition per feature.)

  Q2. Does training trees on partition-selected features use more total
      features than training on all features?

  Q3. Does this improve predictive performance?

Approach:
  - Fit HVRT on data, get partition_ids and X_z
  - For each partition, rank features by z-score variance (high variance =
    geometrically active in that neighborhood)
  - Select top-k features per partition (deterministic, interpretable)
  - Train per-partition XGBoost models on selected features
  - Compare ensemble vs full-feature XGBoost and GeoXGB
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings("ignore")


# ── Helpers ──────────────────────────────────────────────────────────────────

def fit_hvrt_and_get_geometry(X, y, n_partitions=None, min_samples_leaf=None):
    """Fit HVRT and return X_z, partition_ids, and the fitted model."""
    from hvrt import HVRT
    cfg = {"y_weight": 0.25, "random_state": 42}
    if n_partitions is not None:
        cfg["n_partitions"] = n_partitions
        cfg["auto_tune"] = False
    if min_samples_leaf is not None:
        cfg["min_samples_leaf"] = min_samples_leaf
        cfg["auto_tune"] = False
    h = HVRT(**cfg)
    h.fit(X, y)
    X_z = h._to_z(X)
    pids = h.tree_.apply(h._to_z(X))
    return X_z, pids, h


def partition_feature_ranking(X_z, partition_ids):
    """
    For each partition, rank features by z-score variance (descending).
    Returns dict: {part_id: [feat_idx_0, feat_idx_1, ...]} most-active first.
    """
    rankings = {}
    for pid in np.unique(partition_ids):
        mask = partition_ids == pid
        if mask.sum() < 5:
            continue
        z_var = np.var(X_z[mask], axis=0)
        rankings[pid] = np.argsort(-z_var).tolist()
    return rankings


def select_features_for_partition(rankings, pid, k, d):
    """Select top-k features for a partition. Falls back to all features."""
    if pid in rankings:
        return rankings[pid][:k]
    return list(range(d))


# ── Q1: Do partitions activate different features? ───────────────────────────

def q1_feature_diversity(X, y, dataset_name):
    """Measure how much feature activation varies across partitions."""
    X_z, pids, h = fit_hvrt_and_get_geometry(X, y)
    rankings = partition_feature_ranking(X_z, pids)
    d = X.shape[1]
    n_parts = len(rankings)

    if n_parts < 2:
        print(f"  [{dataset_name}] Only {n_parts} partition(s) — skipping Q1")
        return rankings, pids, h

    # For each k, count how many unique features appear in top-k across partitions
    print(f"\n  [{dataset_name}] {n_parts} partitions, {d} features")
    print(f"  {'k':>3s} | {'unique feats in top-k':>22s} | {'pct of d':>8s} | top-k sets (first 5 partitions)")
    for k in [3, 5, min(8, d), min(d // 2, 10)]:
        if k > d or k <= 0:
            continue
        all_selected = set()
        examples = []
        for i, (pid, ranking) in enumerate(sorted(rankings.items())):
            selected = ranking[:k]
            all_selected.update(selected)
            if i < 5:
                examples.append(str(selected))
        n_unique = len(all_selected)
        print(f"  {k:3d} | {n_unique:22d} | {n_unique/d:7.1%} | {', '.join(examples)}")

    # Pairwise Jaccard distance between partition top-k sets
    k_test = min(5, d)
    part_ids_list = sorted(rankings.keys())
    jaccards = []
    for i in range(len(part_ids_list)):
        for j in range(i + 1, len(part_ids_list)):
            s_i = set(rankings[part_ids_list[i]][:k_test])
            s_j = set(rankings[part_ids_list[j]][:k_test])
            jacc = len(s_i & s_j) / len(s_i | s_j)
            jaccards.append(jacc)
    mean_jacc = np.mean(jaccards) if jaccards else 1.0
    print(f"  Mean pairwise Jaccard (top-{k_test}): {mean_jacc:.3f}  "
          f"(0=completely different, 1=identical)")

    return rankings, pids, h


# ── Q2 & Q3: Feature coverage and predictive performance ────────────────────

def q2_q3_prediction_test(X, y, dataset_name, rankings, pids, h, k_feat=5):
    """
    Train per-partition decision trees on HVRT-selected features.
    Compare total feature usage and AUC vs full-feature trees.
    """
    d = X.shape[1]
    k = min(k_feat, d)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_full, auc_selected, auc_per_part = [], [], []
    feats_used_full, feats_used_selected = [], []

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        # Re-fit HVRT on training fold
        X_z_tr, pids_tr, h_tr = fit_hvrt_and_get_geometry(X_tr, y_tr)
        rankings_tr = partition_feature_ranking(X_z_tr, pids_tr)

        # Get partition assignments for validation set
        X_z_va = h_tr._to_z(X_va)
        pids_va = h_tr.tree_.apply(X_z_va)

        unique_parts = sorted(rankings_tr.keys())

        # ── Full-feature baseline: one tree per partition ──
        preds_full = np.full(len(X_va), np.nan)
        used_full = set()
        for pid in unique_parts:
            mask_tr = pids_tr == pid
            mask_va = pids_va == pid
            if mask_tr.sum() < 10 or mask_va.sum() == 0:
                continue
            dt = DecisionTreeRegressor(max_depth=4, random_state=42)
            dt.fit(X_tr[mask_tr], y_tr[mask_tr])
            preds_full[mask_va] = dt.predict(X_va[mask_va])
            used_full.update(np.where(dt.feature_importances_ > 0)[0])

        # Fill NaN with global mean
        nan_mask = np.isnan(preds_full)
        if nan_mask.any():
            preds_full[nan_mask] = y_tr.mean()

        # ── HVRT-selected features: one tree per partition on top-k feats ──
        preds_sel = np.full(len(X_va), np.nan)
        used_sel = set()
        for pid in unique_parts:
            mask_tr = pids_tr == pid
            mask_va = pids_va == pid
            if mask_tr.sum() < 10 or mask_va.sum() == 0:
                continue
            feat_idx = select_features_for_partition(rankings_tr, pid, k, d)
            dt = DecisionTreeRegressor(max_depth=4, random_state=42)
            dt.fit(X_tr[mask_tr][:, feat_idx], y_tr[mask_tr])
            preds_sel[mask_va] = dt.predict(X_va[mask_va][:, feat_idx])
            # Map back to original feature indices
            imp = dt.feature_importances_
            for local_i, orig_i in enumerate(feat_idx):
                if imp[local_i] > 0:
                    used_sel.add(orig_i)

        nan_mask = np.isnan(preds_sel)
        if nan_mask.any():
            preds_sel[nan_mask] = y_tr.mean()

        # ── Per-partition ensemble (average partition predictions) ──
        # This is the same as preds_sel since each sample gets one partition's pred

        # AUC
        try:
            auc_full.append(roc_auc_score(y_va, preds_full))
            auc_selected.append(roc_auc_score(y_va, preds_sel))
        except ValueError:
            pass
        feats_used_full.append(len(used_full))
        feats_used_selected.append(len(used_sel))

    print(f"\n  [{dataset_name}] Q2: Feature coverage (top-{k} per partition)")
    print(f"    Full-feature trees:     {np.mean(feats_used_full):.1f}/{d} features used")
    print(f"    HVRT-selected trees:    {np.mean(feats_used_selected):.1f}/{d} features used")

    if auc_full and auc_selected:
        print(f"  [{dataset_name}] Q3: Predictive performance (3-fold CV AUC)")
        print(f"    Full-feature per-part:  {np.mean(auc_full):.4f}")
        print(f"    HVRT-selected per-part: {np.mean(auc_selected):.4f}")
        delta = np.mean(auc_selected) - np.mean(auc_full)
        print(f"    Delta:                  {delta:+.4f}")


# ── Datasets ─────────────────────────────────────────────────────────────────

def make_imbalanced_multimodal(n=10000, d=19, n_informative=8, random_state=42):
    """
    Synthetic dataset mimicking churn: many features, few informative,
    imbalanced classes, with subgroup structure.
    """
    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=n_informative,
        n_redundant=3, n_clusters_per_class=4,
        weights=[0.73, 0.27], flip_y=0.03,
        random_state=random_state
    )
    return X, y.astype(np.float64)


def load_churn():
    """Load Kaggle churn dataset if available."""
    try:
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        df = pd.read_csv("data/train.csv")
        y = (df["Churn"] == "Yes").astype(int).values.astype(np.float64)
        feat_cols = [c for c in df.columns if c not in ["id", "Churn"]]
        for c in feat_cols:
            if df[c].dtype == object:
                df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        X = df[feat_cols].values.astype(np.float64)
        return X, y, feat_cols
    except FileNotFoundError:
        return None, None, None


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("HVRT-Guided Feature Selection: Independent Validation")
    print("=" * 72)

    # ── Synthetic dataset ────────────────────────────────────────────────
    print("\n── Synthetic: 10k samples, 19 feats (8 informative) ──")
    X_syn, y_syn = make_imbalanced_multimodal()
    rankings, pids, h = q1_feature_diversity(X_syn, y_syn, "synthetic")
    q2_q3_prediction_test(X_syn, y_syn, "synthetic", rankings, pids, h, k_feat=8)

    # ── Larger synthetic ─────────────────────────────────────────────────
    print("\n── Synthetic large: 50k samples, 30 feats (10 informative) ──")
    X_big, y_big = make_classification(
        n_samples=50000, n_features=30, n_informative=10,
        n_redundant=5, n_clusters_per_class=6,
        weights=[0.73, 0.27], flip_y=0.03, random_state=42
    )
    y_big = y_big.astype(np.float64)
    rankings_b, pids_b, h_b = q1_feature_diversity(X_big, y_big, "synthetic_large")
    q2_q3_prediction_test(X_big, y_big, "synthetic_large", rankings_b, pids_b, h_b, k_feat=10)

    # ── Kaggle churn (if available) ──────────────────────────────────────
    X_churn, y_churn, feat_names = load_churn()
    if X_churn is not None:
        # Subsample for speed
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_churn), size=min(50000, len(X_churn)), replace=False)
        X_sub, y_sub = X_churn[idx], y_churn[idx]
        print(f"\n── Kaggle Churn: {len(X_sub)} samples, {X_sub.shape[1]} feats ──")
        if feat_names:
            print(f"  Features: {feat_names}")
        rankings_c, pids_c, h_c = q1_feature_diversity(X_sub, y_sub, "churn")
        q2_q3_prediction_test(X_sub, y_sub, "churn", rankings_c, pids_c, h_c, k_feat=8)

        # Show which features each partition selects (with names)
        if feat_names:
            print(f"\n  [churn] Per-partition top-8 feature names:")
            X_z_full, pids_full, _ = fit_hvrt_and_get_geometry(X_sub, y_sub)
            rankings_named = partition_feature_ranking(X_z_full, pids_full)
            for pid in sorted(rankings_named.keys())[:10]:
                top8 = rankings_named[pid][:8]
                names = [feat_names[i] for i in top8]
                n_in_part = (pids_full == pid).sum()
                churn_rate = y_sub[pids_full == pid].mean()
                print(f"    Part {pid:3d} (n={n_in_part:5d}, churn={churn_rate:.1%}): "
                      f"{names}")
    else:
        print("\n  [churn] data/train.csv not found — skipping")

    print("\n" + "=" * 72)
    print("Done.")
