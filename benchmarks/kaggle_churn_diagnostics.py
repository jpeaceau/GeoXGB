"""
Kaggle Churn — Deep diagnostic of GeoXGB vs XGBoost residuals and tree splits.

Examines:
1. Residual distributions (where are errors concentrated?)
2. Precision-recall and ROC curves
3. Tree split analysis (which features/thresholds dominate?)
4. Per-partition residual density
5. Feature-specific error analysis
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    average_precision_score, classification_report,
)
from sklearn.preprocessing import LabelEncoder
from geoxgb import GeoXGBClassifier
from xgboost import XGBClassifier

DATA_DIR = "data"

# ── Load ─────────────────────────────────────────────────────────────
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
spw = (y == 0).sum() / (y == 1).sum()

# ── Train both ───────────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    tree_method="hist", verbosity=0, random_state=42, scale_pos_weight=spw,
)
xgb.fit(X[tr_idx], y[tr_idx])
p_xgb = xgb.predict_proba(X[va_idx])[:, 1]

geo = GeoXGBClassifier(
    n_rounds=1000, max_depth=6, learning_rate=0.10,
    class_weight="balanced", random_state=42,
)
geo.fit(X[tr_idx], y[tr_idx])
p_geo = geo.predict_proba(X[va_idx])[:, 1]

y_val = y[va_idx]
X_val = X[va_idx]

print(f"AUC:  XGB={roc_auc_score(y_val, p_xgb):.6f}  GEO={roc_auc_score(y_val, p_geo):.6f}")
print(f"AP:   XGB={average_precision_score(y_val, p_xgb):.6f}  GEO={average_precision_score(y_val, p_geo):.6f}")
print()

# ── 1. Residual analysis ────────────────────────────────────────────
print("=" * 70)
print("1. RESIDUAL ANALYSIS")
print("=" * 70)

for name, p in [("XGB", p_xgb), ("GEO", p_geo)]:
    resid = y_val - p  # positive = under-predicted churn, negative = over-predicted
    print(f"\n{name} residuals:")
    print(f"  mean={resid.mean():.4f}  std={resid.std():.4f}")
    print(f"  min={resid.min():.4f}  max={resid.max():.4f}")

    # Per class
    for cls in [0, 1]:
        mask = y_val == cls
        r = resid[mask]
        print(f"  Class {cls} (n={mask.sum()}): mean_resid={r.mean():.4f} std={r.std():.4f}")

    # Residual by magnitude buckets
    abs_resid = np.abs(resid)
    for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        frac = (abs_resid > thresh).mean()
        print(f"  |resid| > {thresh}: {frac:.4f} ({frac*len(resid):.0f} samples)")

# ── 2. Where XGB wins and GEO loses ─────────────────────────────────
print()
print("=" * 70)
print("2. SAMPLES WHERE XGB >> GEO")
print("=" * 70)

# Find samples where XGB is much more correct
xgb_better = np.abs(y_val - p_xgb) < np.abs(y_val - p_geo)
xgb_margin = np.abs(y_val - p_geo) - np.abs(y_val - p_xgb)

# Top disagreement samples
top_k = 1000
top_idx = np.argsort(xgb_margin)[-top_k:]  # XGB much better

print(f"\nTop {top_k} samples where XGB >> GEO:")
print(f"  Mean margin: {xgb_margin[top_idx].mean():.4f}")
print(f"  Class distribution: {y_val[top_idx].mean():.4f} (churn rate)")

# Feature distributions of these disagreement samples vs overall
print("\n  Feature means (disagree vs overall):")
for i, fn in enumerate(feat_names):
    mean_disagree = X_val[top_idx, i].mean()
    mean_overall = X_val[:, i].mean()
    std_overall = X_val[:, i].std() + 1e-10
    z = (mean_disagree - mean_overall) / std_overall
    if abs(z) > 0.3:
        print(f"    {fn:<20} disagree={mean_disagree:.3f}  overall={mean_overall:.3f}  z={z:+.2f}")

# ── 3. Feature-specific error analysis ──────────────────────────────
print()
print("=" * 70)
print("3. FEATURE-CONDITIONED AUC")
print("=" * 70)

# For each categorical feature, compute AUC within each category
for i, fn in enumerate(feat_names):
    if fn in cat_cols or X_val[:, i].max() <= 5:  # categorical or binary
        unique_vals = np.unique(X_val[:, i])
        if len(unique_vals) <= 5:
            print(f"\n  {fn}:")
            for v in unique_vals:
                mask = X_val[:, i] == v
                n_pos = y_val[mask].sum()
                n_neg = (1 - y_val[mask]).sum()
                if n_pos > 10 and n_neg > 10:
                    auc_x = roc_auc_score(y_val[mask], p_xgb[mask])
                    auc_g = roc_auc_score(y_val[mask], p_geo[mask])
                    gap = auc_g - auc_x
                    marker = " ***" if abs(gap) > 0.02 else ""
                    print(f"    val={v:.0f}: n={mask.sum():>6}  XGB={auc_x:.4f}  GEO={auc_g:.4f}  gap={gap:+.4f}{marker}")

# ── 4. Contract-specific analysis (dominant feature) ────────────────
print()
print("=" * 70)
print("4. CONTRACT-SPECIFIC ANALYSIS (dominant feature)")
print("=" * 70)

contract_idx = feat_names.index("Contract")
for v in np.unique(X_val[:, contract_idx]):
    mask = X_val[:, contract_idx] == v
    print(f"\nContract={v:.0f} (n={mask.sum()}, churn={y_val[mask].mean():.4f}):")
    auc_x = roc_auc_score(y_val[mask], p_xgb[mask])
    auc_g = roc_auc_score(y_val[mask], p_geo[mask])
    print(f"  XGB AUC={auc_x:.4f}  GEO AUC={auc_g:.4f}  gap={auc_g-auc_x:+.4f}")
    # Prediction distribution within this contract type
    for name, p in [("XGB", p_xgb), ("GEO", p_geo)]:
        print(f"  {name}: mean_prob={p[mask].mean():.4f} std={p[mask].std():.4f}")

# ── 5. GeoXGB tree split analysis ───────────────────────────────────
print()
print("=" * 70)
print("5. GeoXGB INTERNAL DIAGNOSTICS")
print("=" * 70)

# Access GeoXGB's C++ model internals
cpp = getattr(geo, '_cpp_model', None) or getattr(geo, '_mc_cpp_model', None)
if cpp is not None:
    print(f"  n_trees: {geo.n_trees}")
    print(f"  convergence_round: {cpp.convergence_round()}")

    # Feature importances
    fi = geo.feature_importances(feat_names)
    print("\n  GeoXGB feature importances:")
    for fn in sorted(fi, key=fi.get, reverse=True):
        print(f"    {fn:<20} {fi[fn]:.4f}")

    # XGBoost importances for comparison
    xfi = dict(zip(feat_names, xgb.feature_importances_))
    print("\n  XGBoost feature importances:")
    for fn in sorted(xfi, key=xfi.get, reverse=True):
        print(f"    {fn:<20} {xfi[fn]:.4f}")

    # Importance divergence
    print("\n  Importance divergence (GEO - XGB):")
    for fn in feat_names:
        diff = fi.get(fn, 0) - xfi.get(fn, 0)
        if abs(diff) > 0.01:
            print(f"    {fn:<20} {diff:+.4f} ({'GEO overweights' if diff > 0 else 'XGB overweights'})")

    # Noise estimate
    ne = geo.noise_estimate()
    print(f"\n  Noise estimate: {ne:.4f}")

    # Sample provenance
    prov = geo.sample_provenance()
    print(f"  Sample provenance: {prov}")

# ── 6. Partition-level residual analysis ─────────────────────────────
print()
print("=" * 70)
print("6. PARTITION-LEVEL RESIDUAL DENSITY")
print("=" * 70)

if cpp is not None:
    try:
        part_ids = np.asarray(cpp.apply(X_val))
        unique_parts = np.unique(part_ids)
        print(f"  Unique partitions in validation: {len(unique_parts)}")

        # Per-partition AUC and residual stats
        part_stats = []
        for pid in unique_parts:
            mask = part_ids == pid
            n = mask.sum()
            n_pos = y_val[mask].sum()
            if n_pos > 5 and (n - n_pos) > 5:
                auc_g = roc_auc_score(y_val[mask], p_geo[mask])
                auc_x = roc_auc_score(y_val[mask], p_xgb[mask])
                mean_resid = (y_val[mask] - p_geo[mask]).mean()
                part_stats.append({
                    'partition': pid, 'n': n, 'churn_rate': y_val[mask].mean(),
                    'auc_geo': auc_g, 'auc_xgb': auc_x,
                    'gap': auc_g - auc_x, 'mean_resid': mean_resid,
                })

        df_parts = pd.DataFrame(part_stats).sort_values('gap')
        print(f"\n  Partitions where GEO loses most to XGB:")
        print(df_parts.head(10).to_string(index=False))
        print(f"\n  Partitions where GEO wins vs XGB:")
        print(df_parts.tail(5).to_string(index=False))

        print(f"\n  Overall partition stats:")
        print(f"    Mean gap: {df_parts['gap'].mean():.4f}")
        print(f"    Partitions where GEO > XGB: {(df_parts['gap'] > 0).sum()} / {len(df_parts)}")
        print(f"    Worst partition gap: {df_parts['gap'].min():.4f} (n={df_parts.iloc[0]['n']:.0f})")
    except Exception as e:
        print(f"  Partition analysis failed: {e}")
