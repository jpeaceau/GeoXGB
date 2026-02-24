"""
Heart Disease Dataset -- GeoXGB Thorough Investigation
=======================================================

Dataset: data/Heart_Disease_Prediction.csv
  n=270, 13 features, binary (Absence / Presence)
  Small-n regime -- same class of challenges as diabetes.

Sections
--------
1.  Dataset characterisation: class balance, feature stats, correlations
2.  Noise / partition trace: how noise_mod and n_expanded evolve
3.  Hyperparameter grid: AUC at 1000 rounds across 15+ configs
4.  Checkpoint sweep: AUC vs rounds for top configs
5.  Residual analysis: prediction-error distribution, hardest samples
6.  Feature importance: boosting vs HVRT geometry layer
7.  Class-weight investigation: balanced vs none
8.  Failure audit: samples consistently misclassified across CV folds
9.  XGBoost baseline comparison
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from geoxgb import GeoXGBClassifier

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA = "data/Heart_Disease_Prediction.csv"
df = pd.read_csv(DATA)
le = LabelEncoder()
df["target"] = le.fit_transform(df["Heart Disease"])   # Absence=0, Presence=1
FEATURES = [c for c in df.columns if c not in ("Heart Disease", "target")]
X = df[FEATURES].values.astype(float)
y = df["target"].values
n, d = X.shape
pos_rate = y.mean()

print("=" * 70)
print("  1. DATASET CHARACTERISATION")
print("=" * 70)
print(f"  Samples   : {n}")
print(f"  Features  : {d}  {FEATURES}")
print(f"  Positive  : {y.sum()} ({pos_rate*100:.1f}%)   (Presence=1)")
print(f"  Negative  : {(1-y).sum()} ({(1-pos_rate)*100:.1f}%)   (Absence=0)")
print()
print("  Feature summary (mean | std):")
for i, fn in enumerate(FEATURES):
    print(f"    {fn:<28}  mu={X[:,i].mean():7.2f}  sd={X[:,i].std():7.2f}"
          f"  range=[{X[:,i].min():.0f}, {X[:,i].max():.0f}]")
print()

# Correlation with target
corrs = [(abs(np.corrcoef(X[:, i], y)[0, 1]), FEATURES[i]) for i in range(d)]
corrs.sort(reverse=True)
print("  Pearson |r| with target (descending):")
for r, fn in corrs:
    print(f"    {fn:<28}  |r|={r:.3f}")
print()

# ---------------------------------------------------------------------------
# Helper: CV AUC with a fixed set of splits
# ---------------------------------------------------------------------------
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
FOLDS = list(SKF.split(X, y))

def cv_auc(model_cls, **kw):
    scores = []
    for tr, te in FOLDS:
        m = model_cls(**kw)
        m.fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.mean(scores), np.std(scores)

# ---------------------------------------------------------------------------
# 2. Noise / partition trace at default settings
# ---------------------------------------------------------------------------
print("=" * 70)
print("  2. NOISE / PARTITION TRACE  (default, n_rounds=200, refit_interval=20)")
print("=" * 70)
tr_idx, te_idx = FOLDS[0]
_m = GeoXGBClassifier(n_rounds=200, random_state=42)
_m.fit(X[tr_idx], y[tr_idx])
for e in _m.partition_trace():
    print(f"  round={e['round']:>4}  noise_mod={e['noise_modulation']:.3f}"
          f"  n_reduced={e['n_reduced']:>4}  n_expanded={e['n_expanded']:>4}"
          f"  n_partitions={len(e['partitions']):>3}"
          f"  total={e['n_samples']:>5}")
print()

# ---------------------------------------------------------------------------
# 3. HYPERPARAMETER GRID
# ---------------------------------------------------------------------------
print("=" * 70)
print("  3. HYPERPARAMETER GRID  (5-fold CV AUC, n_rounds=1000)")
print("=" * 70)

configs = [
    ("default",                    dict()),
    ("refit=None",                 dict(refit_interval=None)),
    ("refit=50",                   dict(refit_interval=50)),
    ("refit=100",                  dict(refit_interval=100)),
    ("auto_expand=False",          dict(auto_expand=False)),
    ("hvrt_msl=10",                dict(hvrt_min_samples_leaf=10)),
    ("hvrt_msl=20",                dict(hvrt_min_samples_leaf=20)),
    ("hvrt_msl=10,refit=None",     dict(hvrt_min_samples_leaf=10, refit_interval=None)),
    ("max_depth=3",                dict(max_depth=3)),
    ("reduce=0.5",                 dict(reduce_ratio=0.5)),
    ("reduce=0.9",                 dict(reduce_ratio=0.9)),
    ("expand=0.3",                 dict(expand_ratio=0.3)),
    ("balanced",                   dict(class_weight="balanced")),
    ("balanced,refit=None",        dict(class_weight="balanced", refit_interval=None)),
    ("lr=0.1,depth=3",             dict(learning_rate=0.1, max_depth=3)),
    ("lr=0.1,depth=3,refit=None",  dict(learning_rate=0.1, max_depth=3, refit_interval=None)),
    ("msl=20,refit=None,noexp",    dict(hvrt_min_samples_leaf=20, refit_interval=None, auto_expand=False)),
]

print(f"  {'Config':<30}  {'AUC':>7}  {'Std':>6}")
print("  " + "-" * 48)
results = {}
for label, kw in configs:
    auc, std = cv_auc(GeoXGBClassifier, n_rounds=1000, random_state=42, **kw)
    results[label] = auc
    marker = " <--" if auc == max(results.values()) else ""
    print(f"  {label:<30}  {auc:.4f}  {std:.4f}{marker}")
print()

best_label = max(results, key=results.get)
best_kw = dict(configs)[best_label]
print(f"  Best config: '{best_label}'  AUC={results[best_label]:.4f}")
print()

# ---------------------------------------------------------------------------
# 4. CHECKPOINT SWEEP: rounds vs AUC for top configs
# ---------------------------------------------------------------------------
print("=" * 70)
print("  4. CHECKPOINT SWEEP  (5-fold CV AUC vs n_rounds)")
print("=" * 70)

sweep_configs = [
    ("default",         dict()),
    ("refit=None",      dict(refit_interval=None)),
    ("hvrt_msl=10,rNone", dict(hvrt_min_samples_leaf=10, refit_interval=None)),
    best_label,
]
# Deduplicate preserving order
seen = set()
sweep_configs_dedup = []
for item in sweep_configs:
    if isinstance(item, str):
        key = item
        kw = dict(configs)[item]
    else:
        key, kw = item
    if key not in seen:
        seen.add(key)
        sweep_configs_dedup.append((key, kw))

CHECKPOINTS = [50, 100, 200, 500, 1000, 2000]
COL = 8
print(f"  {'Config':<30}" + "".join(f"  {'r='+str(r):>{COL}}" for r in CHECKPOINTS))
print("  " + "-" * (30 + len(CHECKPOINTS) * (COL + 2) + 4))
for label, kw in sweep_configs_dedup:
    row = f"  {label:<30}"
    for r in CHECKPOINTS:
        auc, _ = cv_auc(GeoXGBClassifier, n_rounds=r, random_state=42, **kw)
        row += f"  {auc:>{COL}.4f}"
    print(row)
print()

# ---------------------------------------------------------------------------
# 5. RESIDUAL ANALYSIS
# ---------------------------------------------------------------------------
print("=" * 70)
print("  5. RESIDUAL ANALYSIS  (out-of-fold probabilities, default config)")
print("=" * 70)

oof_prob = np.zeros(n)
for tr_idx2, te_idx2 in FOLDS:
    m = GeoXGBClassifier(n_rounds=1000, random_state=42)
    m.fit(X[tr_idx2], y[tr_idx2])
    oof_prob[te_idx2] = m.predict_proba(X[te_idx2])[:, 1]

errors = np.abs(oof_prob - y)
print(f"  Overall OOF AUC  : {roc_auc_score(y, oof_prob):.4f}")
print(f"  Mean abs error   : {errors.mean():.4f}")
print(f"  High-error (>0.6): {(errors > 0.6).sum()} samples")
print()

# Error distribution by class
for cls, label in [(0, "Absence"), (1, "Presence")]:
    mask = y == cls
    print(f"  {label} (n={mask.sum()}): "
          f"mean_err={errors[mask].mean():.3f}  "
          f"max_err={errors[mask].max():.3f}  "
          f"p75_err={np.percentile(errors[mask], 75):.3f}")
print()

# Hardest samples
print(f"  Top-15 hardest samples (highest |prob - true_label|):")
hard_idx = np.argsort(errors)[::-1][:15]
print(f"  {'idx':>5}  {'true':>5}  {'prob':>6}  {'error':>6}  feature values (first 5 features)")
for idx in hard_idx:
    fv = "  ".join(f"{X[idx, j]:6.1f}" for j in range(min(5, d)))
    print(f"  {idx:>5}  {y[idx]:>5}  {oof_prob[idx]:>6.3f}  {errors[idx]:>6.3f}  {fv}")
print()

# Calibration check: predicted probability distribution
for bucket, lo, hi in [("p<0.2", 0, 0.2), ("0.2-0.4", 0.2, 0.4),
                        ("0.4-0.6", 0.4, 0.6), ("0.6-0.8", 0.6, 0.8), ("p>0.8", 0.8, 1.01)]:
    mask = (oof_prob >= lo) & (oof_prob < hi)
    if mask.sum() == 0:
        continue
    actual_rate = y[mask].mean()
    print(f"  Bucket {bucket:<10}: n={mask.sum():>3}  actual_rate={actual_rate:.3f}")
print()

# ---------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
print("=" * 70)
print("  6. FEATURE IMPORTANCE  (boosting vs HVRT geometry)")
print("=" * 70)

_mi = GeoXGBClassifier(n_rounds=1000, random_state=42)
_mi.fit(X, y)

boost_imp = _mi.feature_importances(FEATURES)
geo_imp_list = _mi.partition_feature_importances(FEATURES)
geo_imp = geo_imp_list[0]["importances"]   # initial partition importance

print(f"  {'Feature':<28}  {'Boosting':>9}  {'Geometry':>9}  {'Ratio B/G':>10}")
print("  " + "-" * 62)
for fn in FEATURES:
    bi = boost_imp.get(fn, 0.0)
    gi = geo_imp.get(fn, 0.0)
    ratio = bi / (gi + 1e-9)
    print(f"  {fn:<28}  {bi:>9.4f}  {gi:>9.4f}  {ratio:>10.2f}x")
print()

# ---------------------------------------------------------------------------
# 7. CLASS-WEIGHT INVESTIGATION
# ---------------------------------------------------------------------------
print("=" * 70)
print("  7. CLASS-WEIGHT INVESTIGATION")
print("=" * 70)
print(f"  Class imbalance: {(1-pos_rate)*100:.1f}% Absence vs {pos_rate*100:.1f}% Presence")
print()

cw_configs = [
    ("none",                    dict()),
    ("balanced",                dict(class_weight="balanced")),
    ("balanced,refit=None",     dict(class_weight="balanced", refit_interval=None)),
    ("none,refit=None",         dict(refit_interval=None)),
]
ROUNDS_CW = [100, 500, 1000]
print(f"  {'Config':<30}" + "".join(f"  {'r='+str(r):>8}" for r in ROUNDS_CW))
print("  " + "-" * (30 + len(ROUNDS_CW) * 10 + 4))
for label, kw in cw_configs:
    row = f"  {label:<30}"
    for r in ROUNDS_CW:
        a, _ = cv_auc(GeoXGBClassifier, n_rounds=r, random_state=42, **kw)
        row += f"  {a:>8.4f}"
    print(row)
print()

# ---------------------------------------------------------------------------
# 8. FAILURE AUDIT -- consistently misclassified samples
# ---------------------------------------------------------------------------
print("=" * 70)
print("  8. FAILURE AUDIT  (samples misclassified in majority of CV folds)")
print("=" * 70)

fold_preds = np.zeros((n, 5))
for fi, (tr_idx3, te_idx3) in enumerate(FOLDS):
    m = GeoXGBClassifier(n_rounds=1000, random_state=42)
    m.fit(X[tr_idx3], y[tr_idx3])
    prob = m.predict_proba(X[te_idx3])[:, 1]
    fold_preds[te_idx3, fi] = prob

mean_prob = fold_preds.mean(axis=1)
binary_pred = (mean_prob > 0.5).astype(int)
misclassified = binary_pred != y

print(f"  Consistently misclassified (mean prob wrong side): {misclassified.sum()} / {n}")
print()
if misclassified.sum() > 0:
    print(f"  {'idx':>5}  {'true':>5}  {'mean_p':>7}  " +
          "  ".join(f"fold{i+1}" for i in range(5)))
    mc_idx = np.where(misclassified)[0]
    mc_idx = mc_idx[np.argsort(np.abs(mean_prob[mc_idx] - y[mc_idx]))[::-1]][:20]
    for idx in mc_idx:
        fprobs = "  ".join(f"{fold_preds[idx, fi]:.3f}" for fi in range(5))
        print(f"  {idx:>5}  {y[idx]:>5}  {mean_prob[idx]:>7.3f}  {fprobs}")
    print()
    print(f"  Feature values of consistently misclassified samples:")
    print(f"  {'idx':>5}  {'true':>5}  " +
          "  ".join(f"{fn[:8]:>9}" for fn in FEATURES))
    for idx in mc_idx[:10]:
        fv = "  ".join(f"{X[idx, j]:>9.1f}" for j in range(d))
        print(f"  {idx:>5}  {y[idx]:>5}  {fv}")
print()

# ---------------------------------------------------------------------------
# 9. XGBOOST BASELINE
# ---------------------------------------------------------------------------
print("=" * 70)
print("  9. XGBOOST BASELINE COMPARISON")
print("=" * 70)

xgb_configs = [
    ("xgb default",              dict(n_estimators=1000, learning_rate=0.1, max_depth=6)),
    ("xgb depth=4",              dict(n_estimators=1000, learning_rate=0.1, max_depth=4)),
    ("xgb depth=4,subsample=0.8",dict(n_estimators=1000, learning_rate=0.1, max_depth=4, subsample=0.8)),
    ("xgb scale_pos",            dict(n_estimators=1000, learning_rate=0.1, max_depth=4,
                                      scale_pos_weight=(1-pos_rate)/pos_rate)),
]

print(f"  {'Config':<35}  {'AUC':>7}  {'Std':>6}")
print("  " + "-" * 52)
for label, kw in xgb_configs:
    scores = []
    for tr_idx4, te_idx4 in FOLDS:
        xm = XGBClassifier(tree_method="hist", verbosity=0, random_state=42, **kw)
        xm.fit(X[tr_idx4], y[tr_idx4])
        scores.append(roc_auc_score(y[te_idx4], xm.predict_proba(X[te_idx4])[:, 1]))
    print(f"  {label:<35}  {np.mean(scores):.4f}  {np.std(scores):.4f}")
print()

print(f"  GeoXGB best ('{best_label}')  : {results[best_label]:.4f}")
print(f"  GeoXGB default               : {results['default']:.4f}")
print()

print("=== DONE ===")
