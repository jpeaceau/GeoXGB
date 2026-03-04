"""
Interpretability Demo: GeoXGB vs EBM
=====================================

Compares what GeoXGB's cooperation geometry reveals vs EBM pairwise terms.
HPO pattern: search on a 3000-sample subsample, then refit on full training
data — fast and representative.

GeoXGB unique capabilities demonstrated:
  1. LOCAL cooperation matrices — per-prediction, not a global average.
  2. THREE-WAY cooperation tensor — how feature k modulates pair (i, j).
  3. Architecture-native cooperation SCORES — grounded in partition geometry.
  4. Local vs global variance: how much SHAP/EBM global terms hide.

Dataset: California Housing (sklearn, 8 features, ~20k samples).

Usage:
    python benchmarks/interpretability_demo.py

Requirements: pip install optuna interpret matplotlib
"""

import sys
import warnings
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def _p(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)

# ── GeoXGB ───────────────────────────────────────────────────────────────────
from geoxgb import GeoXGBRegressor
from geoxgb.optimizer import GeoXGBOptimizer

# ── EBM ──────────────────────────────────────────────────────────────────────
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    _EBM_AVAILABLE = True
except ImportError:
    _EBM_AVAILABLE = False
    _p("[WARN] interpret not installed — EBM section skipped.  pip install interpret")

# ── Optuna (for EBM HPO) ─────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    _p("[WARN] optuna not installed — HPO sections skipped.  pip install optuna")

# ── matplotlib ───────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _PLOT = True
except ImportError:
    _PLOT = False

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
_p("\n=== California Housing Dataset ===")
housing = fetch_california_housing()
X_raw, y = housing.data, housing.target
feature_names = list(housing.feature_names)
d = len(feature_names)
_p(f"  n={len(X_raw)}, d={d}, features: {feature_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)
_p(f"  train={len(X_train)}, test={len(X_test)}")
_p(f"  NOTE: raw features used (GeoXGB normalises internally via HVRT z-scores;")
_p(f"        pre-scaling would strip natural feature scale from the cooperation geometry)")
FT = ["continuous"] * d   # feature_types required for cooperation API

# HPO subsample: 3000 samples from training set (fast, representative)
rng_sub = np.random.default_rng(42)
sub_idx = rng_sub.choice(len(X_train), size=3000, replace=False)
X_hpo, y_hpo = X_train[sub_idx], y_train[sub_idx]
_p(f"  HPO subsample: {len(X_hpo)} samples (best params refitted on full train)")

# ─────────────────────────────────────────────────────────────────────────────
# 1. GeoXGB baseline
# ─────────────────────────────────────────────────────────────────────────────
_p("\n=== GeoXGB — baseline (default params) ===")
geo_base = GeoXGBRegressor(
    partitioner='pyramid_hart', n_rounds=500,
    max_depth=3, learning_rate=0.02, y_weight=0.25,
    refit_interval=50, random_state=42,
)
geo_base.fit(X_train, y_train, feature_types=FT)
geo_base_r2 = r2_score(y_test, geo_base.predict(X_test))
_p(f"  R2 = {geo_base_r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GeoXGB HPO via GeoXGBOptimizer
# ─────────────────────────────────────────────────────────────────────────────
_p("\n=== GeoXGB — HPO (GeoXGBOptimizer, 20 trials on 3k subsample) ===")
geo_opt = None
geo_hpo_r2 = None

if _OPTUNA_AVAILABLE:
    opt = GeoXGBOptimizer(
        task='regression',
        n_trials=20,
        cv=3,
        random_state=42,
        verbose=False,
    )
    # Search on subsample; fix n_rounds=1000 so HPO focuses on structural params
    opt.fit(X_hpo, y_hpo, n_rounds=1000)
    _p(f"  Best params: {opt.best_params_}")
    _p(f"  Best CV R2 (on subsample): {opt.best_score_:.4f}")

    # Refit with best params on FULL training set; feature_types for cooperation API
    geo_opt = GeoXGBRegressor(**opt.best_params_, random_state=42)
    geo_opt.fit(X_train, y_train, feature_types=FT)
    geo_hpo_r2 = r2_score(y_test, geo_opt.predict(X_test))
    _p(f"  Test R2 (full train refit): {geo_hpo_r2:.4f}  "
       f"(baseline: {geo_base_r2:.4f}, delta: {geo_hpo_r2 - geo_base_r2:+.4f})")
else:
    _p("  [skipped — optuna not installed]")

# Use best available GeoXGB model for interpretability analysis
geo     = geo_opt if geo_opt is not None else geo_base
geo_r2  = geo_hpo_r2 if geo_hpo_r2 is not None else geo_base_r2

# ─────────────────────────────────────────────────────────────────────────────
# 3. EBM baseline
# (EBM HPO omitted — each EBM fit is O(n * d^2 * rounds), making HPO
#  impractically slow on large datasets. Default params are used instead.)
# ─────────────────────────────────────────────────────────────────────────────
ebm_base_r2 = None
ebm_best    = None

if _EBM_AVAILABLE:
    _p("\n=== EBM — baseline (default params) ===")
    _p("  (EBM HPO skipped — EBM fits are O(n*d^2*rounds), too slow for HPO at scale)")
    ebm_base = ExplainableBoostingRegressor(random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ebm_base.fit(X_train, y_train)
    ebm_base_r2 = r2_score(y_test, ebm_base.predict(X_test))
    ebm_best = ebm_base
    _p(f"  R2 = {ebm_base_r2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Interpretability analysis on tuned GeoXGB model
# ─────────────────────────────────────────────────────────────────────────────
_p("\n" + "=" * 60)
_p("INTERPRETABILITY ANALYSIS — tuned GeoXGB model")
_p("=" * 60)

# ── 4a. Boosting importances ─────────────────────────────────────────────────
_p("\n-- Boosting feature importances --")
fi = geo.feature_importances(feature_names)
for name, imp in list(fi.items())[:5]:
    _p(f"  {name:<12} {imp:.4f}")

# ── 4b. Global cooperation matrix ────────────────────────────────────────────
_p("\n-- Global cooperation matrix (partition-weighted avg over training) --")
coop_all = geo.cooperation_matrix(X_test, feature_names)
G        = coop_all["global_matrix"]
mats     = coop_all["matrices"]

_p("  Strongest POSITIVE cooperation pairs (corr > 0.25):")
pairs_pos = [(feature_names[i], feature_names[j], G[i, j])
             for i in range(d) for j in range(i + 1, d) if G[i, j] > 0.25]
for a, b, v in sorted(pairs_pos, key=lambda x: -x[2])[:5]:
    _p(f"    ({a}, {b}): {v:+.3f}")
if not pairs_pos:
    _p("    (none above 0.25)")

_p("  Strongest NEGATIVE cooperation pairs (corr < -0.25):")
pairs_neg = [(feature_names[i], feature_names[j], G[i, j])
             for i in range(d) for j in range(i + 1, d) if G[i, j] < -0.25]
for a, b, v in sorted(pairs_neg, key=lambda x: x[2])[:5]:
    _p(f"    ({a}, {b}): {v:+.3f}")
if not pairs_neg:
    _p("    (none below -0.25)")

# ── 4c. LOCAL cooperation — the key advantage ────────────────────────────────
_p("\n-- LOCAL cooperation: per-prediction variance vs global average --")
_p("  (Global averages hide this — EBM pairwise terms cannot show it.)")

pair_stds = []
for i in range(d):
    for j in range(i + 1, d):
        local_vals = mats[:, i, j]
        pair_stds.append((feature_names[i], feature_names[j],
                          local_vals.std(), G[i, j],
                          local_vals.min(), local_vals.max()))
pair_stds.sort(key=lambda x: -x[2])

_p("  Top-3 pairs by local cooperation variance:")
_p(f"  {'Pair':<32} {'Global':>8}  {'Std':>6}  Local range")
for a, b, std_v, glob_v, lo, hi in pair_stds[:3]:
    _p(f"  ({a}, {b}){'':<{30-len(a)-len(b)}} {glob_v:>+8.3f}  {std_v:>6.3f}  [{lo:+.3f}, {hi:+.3f}]")

# ── 4d. Cooperation score ────────────────────────────────────────────────────
_p("\n-- Cooperation score (PyramidHART: A = |S| - ||z||_1 <= 0) --")
scores = geo.cooperation_score(X_test)
_p(f"  Range: [{scores.min():.3f}, {scores.max():.3f}]  (0 = all features same sign)")
_p(f"  Mean:  {scores.mean():.3f}  |  Std: {scores.std():.3f}")
residuals = np.abs(y_test - geo.predict(X_test))
corr = float(np.corrcoef(scores, residuals)[0, 1])
_p(f"  Correlation(score, |residual|) = {corr:+.3f}")

# ── 4e. Three-way cooperation tensor ─────────────────────────────────────────
_p("\n-- Three-way cooperation tensor (no equivalent in EBM or SHAP) --")
t_result = geo.cooperation_tensor(X_test[:500], feature_names)
GT       = t_result["global_tensor"]
T_local  = t_result["tensor"]

triples = [(feature_names[i], feature_names[j], feature_names[k], GT[i, j, k])
           for i in range(d) for j in range(i+1, d) for k in range(j+1, d)]
triples.sort(key=lambda x: abs(x[3]), reverse=True)
_p("  Top-5 global 3-way interactions |T[i, j, k]| (i < j < k):")
for a, b, c, v in triples[:5]:
    _p(f"    ({a}, {b}, {c}): {v:+.4f}")

top = triples[0]
i0, i1, i2 = (feature_names.index(top[x]) for x in range(3))
local_3way = T_local[:, i0, i1, i2]
_p(f"\n  Locality of strongest interaction ({top[0]}, {top[1]}, {top[2]}):")
_p(f"    Global  = {GT[i0,i1,i2]:+.4f}")
_p(f"    Local   = [{local_3way.min():+.4f}, {local_3way.max():+.4f}]")
_p(f"    Std     = {local_3way.std():.4f}"
   f"  ({local_3way.std() / (abs(GT[i0,i1,i2]) + 1e-6):.1f}x global magnitude)")

# ── 4f. EBM pairwise terms comparison ────────────────────────────────────────
if _EBM_AVAILABLE and ebm_best is not None:  # noqa: SIM102
    _p("\n-- EBM pairwise terms (for comparison) --")
    pair_terms = [
        (ebm_best.term_names_[i], ebm_best.term_importances()[i])
        for i in range(len(ebm_best.term_names_))
        if ' x ' in str(ebm_best.term_names_[i])
    ]
    pair_terms.sort(key=lambda x: -x[1])
    if pair_terms:
        _p(f"  {len(pair_terms)} pairwise terms learned by EBM:")
        for name, imp in pair_terms[:5]:
            _p(f"    {str(name):<35} importance={imp:.4f}")
    else:
        _p("  EBM pruned all pairwise terms (none survived regularization).")
    _p("  NOTE: These are GLOBAL averages — no per-prediction local values.")
    _p("        GeoXGB cooperation_matrix() gives a different matrix per sample.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Accuracy + capability comparison
# ─────────────────────────────────────────────────────────────────────────────
_p("\n" + "=" * 60)
_p("ACCURACY COMPARISON")
_p("=" * 60)
rows = [("GeoXGB baseline",              geo_base_r2)]
if geo_hpo_r2  is not None: rows.append(("GeoXGB HPO (20 trials)",       geo_hpo_r2))
if ebm_base_r2 is not None: rows.append(("EBM baseline (default params)", ebm_base_r2))
for label, r2 in rows:
    _p(f"  {label:<30}  R2 = {r2:.4f}")

_p("\n" + "=" * 60)
_p("CAPABILITY COMPARISON")
_p("=" * 60)
_p(f"  {'Capability':<47} {'GeoXGB':>7} {'EBM':>6}")
_p("  " + "-" * 60)
for name, g, e in [
    ("Per-prediction LOCAL pairwise cooperation",    True,  False),
    ("Global pairwise cooperation summary",          True,  True),
    ("THREE-WAY interaction tensor",                 True,  False),
    ("Architecture-specific cooperation score",      True,  False),
    ("Partition geometry (sizes, noise modulation)", True,  False),
    ("Human-readable global main effects",           False, True),
    ("Additive feature attribution",                 False, True),
]:
    _p(f"  {name:<47} {'Yes':>7} {'Yes' if e else '—':>6}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Cooperation matrix heatmaps
# ─────────────────────────────────────────────────────────────────────────────
if _PLOT:
    min_idx = int(scores.argmin())
    max_idx = int(scores.argmax())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("GeoXGB Cooperation Matrices — California Housing (HPO model)",
                 fontsize=12)

    for ax, (mat, title) in zip(axes, [
        (G,             "Global (all partitions)"),
        (mats[min_idx], f"Local: sample {min_idx} (score={scores[min_idx]:.2f})"),
        (mats[max_idx], f"Local: sample {max_idx} (score={scores[max_idx]:.2f})"),
    ]):
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(d))
        ax.set_yticks(range(d))
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(feature_names, fontsize=7)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out = "benchmarks/cooperation_matrix_demo.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    _p(f"\n[Plot saved to {out}]")

_p("\nDone.")
