# GeoXGB — Geometry-Aware Gradient Boosting

GeoXGB replaces conventional subsampling and bootstrapping with geometry-aware
sample reduction and expansion powered by [HVRT](https://pypi.org/project/hvrt/)
(Hierarchical Variance-Retaining Transformer).

## Installation

```bash
pip install geoxgb
```

For hyperparameter optimisation via Optuna:

```bash
pip install "geoxgb[optimizer]"
```

Requires `hvrt >= 2.2.0`, `scikit-learn`, and `numpy`. Python >= 3.10.

## Quick Start

```python
from geoxgb import GeoXGBRegressor, GeoXGBClassifier

# Regression
reg = GeoXGBRegressor(n_rounds=1000, learning_rate=0.2)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Classification
clf = GeoXGBClassifier(n_rounds=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Pass feature types for mixed data
clf.fit(X_train, y_train, feature_types=["continuous", "categorical", ...])
```

## Key Features

- **Geometry-aware sampling** via HVRT's variance-retaining partitions
- **FPS reduction** — keeps geometrically diverse representatives
- **KDE expansion** — synthesises samples in sparse regions
- **Adaptive noise detection** — automatically backs off on noisy data
- **Multi-fit** — refits partitions on residuals every N rounds
- **No overfitting** — see [Why high `n_rounds` is safe](#why-high-n_rounds-is-safe)
- **Full interpretability** — feature importances, partition traces, sample provenance
- **Gardener** — post-hoc surgical editor: diagnose biased leaves and self-heal
- **GeoXGBOptimizer** — Optuna TPE hyperparameter search
- **Categorical support** — pass `feature_types` to handle mixed data natively
- **Class reweighting** — `class_weight` for imbalanced classification
- **Multiclass parallelism** — `n_jobs` for K-class one-vs-rest ensembles

## Why High `n_rounds` is Safe

Standard gradient boosting memorises: every tree sees the same N rows, so
adding rounds eventually overfits the training set.

GeoXGB cannot memorise. At every `refit_interval`, HVRT re-partitions the
residual landscape and FPS selects a fresh, geometrically diverse subset.
**No boosting tree ever trains on the same sample twice.** There is no fixed
training set to memorise, so train and val loss converge smoothly and continue
to improve with more rounds — the train–val gap remains small regardless of
`n_rounds`.

Practical consequences:

- More rounds is always beneficial (or neutral); it is never harmful.
- `convergence_tol` is a *compute budget* feature, not an anti-overfitting
  guard — use it to stop early once the gradient has genuinely plateaued.
- The default `n_rounds=1000` is a conservative starting point; tuning up
  to 2000–4000 rounds consistently yields further gains on large datasets.

## Parameters

### Shared (GeoXGBRegressor and GeoXGBClassifier)

| Parameter | Default | Description |
|---|---|---|
| `n_rounds` | 1000 | Number of boosting rounds |
| `learning_rate` | 0.2 | Shrinkage per tree |
| `max_depth` | 4 | Maximum depth of each weak learner |
| `min_samples_leaf` | 5 | Minimum samples per leaf |
| `n_partitions` | None | HVRT partition count (None = auto-tuned) |
| `hvrt_min_samples_leaf` | None | HVRT partition minimum leaf size (None = auto-tuned) |
| `reduce_ratio` | 0.7 | Fraction to keep via FPS |
| `expand_ratio` | 0.0 | Fraction to synthesise via KDE (0 = disabled) |
| `y_weight` | 0.5 | HVRT blend: 0 = unsupervised geometry, 1 = y-driven |
| `method` | `'fps'` | HVRT selection method |
| `variance_weighted` | True | Budget allocation by partition variance |
| `bandwidth` | `'auto'` | KDE bandwidth for expansion (`'auto'` = per-partition Scott's rule) |
| `generation_strategy` | `'epanechnikov'` | KDE sampling strategy (see HVRT docs) |
| `adaptive_bandwidth` | False | Scale KDE bandwidth by local partition density |
| `refit_interval` | 20 | Refit HVRT partitions every N rounds (None = off) |
| `auto_noise` | True | Auto-detect noise and modulate resampling |
| `auto_expand` | True | Auto-expand small datasets to `min_train_samples` |
| `min_train_samples` | 5000 | Target training-set size when `auto_expand=True` |
| `cache_geometry` | False | Reuse HVRT partition structure across refits |
| `feature_weights` | None | Per-feature scaling applied before HVRT sees X |
| `convergence_tol` | None | Stop early when gradient improvement < tol (compute budget only) |
| `n_jobs` | 1 | Parallel workers for multiclass one-vs-rest ensembles |
| `random_state` | 42 | |

### GeoXGBClassifier only

| Parameter | Default | Description |
|---|---|---|
| `class_weight` | None | `None`, `'balanced'`, or `{class: weight}` dict |

## Saving and Loading Models

```python
from geoxgb import load_model

# Save (strips large HVRT data arrays — file stays under 100 MB)
model.save("my_model.pkl")

# Load in a new process
model = load_model("my_model.pkl")
predictions = model.predict(X_test)
```

## Gardener — Post-Hoc Tree Surgery

`Gardener` wraps a fitted model and exposes manual editing tools plus
automatic self-healing:

```python
from geoxgb import Gardener

garden = Gardener(fitted_model)

# Automatic: detect biased leaves, correct, validate, commit only if better
result = garden.heal(X_train, y_train, X_val, y_val, strategy="surgery")
print(result["improvement"])   # AUC / R² delta vs baseline

# Manual tools
garden.adjust_leaf(tree_idx=5, leaf_id=3, delta=-0.02)
garden.prune(tree_idx=12, leaf_id=7)
garden.graft(X_targeted, residuals, n_rounds=10, learning_rate=0.05)
garden.rollback()              # undo last operation
garden.reset()                 # restore to original fitted state

# Derive feature weights from gradient vs geometry agreement
weights = garden.recommend_feature_weights(feature_names)
model2 = GeoXGBClassifier(feature_weights=list(weights.values()))
```

## GeoXGBOptimizer — Optuna HPO

```python
from geoxgb import GeoXGBOptimizer

opt = GeoXGBOptimizer(n_trials=50, cv=3, random_state=42)
opt.fit(X_train, y_train)

print(opt.best_params_)   # {'n_rounds': 1000, 'learning_rate': 0.2, ...}
print(opt.best_score_)    # best mean CV score (AUC or R²)

y_pred  = opt.predict(X_test)
y_proba = opt.predict_proba(X_test)   # classifier only

# Access the raw Optuna study for plots / analysis
opt.study_.best_trial
```

Trial 0 is always the v0.1.1 defaults — HPO is guaranteed to match or beat
the baseline. `fast=True` (default) accelerates trials via geometry caching
and convergence-based early stop, then refits the final model at full quality.

## Heterogeneity Detection

The boost/partition importance ratio is a heterogeneity surface map. When the
two importance axes diverge it is not a red flag — it is structural information
about each feature's local role:

| Ratio | Interpretation |
|---|---|
| `ratio >> 1` | Prediction driver — feature drives gradient updates within local regions but does not define them |
| `ratio << 1` | Heterogeneity axis — feature defines *where* different predictive relationships apply; lower predictive contribution within each region |
| `ratio ~= 1` | Universally informative — both structure-defining and predictive |

This operates at the **individual level**, not just population subgroups. Each
HVRT partition is a hyperplane-bounded local region in feature space. With
sufficient partitions these regions can be arbitrarily fine, approaching
individual-level neighbourhoods. `partition_tree_rules()` exposes the exact
conditions defining each individual's local region.

```python
boost = model.feature_importances(feature_names=names)
part  = model.partition_feature_importances(feature_names=names)

avg_part = {f: np.mean([e["importances"].get(f, 0) for e in part]) for f in names}
for f in names:
    ratio = boost[f] / (avg_part[f] + 1e-10)
    print(f"{f}: ratio={ratio:.2f}")
# ratio << 1  =>  heterogeneity axis (defines local structure)
# ratio >> 1  =>  prediction driver (gradient-dominant within regions)
```

Validated across three synthetic scenarios in
[`benchmarks/heterogeneity_detection_test.py`](benchmarks/heterogeneity_detection_test.py):

1. **Regime indicator**: the feature that determines *which* local predictive
   relationship applies consistently has a lower ratio than within-regime
   predictors, regardless of whether it directly enters the prediction formula.

2. **Interaction moderator**: the ratio ordering (moderator < predictor) holds
   for sign-flip interactions. XGBoost tree importance conflates structure and
   prediction roles into a single score; the boost/partition split separates them.

3. **Complementary roles**: among two strong predictors, HVRT allocates one to
   anchor partition geometry (lower ratio) and the other to gradient-driven
   prediction (higher ratio). Role assignment is emergent — the divergence
   reveals local structure, not model error.

## Interpretability

```python
from geoxgb.report import model_report, print_report

# All-in-one structured report
print_report(model_report(model, X_test, y_test, feature_names=names))

# Individual report sections
from geoxgb.report import (
    noise_report,       # data quality assessment
    provenance_report,  # where did the training samples come from?
    importance_report,  # boosting vs partition feature importance
    partition_report,   # HVRT partition structure at a given round
    evolution_report,   # how geometry changed across refits
    validation_report,  # PASS/FAIL checks against known ground truth
    compare_report,     # head-to-head comparison with a baseline
)

# Raw model API
model.feature_importances(feature_names)           # boosting importance
model.partition_feature_importances(feature_names) # geometric importance
model.partition_trace()                             # full partition history
model.partition_tree_rules(round_idx=0)             # human-readable rules
model.sample_provenance()                           # reduction/expansion counts
model.noise_estimate()                              # 1.0=clean, 0.0=pure noise
```

## Imbalanced Classification

Use `class_weight='balanced'` to upweight the minority class in gradient
updates. This stacks with HVRT's geometric diversity preservation.

```python
clf = GeoXGBClassifier(
    class_weight='balanced',
    auto_noise=False,   # recommended for severe imbalance (< 5% minority)
)
```

## Large-Scale Datasets

For datasets with many thousands of samples, HVRT refitting at each
`refit_interval` dominates wall time. Enable geometry caching to reuse the
initial partition structure and reduce HVRT.fit() calls from
`n_rounds / refit_interval` down to 1:

```python
model = GeoXGBRegressor(
    cache_geometry=True,   # reuse HVRT partition structure across refits
    n_rounds=4000,         # safe to go high — no overfitting risk
)
```

For multiclass problems, parallelise the K one-vs-rest ensembles:

```python
clf = GeoXGBClassifier(n_jobs=4)   # ~4x speedup for 5-class problems
```

## Benchmarks

GeoXGB wins all 10 out of 10 head-to-head comparisons against XGBoost across
standard benchmarks (Friedman #1/#2, classification, sparse high-dimensional,
and noisy classification datasets), with and without Optuna HPO. On the hardest
case — a 40-feature dataset with 80% irrelevant features — Optuna selects
`y_weight=0.9`, making HVRT y-driven and cutting through feature-space noise
for a +0.025 R² margin (previously −0.004 without `y_weight` in the search space).

See [`benchmarks/PERFORMANCE_REPORT.md`](benchmarks/PERFORMANCE_REPORT.md)
for the full results including per-dataset AUC / R² tables, error complementarity
analysis, and interpretability walkthroughs.

## Causal Inference

GeoXGB's geometry-aware resampling makes it a strong base estimator for CATE
and ITE tasks. HVRT partitions covariate space into locally homogeneous
regions that naturally align with treatment-effect subgroups; `auto_expand`
prevents information collapse in sparse T=0/T=1 sub-populations.

### ITE metalearner usage

GeoXGB drops into any standard metalearner architecture:

```python
import numpy as np
from sklearn.model_selection import train_test_split

X_tr, X_te, T_tr, T_te, Y_tr, Y_te = train_test_split(X, T, Y, test_size=0.25)

# T-learner
m0 = GeoXGBRegressor(); m0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
m1 = GeoXGBRegressor(); m1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
tau_hat = m1.predict(X_te) - m0.predict(X_te)

# S-learner — best on smooth/linear τ; HVRT treats T as a first-class
# geometry axis alongside X, recovering T×X interactions via tree depth
XT_tr = np.column_stack([X_tr, T_tr])
m = GeoXGBRegressor(); m.fit(XT_tr, Y_tr)
tau_hat = (m.predict(np.column_stack([X_te, np.ones(len(X_te))]))
         - m.predict(np.column_stack([X_te, np.zeros(len(X_te))])))
```

PEHE benchmark on randomised trials (lower is better):

| τ(x) type | GeoXGB | XGBoost | Honest R-forest¹ |
|---|---|---|---|
| Linear (2X₁ + 1) | **0.180** | 0.207 | 0.247 |
| Nonlinear (2·sin(X₁π) + X₂²) | **0.408** | 0.608 | 0.796 |

¹ 2-fold cross-fitted R-forest with sample-weighted RandomForestRegressor —
the functional core of GRF. `econml` CausalForestDML unavailable on Python 3.14.

### Mediation fingerprint

The boost/partition importance ratio surfaces causal structure without a
separate statistical test. Features that are causally *upstream* of Y (i.e.
X where part of the effect passes through a mediator M) have
`boost_imp >> partition_imp` — the gradient signal recognises X as important
even when HVRT geometry anchors on M:

```python
part  = model.partition_feature_importances(feature_names=names)
boost = model.feature_importances(feature_names=names)

avg_part = {f: np.mean([e["importances"].get(f, 0) for e in part])
            for f in names}
for f in names:
    ratio = boost[f] / (avg_part[f] + 1e-10)
    print(f"{f}: boost/partition = {ratio:.2f}")
# Causally upstream features show ratio >> 1
# Mediator features show ratio < 1 (geometry anchors on them)
```

### Doubly-robust ATE

For average treatment effect estimation under confounding, use GeoXGB as the
outcome model in a doubly-robust (DR) pipeline — its nonlinear surface quality
reduces IPW residuals and tightens the DR correction:

```python
from sklearn.linear_model import LogisticRegression

prop = LogisticRegression().fit(X_tr, T_tr)
pi   = np.clip(prop.predict_proba(X_te)[:, 1], 0.05, 0.95)
mu0, mu1 = m0.predict(X_te), m1.predict(X_te)

dr_ate = (mu1 - mu0
          + T_te * (Y_te - mu1) / pi
          - (1 - T_te) * (Y_te - mu0) / (1 - pi)).mean()
```

See [`notebooks/geoxgb_causal_analysis.ipynb`](notebooks/geoxgb_causal_analysis.ipynb)
for the full analysis: mediators, colliders, CATE, ITE metalearners, and ATE.

### When to use AutoITE instead

If you have **panel / time-series data** — repeated observations per entity
over time — consider [AutoITE (geo branch)](https://github.com/jpeaceau/AutoITE/tree/geo)
instead. AutoITE is purpose-built for ITE estimation from longitudinal data,
where the temporal dimension provides a richer identification strategy than
cross-sectional metalearners.

**Decision rule:**

| Data available | Recommended tool |
|---|---|
| Repeated observations per entity (panel / time-series) | [AutoITE geo branch](https://github.com/jpeaceau/AutoITE/tree/geo) |
| Cross-sectional data only | GeoXGB + metalearner (T/S/X/DR-learner) |

## License

MIT
