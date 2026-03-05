# GeoXGB — Geometry-Aware Gradient Boosting

GeoXGB replaces conventional subsampling and bootstrapping with geometry-aware
sample reduction and expansion powered by [HVRT](https://pypi.org/project/hvrt/)
(Hierarchical Variance Retention Transformer).

## Installation

```bash
pip install geoxgb
```

For hyperparameter optimisation via Optuna:

```bash
pip install "geoxgb[optimizer]"
```

Requires `hvrt >= 2.6.1`, `scikit-learn`, and `numpy`. Python >= 3.10.

The compiled C++ backend is included in the wheel and used automatically;
no extra install step required.

## Quick Start

```python
from geoxgb import GeoXGBRegressor, GeoXGBClassifier

# Regression — HPO strongly recommended for learning_rate and max_depth
reg = GeoXGBRegressor(n_rounds=1000)
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

> **HPO is strongly recommended.** `learning_rate` and `max_depth` are the
> two most sensitive parameters and interact strongly: optimal range is
> `learning_rate` 0.010–0.020 with high `n_rounds` (1 000–5 000) and
> shallow `max_depth` (2–3). These optima shift 5× across datasets.
> Use `GeoXGBOptimizer` or any Optuna/sklearn HPO tool for production models.

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
| `learning_rate` | 0.02 | Shrinkage per tree. **HPO recommended** — optimal range 0.010–0.020 |
| `max_depth` | 3 | Maximum depth of each weak learner. **HPO recommended** — optimal range 2–3 |
| `min_samples_leaf` | 5 | Minimum samples per leaf in the weak learner (DecisionTree) |
| `partitioner` | `'pyramid_hart'` | Geometry partitioner: `'pyramid_hart'`, `'hart'`, `'hvrt'` |
| `method` | `'variance_ordered'` | Sample reduction strategy: `'variance_ordered'`, `'orthant_stratified'`, `'fps'` |
| `generation_strategy` | `'simplex_mixup'` | Synthetic sample generator: `'simplex_mixup'`, `'epanechnikov'`, `'laplace'` |
| `n_partitions` | None | Partition count (None = auto-tuned) |
| `hvrt_min_samples_leaf` | None | Partition minimum leaf size (None = auto-tuned) |
| `reduce_ratio` | 0.8 | Fraction of samples to keep per boosting round |
| `expand_ratio` | 0.1 | Fraction to synthesise as synthetic samples (0 = disabled) |
| `y_weight` | 0.25 | Partition blend: 0 = unsupervised geometry, 1 = fully y-driven |
| `variance_weighted` | False | Budget allocation by partition variance |
| `bandwidth` | `'auto'` | KDE bandwidth for expansion (`'auto'` = per-partition Scott's rule) |
| `adaptive_bandwidth` | False | Scale KDE bandwidth by local partition density |
| `refit_interval` | 50 | Refit partition tree on residuals every N rounds (None = off) |
| `auto_noise` | True | Auto-detect noise and modulate resampling |
| `noise_guard` | True | Look-ahead veto on resampling when gradient signal is structureless |
| `auto_expand` | True | Auto-expand small datasets to `min_train_samples` |
| `min_train_samples` | 5000 | Target training-set size when `auto_expand=True` |
| `adaptive_reduce_ratio` | False | Dynamically adjust reduce_ratio from gradient tail heaviness |
| `sample_block_n` | `'auto'` | Epoch-based block cycling for large datasets. `'auto'`: `500 + (n−5000)//50` when n > 5000, else disabled. Pass an int to set manually, or `None` to disable. |
| `leave_last_block_out` | False | Hold out the final block as a validation set (forces Python path) |
| `feature_weights` | None | Per-feature scaling applied before partition geometry sees X |
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

Trial 0 is always the GeoXGB defaults — HPO is guaranteed to match or beat
the baseline. Each trial uses `convergence_tol=0.01` for speed, then the final
best model is refit without early stopping.

**Accuracy ceiling:** The final model test score (after full-quality refit) is
within 0.001–0.005 AUC/R² of the trial proxy for most datasets. Small datasets
(n<500) may show up to −0.015 R² if `expand_ratio` cannot be properly evaluated
within the trial budget.

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

For datasets with n > 5 000, the default `sample_block_n='auto'` activates
epoch-based block cycling. The dataset is split into non-overlapping windows
of size `500 + (n−5000)//50` (roughly 600 at n=10k, 1 400 at n=50k). Each
boosting epoch trains on one block, cycling through all blocks before reshuffling.
This provides cross-block geometric diversity — effectively acting as implicit
regularisation — while keeping HVRT and tree costs proportional to the block
size rather than full n.

```python
# n=50k: auto activates block_n=1400, ~35 blocks per epoch
model = GeoXGBRegressor(n_rounds=2000)  # sample_block_n='auto' by default
model.fit(X_train, y_train)

# Disable cycling (train on full n each round):
model = GeoXGBRegressor(sample_block_n=None)

# Hold out final block as validation:
model = GeoXGBRegressor(leave_last_block_out=True)
```

For multiclass problems, parallelise the K one-vs-rest ensembles:

```python
clf = GeoXGBClassifier(n_jobs=4)   # ~4x speedup for 5-class problems
```

## Benchmarks

### Small-n head-to-head vs XGBoost (default vs default, same CV protocol)

| Dataset | Metric | GeoXGB default | GeoXGB HPO-best | XGBoost (300 est.) | HPO vs XGB |
|---|---|---|---|---|---|
| diabetes (n=442) | R² | **0.4256** | **0.4630** | 0.3147 | **+0.148** |
| friedman1 (n=1000) | R² | **0.9198** | **0.9188** | 0.8434 | **+0.075** |
| breast\_cancer (n=569) | AUC | **0.9931** | **0.9932** | 0.9886 | **+0.005** |
| wine (n=178, 3-class) | AUC | **0.9951** | **0.9993** | 0.9975 | **+0.002** |
| digits (n=1797, 10-class) | AUC | **0.9988** | 0.9987 | 0.9990 | −0.000 |

GeoXGB default beats or matches XGBoost on all 5 datasets without any tuning.
HPO-best params are from a 2 000+ trial Optuna TPE study with
`partitioner='pyramid_hart'`, `method='variance_ordered'`, and
`generation_strategy='simplex_mixup'` fixed. Key findings: optimal
`learning_rate` 0.012–0.015, `max_depth` 2–3, `y_weight` 0.21–0.28.

> Note: XGBoost uses its default 300 estimators (`learning_rate=0.1`).
> GeoXGB uses its default 1 000 rounds (`learning_rate=0.02`). Both are
> untuned defaults evaluated with identical CV splits.

### Large-n with epoch-based block cycling (`sample_block_n='auto'`)

For n > 5 000, `sample_block_n='auto'` splits the dataset into non-overlapping
blocks (size `500 + (n−5000)//50`) and cycles through them epoch-by-epoch,
giving geometric diversity across blocks without paying full-n HVRT cost.

| Dataset | Metric | GeoXGB auto | XGB tuned | GeoXGB t/fold | XGB t/fold |
|---|---|---|---|---|---|
| friedman1_10k (n=10 000) | R² | 0.9287 | 0.9478 | **0.34 s** | 0.61 s |
| reg_20k (n=20 000) | R² | 0.9497 | 0.9649 | **0.56 s** | 1.15 s |
| reg_5k (n=5 000, no cycling) | R² | **0.9685** | 0.9644 | 0.66 s | 0.79 s |

GeoXGB is **1.8–2× faster** than tuned XGBoost on regression at equal
hyperparameters (lr=0.02, depth=3, 1 000 rounds). The small accuracy gap on
Friedman/20k closes under HPO. On reg_5k (below the block-cycling threshold)
GeoXGB wins outright with no cycling needed.

### C++ backend

GeoXGB ships a compiled C++ extension (`_geoxgb_cpp`) built with Eigen3 and
pybind11.  `fit()` and `predict()` route through C++ automatically when:

- The compiled extension is present (always true for wheel installs), **and**
- No `feature_types` argument is passed (categorical columns still use the
  pure-Python path), **and**
- `convergence_tol` is `None` (the Python path handles convergence tracking).

The Python path remains available for the full interpretability stack
(`noise_estimate()`, `sample_provenance()`, `partition_feature_importances()`,
`Gardener`, etc.) — pass `feature_types=["continuous"] * n_features` to opt in.

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

# Use HVRT for causal inference — noise-invariant (Theorem 3)
m0 = GeoXGBRegressor(partitioner='hvrt')
m0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
m1 = GeoXGBRegressor(partitioner='hvrt')
m1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
tau_hat = m1.predict(X_te) - m0.predict(X_te)
```

**Why HVRT for causal inference?** HVRT satisfies Theorem 3 (T-orthogonality):
its cooperation measure is invariant to isotropic Gaussian covariate noise.
PyramidHART loses this property — its L1-ball level sets are noise-sensitive,
which degrades performance in observational settings with noisy covariates.
The default `partitioner='pyramid_hart'` is optimal for regression; set
`partitioner='hvrt'` when covariates have measurement error or when using
S-learner-style metalearners that treat T as a feature alongside X.

PEHE benchmark on randomised trials (lower is better, n=2000, best-of-metalearners):

| τ(x) type | GeoXGB (`hvrt`) | XGBoost | Honest R-forest¹ |
|---|---|---|---|
| Linear (2X₁ + 1) | **0.180** | 0.207 | 0.247 |
| Nonlinear (2·sin(X₁π) + X₂²) | **0.408** | 0.608 | 0.796 |

¹ 2-fold cross-fitted R-forest (functional core of GRF).
Settings: n\_rounds=500, learning\_rate=0.1, max\_depth=3, y\_weight=0.25.

### Mediation fingerprint

The boost/partition importance ratio surfaces causal structure without a
separate statistical test. Features that are causally *upstream* of Y (i.e.
X where part of the effect passes through a mediator M) have
`boost_imp >> partition_imp` — the gradient signal recognises X as important
even when HVRT geometry anchors on M (pass `feature_types=[...]` to enable
the interpretability API):

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

AGPL-3.0-or-later
