# GeoXGB -- Geometry-Aware Gradient Boosting

GeoXGB replaces conventional subsampling and bootstrapping with geometry-aware
sample reduction and expansion powered by [HVRT](https://pypi.org/project/hvrt/)
(Hierarchical Variance-Retaining Transformer).

## Installation

```bash
pip install geoxgb
```

Requires `hvrt >= 2.1.0`, `scikit-learn`, and `numpy`. Python >= 3.10.

## Quick Start

```python
from geoxgb import GeoXGBRegressor, GeoXGBClassifier

# Regression
reg = GeoXGBRegressor(n_rounds=100, learning_rate=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Classification
clf = GeoXGBClassifier(n_rounds=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

# Pass feature types for mixed data
clf.fit(X_train, y_train, feature_types=["continuous", "categorical", ...])
```

## Key Features

- **Geometry-aware sampling** via HVRT's variance-retaining partitions
- **FPS reduction** -- keeps geometrically diverse representatives
- **KDE expansion** -- synthesizes samples in sparse regions
- **Adaptive noise detection** -- automatically backs off on noisy data
- **Multi-fit** -- refits partitions on residuals every N rounds
- **Full interpretability** -- feature importances, partition traces, sample provenance
- **Categorical support** -- pass `feature_types` to handle mixed data natively
- **Class reweighting** -- `class_weight` for imbalanced classification

## Parameters

### Shared (GeoXGBRegressor and GeoXGBClassifier)

| Parameter | Default | Description |
|---|---|---|
| `n_rounds` | 100 | Number of boosting rounds |
| `learning_rate` | 0.1 | Shrinkage per tree |
| `max_depth` | 6 | Maximum depth of each weak learner |
| `min_samples_leaf` | 5 | Minimum samples per leaf |
| `n_partitions` | None | HVRT partition count (None = auto-tuned) |
| `reduce_ratio` | 0.7 | Fraction to keep via FPS |
| `expand_ratio` | 0.0 | Fraction to synthesize via KDE (0 = disabled) |
| `y_weight` | 0.5 | HVRT blend: 0 = unsupervised geometry, 1 = y-driven |
| `method` | `'fps'` | HVRT selection method |
| `variance_weighted` | True | Budget allocation by partition variance |
| `bandwidth` | 0.5 | KDE bandwidth for expansion |
| `refit_interval` | 10 | Refit partitions every N rounds (None = off) |
| `auto_noise` | True | Auto-detect noise and modulate resampling |
| `auto_expand` | True | Auto-expand small datasets to `min_train_samples` |
| `min_train_samples` | 5000 | Target training-set size when `auto_expand=True` |
| `cache_geometry` | False | Reuse HVRT partition structure across refits |
| `random_state` | 42 | |

### GeoXGBClassifier only

| Parameter | Default | Description |
|---|---|---|
| `class_weight` | None | `None`, `'balanced'`, or `{class: weight}` dict |

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
model.feature_importances(feature_names)          # boosting importance
model.partition_feature_importances(feature_names) # geometric importance
model.partition_trace()                            # full partition history
model.partition_tree_rules(round_idx=0)            # human-readable rules
model.sample_provenance()                          # reduction/expansion counts
model.noise_estimate()                             # 1.0=clean, 0.0=pure noise
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
    refit_interval=10,
)
```

**Trade-off**: caching freezes the partition structure at the geometry
learned from the initial y (raw targets / first gradients). Subsequent
refits still curate samples geometrically but do not re-adapt partitions
to the current residual distribution. For most datasets this is negligible;
set `cache_geometry=False` (the default) if partition adaptivity matters.

## Benchmarks

See [`benchmarks/`](benchmarks/) for classification and regression benchmarks
comparing GeoXGB against XGBoost, including hyperparameter optimisation to
validate the defaults and a full walkthrough of GeoXGB's unique
interpretability outputs.

## License

MIT
