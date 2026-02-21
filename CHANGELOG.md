# Changelog

All notable changes to GeoXGB are documented here.

---

## [0.1.1] — 2026-02-22

### Dependency

- **Minimum HVRT version bumped to `>=2.1.1`** (was `>=2.1.0`). HVRT 2.1.1
  introduced the `is_reduction` parameter in `auto_tune_tree_params`, which
  relaxes the 40:1 sample-to-feature ratio for non-reduction use cases. This
  fixes the critical single-partition problem on small datasets (< ~350 samples
  with 13+ features) that caused partition importance to be all zeros, noise
  modulation to be stuck at 0.0, and the reporting pipeline to degrade.
  Example impact on UCI Heart Disease (270 samples, 13 features):
  | Metric | HVRT 2.1.0 | HVRT 2.1.1 |
  |---|---|---|
  | Partitions | 1 | 7 |
  | Noise modulation | 0.000 | 0.290 |
  | Spearman agreement | −0.258 (degenerate) | 0.522 (meaningful) |

### New features

- **`hvrt_min_samples_leaf` parameter** (`_GeoXGBBase`, `GeoXGBRegressor`,
  `GeoXGBClassifier`): pass-through to HVRT's `min_samples_leaf`. Default
  `None` preserves existing behaviour (HVRT auto-tunes via its default
  formula). Set explicitly to control partition granularity — smaller values
  (20–30) produce finer partitions for richer interpretability; larger values
  (50+) produce more stable KDE fits for synthetic expansion.

- **`evolution_report` — `partition_stability` field**: always included in the
  evolution report. Summarises whether the HVRT partition count changed across
  refits, with `min_partitions`, `max_partitions`, `changed` (bool), and an
  `interpretation` string. With functional partitioning now working on small
  datasets (HVRT 2.1.1), partition counts can meaningfully evolve as the
  residual landscape changes during training.

- **`evolution_report` — `importance_drift` field** (when
  `detail='standard'` or `'full'` and `feature_names` is provided): compares
  HVRT partition feature importances between the first and final resamples.
  Reports which features shifted by more than 5 percentage points, sorted by
  magnitude of drift. Answers the question: "Did the model's understanding of
  data geometry change as it learned?"

### Documentation

- **`n_partitions` docstring clarified** (`GeoXGBRegressor`): documents the
  interaction with `hvrt_min_samples_leaf` constraints and gives guidance on
  tuning direction (increase for finer reporting, decrease for more stable KDE
  expansion).
- **`min_samples_leaf` docstring clarified** (`GeoXGBRegressor`): now
  explicitly notes that this parameter controls the weak learner
  (DecisionTree), distinct from `hvrt_min_samples_leaf`.

### Internal

- `hvrt_resample()` in `_resampling.py` accepts a new `min_samples_leaf`
  keyword argument and forwards it to the `HVRT()` constructor.
- `_do_resample()` in `_base.py` passes `self.hvrt_min_samples_leaf` through
  to `hvrt_resample()`.

---

## [0.1.0] — 2026-02-21

Initial release. Incorporates UPDATE-001 through UPDATE-005:

- **UPDATE-001**: Auto-expand small datasets to `min_train_samples` via KDE
  when `auto_expand=True` and `n_reduced < min_train_samples`.
- **UPDATE-002**: `class_weight` parameter for `GeoXGBClassifier` (`None`,
  `'balanced'`, or `{label: weight}` dict).
- **UPDATE-003**: Class-conditional noise estimation for classifiers (partition
  purity vs random baseline).
- **UPDATE-004**: `cache_geometry=True` option to reuse HVRT geometry across
  refits (one HVRT fit per model; faster for large datasets).
- **UPDATE-005**: Correct target reconstruction at refit intervals — classifiers
  recover class probabilities via `sigmoid(pred) + gradient` rather than the
  regression identity.

Performance improvements also included in v0.1.0:
- Incremental `preds_on_X` tracker eliminates O(i × n) `_raw_predict` at each
  refit interval.
- `n_trees` property is O(1).
