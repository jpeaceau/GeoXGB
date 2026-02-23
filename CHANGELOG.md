# Changelog

All notable changes to GeoXGB are documented here.

---

## [0.1.1] — 2026-02-24

### Dependency

- **Minimum HVRT version bumped to `>=2.2.0`** (was `>=2.1.0`). HVRT 2.2.0
  introduces `generation_strategy` and `adaptive_bandwidth` in `expand()`,
  which GeoXGB now uses to enable Epanechnikov KDE by default. It also includes
  the `is_reduction` parameter in `auto_tune_tree_params` (first in 2.1.1) that
  relaxes the 40:1 sample-to-feature ratio for non-reduction use cases, and
  the `bandwidth='auto'` default that auto-selects the KDE bandwidth.
  Example impact on UCI Heart Disease (270 samples, 13 features):
  | Metric | HVRT 2.1.0 | HVRT 2.2.0 |
  |---|---|---|
  | Partitions | 1 | 7 |
  | Noise modulation | 0.000 | 0.290 |
  | Spearman agreement | −0.258 (degenerate) | 0.522 (meaningful) |

### New features

- **`Gardener` class** (`geoxgb.gardener`): post-hoc surgical editor for any
  fitted `GeoXGBClassifier` or `GeoXGBRegressor`. Provides manual tools
  (`adjust_leaf`, `prune`, `graft`, `rollback`, `reset`) and automatic
  self-healing (`diagnose`, `heal`). `heal()` detects systematically biased
  leaves, attempts corrections, validates on a held-out set, and commits only
  if the change is beneficial. Iterates until convergence. Also exposes
  `recommend_feature_weights()` to derive per-feature scaling from the ratio
  of gradient importances to partition importances.

- **`GeoXGBOptimizer` class** (`geoxgb.optimizer`): Optuna TPE hyperparameter
  search mirroring the `HVRTOptimizer` API. Categorical search space over
  `n_rounds`, `learning_rate`, `max_depth`, and `refit_interval`. Trial 0 is
  always the v0.1.1 defaults (guarantees HPO ≥ baseline). Exposes
  `best_params_`, `best_score_`, `best_model_`, `study_`, and `task_` after
  fitting. Optuna is an optional dependency (`pip install geoxgb[optimizer]`);
  importing `GeoXGBOptimizer` without optuna installed does not raise until
  `.fit()` is called. A `fast=True` mode (default) speeds up trials via
  `cache_geometry=True`, `auto_expand=False`, and `convergence_tol=0.01`,
  then refits the final best model with full-quality settings.

- **`save()` and `load_model()`**: `model.save(path)` serialises the fitted
  model to disk via joblib. `from geoxgb import load_model; load_model(path)`
  restores it. `save()` strips the large training data arrays (`X_`, `X_z_`,
  `partition_ids_`) from all HVRT instances in the resample history before
  pickling, reducing file size from ~13 GB (for a 630 k-sample, 4 000-round
  model) to under 100 MB.

- **`feature_weights` parameter** (`_GeoXGBBase`, `GeoXGBRegressor`,
  `GeoXGBClassifier`): per-feature scaling array applied to `X` before HVRT
  sees it. HVRT geometry, noise estimation, and KDE expansion all operate in
  the scaled space; gradient-boosting trees are trained on the original unscaled
  data. Allows the user (or `Gardener.recommend_feature_weights()`) to
  emphasise features whose gradient importance substantially exceeds their
  geometric importance.

- **`convergence_tol` parameter**: when set (e.g. `0.001`), training stops
  early if the mean-absolute-gradient improvement over the last two refit cycles
  falls below `convergence_tol × initial_gradient`. A compute-efficiency
  feature only — GeoXGB does not overfit with high `n_rounds`, so early
  stopping does not improve generalisation; it only saves wall time when the
  gradient has genuinely converged.

- **`generation_strategy` parameter** (default `"epanechnikov"`): KDE sampling
  strategy forwarded to `hvrt.expand()`. `"epanechnikov"` uses per-partition
  Epanechnikov kernels with Scott's rule bandwidth, which is more robust than
  fixed-bandwidth Gaussian KDE on noisy or high-dimensional data. Never hurts
  on any of the five standard benchmark datasets; wins on `friedman1`
  (+0.012 R²) and the classification benchmark (+0.004 AUC) vs `generation_strategy=None`.

- **`adaptive_bandwidth` parameter** (default `False`): when `True`, scales
  KDE bandwidth per partition by local sample density, passed directly to
  `hvrt.expand()`.

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
  datasets (HVRT 2.2.0), partition counts can meaningfully evolve as the
  residual landscape changes during training.

- **`evolution_report` — `importance_drift` field** (when
  `detail='standard'` or `'full'` and `feature_names` is provided): compares
  HVRT partition feature importances between the first and final resamples.
  Reports which features shifted by more than 5 percentage points, sorted by
  magnitude of drift. Answers the question: "Did the model's understanding of
  data geometry change as it learned?"

- **`n_jobs` multiclass parallelism** (`GeoXGBClassifier`): when training a
  K-class classifier, the K one-vs-rest ensembles are independent and are now
  fitted in parallel using `joblib.Parallel(prefer="processes")`. Measured
  speedups vs `n_jobs=1`: 2.5× (K=3), 3.9× (K=5), 5.1× (K=8). AUC is
  bit-for-bit identical across all `n_jobs` values. Binary and regression
  remain inherently sequential and are unaffected.

### Documentation

- **`n_partitions` docstring clarified** (`GeoXGBRegressor`): documents the
  interaction with `hvrt_min_samples_leaf` constraints and gives guidance on
  tuning direction (increase for finer reporting, decrease for more stable KDE
  expansion).
- **`min_samples_leaf` docstring clarified** (`GeoXGBRegressor`): now
  explicitly notes that this parameter controls the weak learner
  (DecisionTree), distinct from `hvrt_min_samples_leaf`.

### Bug fixes

- **Gradient-signal noise estimator** (`_noise.py`): replaced the
  `mean_abs_z` between-partition variance metric with a k-NN local-mean
  variance metric computed in HVRT z-score space. The old metric measured
  radial distance from the feature-space origin, which is near-zero across
  all partitions on datasets like California Housing where partitions are
  geographically balanced — causing `noise_mod = 0.000` throughout training
  and disabling resampling entirely. The new metric computes `snr =
  Var(local_means) / Var(y)`, where each sample's local mean is the average
  y over its k=10 nearest neighbours in z-space. This directly measures y
  predictability from X: clean data → neighbours have similar y → high snr →
  high modulation; noisy data → neighbours have random y → snr ≈ 1/k → low
  modulation → `eff_reduce → 1.0` (keep all samples). The approach is robust
  regardless of the number of HVRT partitions, which is critical because
  HVRT 2.1.0 may create only 1–2 partitions on small datasets.

  Signature change: `estimate_noise_modulation(hvrt_model)` →
  `estimate_noise_modulation(hvrt_model, y, X)`. The call site in
  `_resampling.py` is updated accordingly; `y` and `X` are already in scope.

  This fix resolves three previously failing tests: `test_clean_data_modulation`,
  `test_noise_estimate_clean`, and `test_noise_dampens_resampling`. The fourth
  test (`test_partition_feature_importances_signal_over_noise`) requires HVRT
  2.1.1's relaxed partitioning for small datasets and will pass once HVRT 2.1.1
  is installed.

- **`sample_provenance()` — `reduction_ratio` semantics** (`_base.py`):
  `reduction_ratio` now reports the FPS-reduction fraction (`n_reduced /
  original_n`) rather than the total training set fraction including KDE
  expansion (`total_training / original_n`). The old value exceeded 1.0
  whenever auto-expansion was active (e.g., 16.67× for 300-sample datasets
  expanded to 5000), making it meaningless as a "reduction" metric. The new
  value is always in (0, 1] and correctly shows how aggressively noise
  modulation dampened the FPS reduction step.

- **KDE expansion floor in `_resampling.py`**: when `expand_ratio > 0`, the
  effective noise modulation is floored at 0.1 for the expansion calculation,
  so expansion always occurs when the user requests it. Previously, correctly
  detecting a noisy dataset (noise_mod ≈ 0) would suppress expansion entirely
  even when expand_ratio was explicitly set.

### Default changes

- **`bandwidth` default**: `0.5` → `"auto"`. HVRT 2.2.0's `"auto"` mode
  selects `h=0.10` Gaussian or Epanechnikov based on mean partition size.
  Benchmark: `"auto"` wins or ties in 30/18 conditions vs fixed bandwidth;
  Scott/Silverman win zero (over-smoothed).

- **`refit_interval` default**: `10` → `20`. Wins 4/5 standard benchmark
  datasets vs `refit_interval=10` while being ~25% faster.

### Internal

- `hvrt_resample()` in `_resampling.py` accepts `min_samples_leaf`,
  `generation_strategy`, and `adaptive_bandwidth` keyword arguments and
  forwards them to the `HVRT()` constructor and `expand()` respectively.
- `_do_resample()` in `_base.py` passes all new parameters through to
  `hvrt_resample()`.
- `preds_on_X` incremental tracker eliminates the O(i × n) `_raw_predict`
  call at each refit interval, replacing it with a single running accumulation.
- `n_trees` property is O(1) via `len(self._trees)`; multiclass overrides
  with `sum(len(tk) for tk in _mc_trees)`.

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
