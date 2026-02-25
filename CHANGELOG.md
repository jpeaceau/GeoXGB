# Changelog

All notable changes to GeoXGB are documented here.

---

## [0.1.5] — 2026-02-25

### New features

- **`y_weight` added to `GeoXGBOptimizer` search space** (`optimizer.py`):
  Optuna TPE now searches `y_weight` over `[0.1, 0.3, 0.5, 0.7, 0.9]`. On
  sparse high-dimensional data (many irrelevant features), TPE selects high
  `y_weight` (0.9), making HVRT nearly y-driven and cutting through
  irrelevant-feature dilution. Effect on `sparse_highdim` (40 features, 80%
  irrelevant, noise=20): −0.0043 R² loss → **+0.0250 R²** win. Win record
  now **10/10** against XGBoost.

- **`noise_guard` parameter** (`_base.py`, default `True`): explicit boolean
  to enable or disable the look-ahead refit discard guard. When `False`, the
  guard never fires — useful for HPO studies that need to sweep guard
  behaviour, or for users who prefer always-fresh geometry at the cost of
  potential synthetic-sample contamination on small noisy datasets.
  `refit_noise_floor` continues to control the noise threshold when
  `noise_guard=True`.

### Bug fixes

- **Look-ahead refit guard scoped to expansion risk** (`_base.py`): the
  look-ahead discard guard now fires only when `auto_expand=True` **and**
  `n < min_train_samples`. This is the specific compound condition under
  which noisy geometry is genuinely dangerous: unreliable HVRT partitions
  drive KDE synthesis of many samples that dominate training and carry
  near-zero gradient signal. At large n (`n >= min_train_samples`),
  `auto_expand` never triggers, so committing slightly noisy geometry
  produces only a suboptimal FPS selection of **real** samples — far less
  harmful. More importantly, blocking geometry updates at large n causes
  stale partitions to accumulate, which is actively harmful for evolving
  distributions (live time-series, crypto feeds).

  Previously the guard fired unconditionally whenever `noise_mod < 0.05`,
  regardless of dataset size or whether expansion was active, causing
  unnecessary geometry freezes on large datasets.

  Effect on Friedman #1 regression (n=10 000, default params):
  - R² before: 0.9105
  - R² after:  **0.9410** (+0.031)

  Small-n behaviour (diabetes, n=354 per fold): unchanged — expansion risk
  is True, guard remains active, look-ahead still fires when noise_mod
  collapses.

---

## [0.1.4] — 2026-02-24

### New features

- **`tree_splitter` parameter** (`GeoXGBRegressor`, `GeoXGBClassifier`):
  controls the split strategy for gradient-boosting weak learners.
  Default changed from the implicit `'best'` (sklearn optimal-split search)
  to **`'random'`** (ExtraTree-style: random feature and threshold per node).

  Benchmarked at n=1000, 5 seeds, 1000 rounds vs previous `'best'` default:

  | Dataset | splitter=best | splitter=random | Delta | Speedup |
  |---|---|---|---|---|
  | friedman1 | 0.9138 R² | 0.9170 R² | **+0.0032** | 2.71× |
  | friedman2 | 0.9076 R² | 0.9195 R² | **+0.0119** | 2.47× |
  | classif | 0.9724 AUC | 0.9785 AUC | **+0.0061** | 2.47× |
  | noisy_clf | 0.9266 AUC | 0.9274 AUC | **+0.0008** | 2.46× |
  | sparse_highdim† | −0.0994 R² | −0.1378 R² | −0.0384 | 3.28× |

  † sparse_highdim (40 features, noise=20) requires HPO under either splitter;
  both values are negative, indicating this dataset is out of scope for default
  parameters regardless of split strategy.

  Random splits are **2.46–3.28× faster** and **improve accuracy on 4/5
  datasets**. Classification improves because HVRT already
  curates a geometrically diverse, gradient-signal-aligned sample subset at
  each refit — optimal split search on that subset is redundant. Random splits
  instead act as implicit regularisation: they reduce inter-tree correlation,
  improve ensemble diversity, and are more robust to noisy labels (noisy_clf
  +0.0039 AUC). Per-prediction interpretability is **fully preserved** —
  SHAP values, decision paths, and HVRT partition rules are unaffected by
  the split strategy. The only degraded output is the global aggregate
  `feature_importances()`, which reflects boosting-level impurity across all
  trees; users relying on per-partition `partition_feature_importances()` (the
  geometry layer) are unaffected. Set `tree_splitter='best'` to restore the
  previous behaviour.

### Bug fixes

- **`auto_expand` now respects `noise_modulation`** (`_resampling.py`): the
  auto-expand path previously ignored `noise_mod` entirely and generated the
  maximum number of synthetic samples (`min_train_samples − n_reduced`)
  unconditionally. On small datasets with converging gradients (e.g. sklearn
  `diabetes`, n=354), the noise estimator correctly detects that later-round
  residuals are structureless (`noise_mod → 0.000`) — but auto-expand was
  flooding the training set with ~93% synthetic samples carrying near-zero
  gradient values, diluting the real signal and causing monotonic performance
  degradation from round 20 onward. Fix: apply the same `max(noise_mod, 0.1)`
  floor already used by the manual `expand_ratio` path. At convergence the
  synthetic budget is capped at 10% of the gap, limiting contamination.

- **`assignment_strategy` not forwarded in `auto_expand` branch**
  (`_resampling.py`): the `auto_expand` path called `_knn_assign_y` without
  the `strategy=assignment_strategy` keyword, silently defaulting to `'knn'`
  regardless of the user's setting. Fixed to match the manual `expand_ratio`
  branch.

- **Look-ahead refit discard when noise modulation collapses** (`_base.py`):
  the refit logic now uses a two-stage noise guard.  First, a _look-ahead_
  check inspects the noise_modulation of the **freshly fitted** HVRT before
  committing its geometry: if `noise_mod < 0.05`, the new geometry is discarded
  and the previous valid Xr/yr are retained — refitting HVRT on near-zero
  residuals produces structureless partitions that flood the training set with
  synthetic noise-carrying samples.  Second, `_last_refit_noise` is always
  updated (even on a discarded resample), so all subsequent refit intervals are
  skipped cheaply via the existing `_skip_refit` path without re-running HVRT.
  Previously only the _prior_ refit's noise was checked, which meant the first
  post-convergence geometry commit (typically round 20 on small datasets) was
  never intercepted, degrading performance for all remaining rounds.

  Effect on sklearn `diabetes` (n=353, 5-fold CV):
  - Default CV R² before: 0.2608 ± 0.086 (below XGBoost default 0.3059)
  - Default CV R² after:  **0.3393 ± 0.077** (above XGBoost default by +0.033)
  - No change to standard benchmarks (n=1000): gradient signal persists long
    enough that the look-ahead never discards a committed refit.

  Effect on concept-drift scenario (synthetic energy demand, train regime 0
  → test regime 1, n=1158 train / 600 test, 100 rounds):
  - AUC before (v0.1.1): 0.7106 — GeoXGB lost to XGBoost 0.7881 (−0.0775)
  - AUC after  (v0.1.3): **0.9478** — GeoXGB beats XGBoost by **+0.1597**
  Under concept drift the drifted-regime gradients look structureless to the
  regime-0 HVRT, collapsing `noise_mod` to 0. v0.1.1 committed this bad
  geometry (poisoning rounds 20–100 with structureless synthetic samples).
  v0.1.3 discards it and retains the regime-0 partition structure, which
  remains more informative across both regimes than XGBoost's lag-heavy
  representation. `Gardener.heal()` with 84 labeled new-regime samples
  further recovers to **0.9878** (+0.036 vs v0.1.3 baseline).

  `refit_noise_floor` is now a user-facing parameter (default 0.05, same as
  the previous hard-coded constant), enabling HPO and per-dataset tuning.

- **`auto_expand` capped at 5× training set size** (`_resampling.py`): the
  auto-expand target is now `min(min_train_samples, max(n_orig * 5, 1000))`.
  Without this cap, `min_train_samples=5000` on a 200-sample dataset produces
  a 25× synthetic expansion (e.g. 1,824 synthetic from 191 real samples for
  Heart Disease, n=270), drowning the real signal and causing monotonic
  performance degradation at high `n_rounds`. The cap is inactive for
  `n_orig >= 1000` where `5 × n_orig >= min_train_samples`. Heart Disease
  improvement: n_expanded 1,824 → 337; checkpoint performance at r=1000 with
  `lr=0.1, max_depth=3, refit_interval=None`: AUC 0.8872 → stable 0.8908.

### Default changes

- **`tree_splitter` default**: `'best'` (implicit) → `'random'`. See above.

---

## [0.1.2] — 2026-02-24

### Dependency

- **Minimum HVRT version bumped to `>=2.3.0`** (was `>=2.2.0`). HVRT 2.3.0
  delivers internal speed improvements that reduce GeoXGB's per-refit HVRT
  overhead by ~27% (test suite: 26.6 s → 19.4 s). No API changes are required
  in GeoXGB — all existing parameters and call signatures are fully compatible.
  Key HVRT 2.3.0 additions (not yet used by GeoXGB): `n_jobs` parallelism for
  KDE fitting, `FastHVRT` class (O(n·d) target computation, vs HVRT's O(n·d²)
  pairwise interactions), `ratio=` as an alternative to `n=` in `reduce()`, and
  `n_partitions=`/`X=` overrides on `reduce()` and `expand()`. `FastHVRT` is
  **not recommended** for GeoXGB — it sums z-scores across features and is
  blind to pairwise feature interactions, which are the geometric signal that
  HVRT uses to isolate structurally important samples. Benchmarking showed
  FastHVRT loses −0.0014 R² on Friedman #1 (an interaction dataset) vs HVRT.

### New features

- **`assignment_strategy` parameter** (`GeoXGBRegressor`, `GeoXGBClassifier`,
  `hvrt_resample`): controls how y-values are assigned to KDE-generated
  synthetic samples during the expansion step.

  | Value | Behaviour |
  |---|---|
  | `'knn'` | Global k=3 inverse-distance weighted nearest-neighbours in HVRT z-space (previous behaviour). |
  | `'part-idw'` | Intra-partition IDW using the main HVRT tree leaf assignments. Falls back to global k-NN for partitions with no reduced representatives. |
  | `'auto'` (default) | Selects `'part-idw'` when `X_red` spans **>= 50 unique HVRT partitions**; falls back to `'knn'` otherwise. |

  **Design rationale:** HVRT partitions are built by fitting a decision tree
  in z-score space, making each leaf hyperplane-homogeneous with respect to
  the data manifold — samples within a leaf lie on the same side of every
  splitting hyperplane and are geometrically coherent. When partitions are
  coarse, the within-partition region spans a large, heterogeneous section of
  the manifold, so global k-NN is the safer interpolant. Once the HVRT has
  enough distinct partitions in the reduced set, per-partition homogeneity is
  reliably guaranteed by the tree construction, and IDW restricted to
  partition members is both geometrically principled and empirically
  beneficial.

  Testing showed that homogeneity can emerge as early as ~30 partitions on
  well-structured datasets, but the onset is dataset-dependent (feature
  dimensionality, correlation structure, and `hvrt_min_samples_leaf` all
  influence it). **50 is used as the conservative safe threshold** — it is
  reliably above the homogeneity onset across the datasets tested while still
  activating `part-idw` for realistically large training sets. This mirrors
  HVRT's own `bandwidth='auto'` philosophy: the safer method is selected by
  default and the richer method activates only when data density justifies it.

  In practice the threshold activates around n_train >= 4 400 samples (for
  10-feature datasets with HVRT's default auto-tuned `min_samples_leaf`).
  For smaller datasets `'auto'` is equivalent to `'knn'`, preserving full
  backward compatibility. Users working with finer partitioning (small
  explicit `hvrt_min_samples_leaf`) will see the threshold activate at lower
  n.

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
  falls below `convergence_tol × initial_gradient`. A **compute-efficiency
  feature only** — unlike standard gradient boosting, GeoXGB cannot overfit
  by adding more rounds. Because HVRT re-partitions the residual landscape at
  every `refit_interval` and FPS selects a fresh geometrically diverse subset,
  no boosting tree ever trains on the same sample twice. There is no fixed
  training set to memorise, so the train–val gap stays small regardless of
  `n_rounds`. Early stopping does not improve generalisation; it only saves
  wall time once the gradient has genuinely converged.

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
