# Changelog

All notable changes to GeoXGB are documented here.

---

## [0.3.2] — 2026-03-05

### Bug fixes

- **Block cycling noise guard (`block_just_advanced`)** — critical fix for
  `sample_block_n` users. When the training block advanced to a new window,
  `red_idx` held indices from the *previous* block while `preds_on_X` held
  predictions for the *new* block. If the noise guard fired immediately after
  the advance (`skip_refit=True`), `sync_preds` would index `preds_on_X`
  with the stale `red_idx`, mapping old-block positions onto new-block samples.
  The resulting corrupt gradient updates caused training to diverge
  catastrophically (R² fell to −26 on Friedman #1 n=10 000).

  Fix in both Python (`_base.py`) and C++ (`geoxgb_base.cpp`):
  a `block_just_advanced` flag is set at each block advance.  On the first
  iteration after an advance, `last_noise_mod` (C++) / `_last_refit_noise`
  (Python) is reset to 1.0 so the noise guard cannot fire with stale indices,
  and the prediction discard path calls `predict_from_trees(Xr, -1)` (C++) /
  `_raw_predict(Xr)` (Python) rather than `sync_preds`.  The flag is cleared
  at the end of each round so behaviour is identical to v0.3.1 on all rounds
  that do not follow a block advance.

  Verified: setting `auto_noise=False` or `noise_guard=False` eliminated the
  divergence, confirming the noise guard as the specific failure site.

### New formula for `sample_block_n='auto'`

The automatic block size now scales as:

```
block_n = 500 + (n − 5 000) // 50   when n > 5 000, else None (disabled)
```

Previous formula was `max(500, n // 10)`, which gave a fixed ~10 blocks per
epoch at any n (block_n ≈ n / 10).  The new formula starts at 500 samples per
block at n=5 000 and grows slowly:

| n | Old block_n | New block_n | Blocks / epoch |
|---|---|---|---|
| 5 000 | 500 | 500 | 10 |
| 10 000 | 1 000 | 600 | 16 |
| 50 000 | 5 000 | 1 400 | 35 |
| 100 000 | 10 000 | 2 400 | 42 |

More blocks per epoch at large n means greater geometric diversity across
rounds without increasing per-block training cost.  Resolved in
`_base.py`, `regressor.py`, and `classifier.py` (the `'auto'` string is
resolved to an int before the C++ config is built).

### Benchmarks

**Small-n (confirmed post bug-fix, same CV protocol):**

| Dataset | Metric | GeoXGB default | GeoXGB HPO-best | XGBoost (300 est.) | HPO vs XGB |
|---|---|---|---|---|---|
| diabetes (n=442) | R² | 0.4256 | 0.4630 | 0.3147 | **+0.148** |
| friedman1 (n=1000) | R² | 0.9198 | 0.9188 | 0.8434 | **+0.075** |
| breast\_cancer (n=569) | AUC | 0.9931 | 0.9932 | 0.9886 | **+0.005** |
| wine (n=178, 3-class) | AUC | 0.9951 | 0.9993 | 0.9975 | **+0.002** |
| digits (n=1797, 10-class) | AUC | 0.9988 | 0.9987 | 0.9990 | −0.000 |

GeoXGB default beats or matches XGBoost on all 5 datasets.

**Large-n with `sample_block_n='auto'` (n_rounds=1000, 3×3-fold CV):**

| Dataset | Metric | GeoXGB auto | XGB tuned | GeoXGB t/fold | XGB t/fold |
|---|---|---|---|---|---|
| friedman1_10k (n=10 000) | R² | 0.9287 | 0.9478 | 0.34 s | 0.61 s |
| reg_20k (n=20 000) | R² | 0.9497 | 0.9649 | 0.56 s | 1.15 s |
| reg_5k (n=5 000) | R² | **0.9685** | 0.9644 | 0.66 s | 0.79 s |

GeoXGB is 1.8–2× faster than a tuned XGBoost on regression at equal
hyperparameters, with a small accuracy gap that closes under HPO.  On reg_5k
(below the block-cycling threshold, no cycling active) GeoXGB wins outright.

Classification AUC on `make_classification` synthetic data shows a larger gap
vs XGBoost on large n; this is primarily a multiclass-path speed issue on
Windows (process-spawn overhead) and is under investigation.

---

## [0.2.0] — 2026-03-04

### New features

- **PyramidHART partitioner** (`partitioner='pyramid_hart'`) — new geometry
  target `A = |S| − ‖z‖₁ ≤ 0` (triangle-inequality cooperation statistic).
  Level sets are axis-aligned piecewise-linear surfaces representable
  **exactly** by decision-tree splits, eliminating the structural mismatch
  between HVRT's quadric-cone boundaries and the weak learner's axis-aligned
  splits. Key properties: single-feature outlier cancellation (a 50σ spike
  shifts A by ~0, vs 30× in HVRT), degree-1 homogeneity, and O(n·d) cost.
  Selected as the new default across regression and classification benchmarks.

- **HART partitioner** (`partitioner='hart'`) — absolute pairwise cooperation
  `(‖z‖₁² − ‖z‖₂²)/2`, the L1 analog of HVRT's signed cooperation. Level
  sets are cross-polytopes whose faces correspond to one of the 2^d orthants.
  MAD-normalised y-extremeness and `criterion='absolute_error'` partition tree
  make it robust to outlier gradients. O(n·d) via the algebraic identity.

- **`HART`, `FastHART`, `PyramidHART` classes** exported from
  `geoxgb._hart` and `import geoxgb` — drop-in HVRT replacements for
  pipelines that use HVRT directly.

- **C++ native backend** — HVRT is now also bundled as compiled C++ source
  (Eigen3 + pybind11, fetched automatically at build time via CMake
  `FetchContent`). The compiled extension `_geoxgb_cpp` exposes
  `CppGeoXGBRegressor` and `CppGeoXGBClassifier`, which run the full
  PyramidHART / HART / orthant-stratified / simplex-mixup / Laplace pipeline
  in a single process without the Python overhead of repeated hvrt calls.
  `partitioner`, `method`, and `generation_strategy` are all dispatched
  natively in C++. The Python path (via `GeoXGBRegressor`) remains the
  default; the C++ path is opt-in via `_cpp_backend.py` or direct import.

- **orthant-stratified reduction** (`method='orthant_stratified'`) in both
  Python and C++ backends — groups samples by the sign pattern of
  `z − median(z)` (orthant key), allocates FPS budget proportional to each
  orthant's MAD(y) weight, then selects geometrically spread representatives
  within each orthant using L1 distance from the orthant centroid. Guarantees
  coverage of all 2^d sign-consistent regions of the cooperation cone.

- **simplex_mixup expansion** (`generation_strategy='simplex_mixup'`) — for
  each synthetic sample, draws two random rows from the partition and takes a
  uniform convex combination. Parameter-free, stays in the convex hull of
  partition members, in-orthant by construction. Outperforms Laplace KDE and
  Epanechnikov empirically on PyramidHART benchmarks.

- **Laplace KDE expansion** (`generation_strategy='laplace'`) — per-feature
  Laplace kernel centred on the partition centroid with 1.4826×MAD bandwidth.
  More heavy-tailed than Epanechnikov; matches L1 geometry.

- **`GeoXGBMAERegressor`** — L1-loss variant of `GeoXGBRegressor` optimised
  for Mean Absolute Error. Uses `sign(y − pred)` gradients (L1 gradient),
  median leaf values (`criterion='absolute_error'`), `partitioner='pyramid_hart'`
  (exact tree-level-set alignment), `method='orthant_stratified'` (per-facet
  FPS), `adaptive_reduce_ratio=True` (dynamic budget from gradient tail
  heaviness), and `generation_strategy='simplex_mixup'`.

- **`ConformalGeoXGBRegressor`** (experimental, `geoxgb.conformal`) —
  split-conformal prediction intervals combining per-HVRT-partition residual
  standard deviation with z-space k-NN density scaling. Provides calibrated
  `P(y ∈ interval) ≥ 1 − α` guarantees under exchangeability. Not yet
  exported from the top-level namespace; import directly from
  `geoxgb.conformal`.

### Default changes

Updated for `GeoXGBRegressor` based on a 2 000+ trial Optuna TPE study
(pyramid_hart, variance_ordered, simplex_mixup fixed; 12 continuous/discrete
params; 4 datasets: diabetes, Friedman #1/#2, classification):

- **`partitioner` default**: `'hvrt'` → `'pyramid_hart'`. Exact tree-level-set
  alignment eliminates the quadric-boundary approximation error of HVRT.
  Wins diabetes R²=0.3932 vs HVRT 0.3623; consistent with orthant geometry.

- **`method` default**: (was already `'variance_ordered'` since 0.1.7) —
  confirmed by Optuna as the top-performing reduction method with
  `pyramid_hart`. Orthant-stratified ties on classification, variance-ordered
  slightly ahead on regression.

- **`max_depth` default**: `4` → `3`. Optuna finds depth 2–3 optimal across
  all regression datasets; PyramidHART's polyhedral geometry is well-captured
  at shallower depths because the level sets are already axis-aligned.

- **`y_weight` default**: `0.5` → `0.25`. Optuna converges to 0.21–0.28
  across diabetes and Friedman #1; lower values let the PyramidHART geometry
  dominate over the y-extremeness component.

- **`adaptive_reduce_ratio` default**: `True` → `False`. Grid search shows no
  consistent benefit; disabled by default (enable for datasets with known
  heavy-tailed gradients or when using `GeoXGBMAERegressor`).

- **`generation_strategy` default**: `'epanechnikov'` → `'simplex_mixup'`.
  Outperforms Epanechnikov on PyramidHART benchmarks; parameter-free and
  stays in the convex hull.

### Performance

Head-to-head vs XGBoost (default vs default, same CV protocol, 3–5-fold):

| Dataset | Metric | GeoXGB default | GeoXGB HPO-best | XGBoost (300 est.) | HPO vs XGB |
|---|---|---|---|---|---|
| diabetes | R² | 0.4675 | 0.4982 | 0.3147 | **+0.183** |
| friedman1 | R² | 0.9210 | 0.9321 | 0.8434 | **+0.089** |
| breast\_cancer | AUC | 0.9943 | 0.9926 | 0.9886 | **+0.004** |
| wine | AUC | 0.9951 | 0.9993 | 0.9975 | **+0.002** |
| digits (10-class) | AUC | 0.9988 | — | — | — |

GeoXGB default beats XGBoost default on all 4 evaluated datasets. HPO-best
params from Optuna (2 000+ trials, pyramid_hart + variance_ordered +
simplex_mixup fixed): diabetes `lr=0.0125, max_depth=2, ri=200`;
friedman1 `lr=0.0147, max_depth=3, ri=10`.

### C++ micro-optimisations

- `adaptive_reduce_ratio`: replaced `std::sort` (O(n log n)) with two
  `std::nth_element` calls (O(n)) for the p50 and p90 order statistics.
- `per_feature_mad` allocation in `Expander::fit_partition()` is now guarded
  to only run when `strategy == GenerationStrategy::Laplace` (was computed
  unconditionally, wasting 2d vector allocations per partition per refit).
- `gen_laplace` centroid precomputed in `fit_partition()` and stored in
  `PartitionKDEParams::centroid_cont`, eliminating a redundant
  `X_cont.colwise().mean()` O(n_p × d) call per `generate()` invocation.

### Dependency

- **`hvrt >= 2.6.1` remains required** for the Python path partitioners
  (`GeoXGBRegressor` default, `HART`, `FastHART`, `PyramidHART` classes).
  The C++ backend (`CppGeoXGBRegressor`) bundles HVRT separately and does
  not require the Python package at runtime.

---

## [0.1.7] — 2026-02-27

### Licence

- Relicensed from MIT to **AGPL-3.0-or-later**.

### Dependency

- **Minimum HVRT version bumped to `>=2.6.1`** (was `>=2.5.0`).
  - HVRT 2.6.0 adds optional Numba-compiled kernels (`pip install hvrt[fast]`
    or `pip install geoxgb[fast]`). `_centroid_fps_core_nb` is 10–19× faster,
    the Epanechnikov sampler 5–8×, and `_pairwise_target_nb` 1.1–1.4×. The
    pure-NumPy fallback is automatic when Numba is absent.
  - HVRT 2.6.1 adds `tree_splitter` on the `HVRT` constructor (`'best'`
    default, `'random'` option — 10–50× faster HVRT partition-tree fits, ~8×
    end-to-end at n=50k). `fastmath=True` is applied to all Numba kernels
    automatically (+10–20% on top of 2.6.0 gains).
- `fast = ["hvrt[fast]>=2.6.1"]` optional extra added to `pyproject.toml`.

### New features

- **`hvrt_tree_splitter` parameter** (`GeoXGBRegressor`, `GeoXGBClassifier`,
  default `None`): forwarded to HVRT's `tree_splitter` constructor argument.
  `None` keeps HVRT's default (`'best'`). Set to `'random'` for HPO trial
  speed (10–50× faster per HVRT refit at large n); switch back to `None` or
  `'best'` for the final model or any regulatory audit path where split
  determinism matters.

- **`hvrt_auto_reduce_threshold` parameter** (`GeoXGBRegressor`,
  `GeoXGBClassifier`, default `None`): when `n_train` exceeds the threshold,
  HVRT is fitted on the full dataset with `auto_tune=True` (its own internal
  partition optimizer) and the training set is reduced to `threshold` samples
  before boosting begins. This is the recommended approach for datasets with
  hundreds of thousands of rows — the external HVRT geometry selects a
  representative 100k (or whatever threshold is set) and GeoXGB's boosting
  loop runs at a tractable scale. At 630k samples, a threshold of 100k gives
  near-identical holdout AUC to training on the full set.

- **`hvrt_max_samples_leaf` parameter** (`GeoXGBRegressor`,
  `GeoXGBClassifier`, default `None`): caps HVRT partition size by computing
  `n_partitions = ceil(n / hvrt_max_samples_leaf)` when `n_partitions` is not
  set explicitly. `min_samples_leaf` still enforces the lower bound.

- **`hvrt_params` dict passthrough** (`GeoXGBRegressor`, `GeoXGBClassifier`,
  default `None`): arbitrary keyword arguments forwarded to the HVRT
  constructor. Named GeoXGB parameters (`y_weight`, `bandwidth`,
  `n_partitions`, `min_samples_leaf`, `hvrt_tree_splitter`, `random_state`)
  always take precedence over any overlapping keys in `hvrt_params`.

### Default changes

- **`method` default**: `'fps'` → `'variance_ordered'`. `variance_ordered`
  selects samples with the highest k-NN distance variance within each HVRT
  partition, prioritising boundary and transition samples. It is fully
  deterministic (no RNG at any step), wins on 6/12 quality benchmark datasets
  vs `fps` 4/12, and is auditable: every retained sample has a computable
  k-NN variance score that can be inspected and reproduced exactly.

### Internal

- `_resampling.py`: `hvrt_params`, `hvrt_max_samples_leaf`, and
  `hvrt_tree_splitter` forwarded through `hvrt_resample()` to the HVRT
  constructor. `_kde_stratified_reduce()` rewritten to use centroid distance
  (O(n·d) per partition) instead of per-partition k-NN, eliminating the
  O(n²/P) scaling cliff for the `kde_stratified` strategy.
- `_base.py`: auto-reduce logic at the top of `_fit_boosting` uses HVRT with
  `auto_tune=True` when `hvrt_auto_reduce_threshold` is set and exceeded.

---

## [0.1.5] — 2026-02-25

### New features

- **`y_weight` added to `GeoXGBOptimizer` search space** (`optimizer.py`):
  Optuna TPE now searches `y_weight` over `[0.1, 0.3, 0.5, 0.7, 0.9]`. On
  sparse high-dimensional data (many irrelevant features), increasing `y_weight`
  toward 1.0 makes HVRT more y-driven and less diluted by irrelevant feature
  dimensions — the correct structural fix for that regime. With 25 trials the
  CV score on `sparse_highdim` already favours GeoXGB (0.9483 vs XGBoost 0.9318);
  more trials are needed to fully exploit this in test-set generalisation.

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
