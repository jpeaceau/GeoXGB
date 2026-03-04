# Parameters Reference

This page documents every constructor parameter for `GeoXGBRegressor`,
`GeoXGBClassifier`, and `GeoXGBMAERegressor`. Parameters are grouped by role.

---

## Priority guide at a glance

| Priority | Parameter | Why it matters |
|---|---|---|
| **Must tune** | `learning_rate`, `max_depth` | Dominant accuracy axis; interact strongly |
| **Should tune** | `n_rounds`, `reduce_ratio`, `refit_interval` | Large effect on speed and accuracy |
| **Dataset-dependent** | `expand_ratio`, `y_weight` | Consistent secondary effect |
| **Leave alone** | `auto_noise`, `noise_guard`, `auto_expand` | Sensible defaults; rarely need changing |
| **Advanced only** | `partitioner`, `method`, `generation_strategy` | Changing without understanding hurts performance |

---

## Overfitting and rounds

**GeoXGB cannot overfit if `learning_rate` is sufficiently small.**

Each boosting round adds a weak learner whose prediction is multiplied by
`learning_rate` before being added to the ensemble. When `learning_rate` is low
enough (roughly ≤ 0.03, and comfortably ≤ 0.02), each incremental tree
contributes so little that the cumulative update saturates at the true signal
rather than the noise. The sample curation from PyramidHART further suppresses
overfitting by ensuring the training distribution seen at each round is
geometrically representative rather than gradient-dominated.

**Practical consequence:** once you have found a good `learning_rate`, you can
increase `n_rounds` freely without risk of overfitting. More rounds will continue
to reduce loss until convergence. The recommended workflow is:

1. Find `learning_rate` via HPO (see [HPO guide](hpo_guide.md)).
2. Set `n_rounds` high (1 000–5 000) and let training run to completion.
3. Use `convergence_tol` to stop automatically if desired.

---

## Boosting parameters

### `n_rounds` — int, default `1000`

Total number of boosting rounds (weak learners). With a well-tuned
`learning_rate` (≤ 0.02) this is safe to set high: GeoXGB will not overfit.
More rounds improve accuracy until the gradient signal is exhausted.

- **Regression HPO range:** 500–5 000
- **Classification HPO range:** 500–3 000
- **Interaction:** must be scaled with `learning_rate` — halving the learning
  rate roughly requires doubling `n_rounds` for equivalent accuracy.

---

### `learning_rate` — float, default `0.02`

Shrinkage multiplied onto each tree's prediction before adding it to the
ensemble. The single most important parameter. Lower is almost always better
given sufficient `n_rounds`.

- **Regression HPO range:** 0.005–0.05 (Optuna optimum: 0.010–0.020)
- **Classification HPO range:** 0.005–0.05
- **Overfitting threshold:** values ≤ 0.03 effectively eliminate overfitting
  risk with PyramidHART geometry. Values > 0.05 risk overfitting on small
  datasets.
- **Interaction with `max_depth`:** the optimal `learning_rate` decreases as
  `max_depth` increases. Shallow trees (depth 2–3) tolerate slightly higher
  rates than deep trees.

---

### `max_depth` — int, default `3`

Maximum depth of each decision tree weak learner. Shallower trees are more
robust and work better with PyramidHART's polyhedral geometry (which provides
strong structural priors). Deeper trees can overfit the geometric partition
when gradients are noisy.

- **Regression HPO range:** 2–5 (Optuna rank-1 at depth 2–3)
- **Classification HPO range:** 2–6
- **Recommended starting point:** 3 for regression, 4 for classification
- **Interaction:** `learning_rate × max_depth` is the dominant accuracy axis.
  Increase depth only if you also reduce `learning_rate`.

---

### `min_samples_leaf` — int, default `5`

Minimum samples per leaf in each weak learner. Prevents the tree from fitting
partitions that contain only a handful of samples. Higher values act as implicit
regularisation.

- **HPO range:** 1–20
- **Leave at default unless:** your dataset is very small (< 200 samples), in
  which case increase to 10–20.

---

## Geometry / partitioning parameters

These control the HVRT/PyramidHART geometry that GeoXGB is built around. They
have a significant effect on accuracy but require understanding the geometric
mechanism to tune well.

### `partitioner` — str, default `'pyramid_hart'`

Which geometric partitioner to use for sample curation.

| Value | Geometry | Best for |
|---|---|---|
| `'pyramid_hart'` | Axis-aligned polyhedral (cross-polytope) level sets; `A = \|S\| − ‖z‖₁` | General regression and classification (default) |
| `'hvrt'` | Smooth quadric level sets; variance-based; satisfies T⊥Q noise-invariance (Theorem 3) | **Causal inference**, treatment effect estimation |
| `'hart'` | MAD-based HVRT variant; L1-natural geometry | MAE regression, heavy-tailed data |
| `'fasthvrt'` | Linear approximation of HVRT target; O(n·d) | Very large datasets where HVRT is too slow |

**Recommendation:** leave at `'pyramid_hart'` for regression and classification.
Switch to `'hvrt'` for any causal inference application — HVRT's
T⊥Q orthogonality guarantee means partitions are invariant to covariate noise
in the treatment assignment, which is critical for unbiased effect estimation.

**Advanced only.** Changing this parameter without understanding the geometric
implications typically hurts performance.

---

### `n_partitions` — int or None, default `None`

Target number of geometric partitions. When `None`, the HVRT library
auto-selects based on dataset size (roughly `√n`). Larger `n_partitions` gives
finer geometric resolution but more expensive refits.

- **Leave at `None`** in almost all cases. The auto-selection is well-calibrated.
- **Set manually only if:** you observe the partition tree collapsing to a single
  region (check `model.partition_feature_importances()`), or you need strict
  control over memory use at very large `n`.

---

### `y_weight` — float, default `0.25`

Blend between unsupervised geometry (0.0) and fully gradient-driven geometry
(1.0). At `0.25`, the partition tree uses both the feature distribution and the
gradient signal to define regions. This is the Optuna-optimised default across
regression tasks.

- **Regression HPO range:** 0.1–0.5 (Optuna consistently finds 0.21–0.28)
- **Classification HPO range:** 0.1–0.6
- **Lower values:** more stable geometry, better for noisy data
- **Higher values:** more gradient-responsive geometry, better for clean
  low-noise data with strong feature–target relationships

---

### `hvrt_min_samples_leaf` — int or None, default `None`

Minimum samples per leaf in the partition tree (HVRT/PyramidHART internal
tree). Controls the minimum partition size. When `None`, the HVRT library
auto-selects.

- **Leave at `None`** unless you have domain knowledge about the minimum
  meaningful cluster size in your data.
- **Set to 20–50** on small datasets (n < 500) to prevent degenerate
  single-sample partitions.

---

### `refit_interval` — int or None, default `50`

Re-fit the partition tree on the current residuals every N boosting rounds.
More frequent refits keep the geometry aligned with the evolving residuals but
add computational cost.

- **Regression HPO range:** 10–200 (OAT rank-2; Optuna optimum varies widely)
- **Classification HPO range:** 20–200
- **Set to `None`:** disables all refits — the initial geometry is reused for
  all rounds. Fastest option; reasonable on small clean datasets.
- **Set low (10–20):** adapts geometry aggressively; good for complex nonlinear
  targets where residual structure changes rapidly.
- **Set high (100–200):** stable geometry; good for noisy data where gradient
  signal is weak.

---

## Sample reduction / expansion parameters

### `reduce_ratio` — float, default `0.8`

Fraction of training samples to retain after geometric reduction. The HVRT
partition tree selects a geometrically representative subset of this size.
Noise modulation (see `auto_noise`) adjusts this upward automatically on noisy
data: noisier data is reduced less aggressively.

- **Regression HPO range:** 0.3–0.95
- **Classification HPO range:** 0.4–0.95
- **Lower values:** faster training, more aggressive curation — only the most
  geometrically representative samples are kept. Risks losing signal on small
  datasets.
- **Higher values:** closer to training on the full set; slower but safer.
- **Dataset-dependent:** Optuna found 0.44 optimal for diabetes and 0.83 for
  Friedman-1, illustrating the wide variation across datasets.

---

### `expand_ratio` — float, default `0.1`

Fraction of `n` to generate as synthetic samples via the `generation_strategy`
kernel. Synthetic samples are generated within each geometric partition,
staying in the convex hull of real data.

- **Regression HPO range:** 0.0–0.4
- **Classification HPO range:** 0.0–0.3
- **Set to 0.0** to disable expansion entirely (safest default for large datasets
  where `n_reduced > min_train_samples`).
- **Most effective** on small datasets (n < 2 000) where the geometric regions
  are under-sampled.
- **Interaction with `auto_expand`:** when `expand_ratio=0` and
  `auto_expand=True`, expansion is triggered automatically only when
  `n_reduced < min_train_samples`.

---

### `method` — str, default `'variance_ordered'`

Reduction strategy used to select which samples to keep within each partition.

| Value | Description | Best for |
|---|---|---|
| `'variance_ordered'` | FPS weighted by within-partition variance | General regression (default) |
| `'orthant_stratified'` | Orthant-aware coverage of cross-polytope facets | MAE regression, PyramidHART geometry |
| `'kde_stratified'` | Centroid-distance quantile selection (L2) | Dense unimodal partitions |
| `'kde_stratified_l1'` | Centroid-distance quantile selection (L1) | HART/L1 geometry |
| `'residual_stratified'` | Joint centroid + residual magnitude ranking | High-residual-priority tasks |

**Advanced only.** The default works well with PyramidHART. `'orthant_stratified'`
is the correct companion for `GeoXGBMAERegressor`.

---

### `generation_strategy` — str, default `'simplex_mixup'`

Synthetic sample generation method used during expansion.

| Value | Description | Best for |
|---|---|---|
| `'simplex_mixup'` | Convex combination of two in-partition samples; parameter-free | General use (default) |
| `'laplace'` | Per-feature Laplace KDE centred on partition centroid | MAE / L1-natural geometry |
| `'epanechnikov'` | Epanechnikov kernel KDE | Smooth continuous features |

**Advanced only.** `'simplex_mixup'` is parameter-free and generates strictly
in-distribution samples. Only change this if you have a specific reason related
to the geometry of your data.

---

## Noise modulation parameters

These parameters govern GeoXGB's automatic noise detection and response system.
They are well-calibrated by default and should be left alone unless you have a
specific reason to change them.

### `auto_noise` — bool, default `True`

When `True`, GeoXGB estimates the signal-to-noise ratio of the current
gradient field at each refit and modulates `reduce_ratio` and `expand_ratio`
accordingly: noisy gradients → less aggressive reduction → more real samples
retained. This is particularly important for PyramidHART, which (unlike HVRT)
does not have a noise-invariance guarantee.

**Leave at `True`.** Disabling removes a key robustness mechanism.

---

### `noise_guard` — bool, default `True`

When `True`, if a freshly fitted partition tree detects near-zero gradient SNR,
the new geometry is discarded and the previous valid geometry is reused for the
current round. Prevents contaminating the training set with synthetic samples
generated from structureless gradient noise.

**Leave at `True`.** Disabling can cause training instability on noisy datasets.

---

### `refit_noise_floor` — float, default `0.05`

SNR threshold below which a refit is considered to have produced degenerate
geometry. A noise modulation value below this floor triggers the noise guard.

- **Raise (0.1–0.2):** more aggressive noise filtering; safer but may discard
  valid refits in genuine noisy-but-structured data.
- **Lower (0.01–0.03):** more permissive; allows geometry with weaker signal.
- **Leave at default** in almost all cases.

---

## Training set size parameters

### `auto_expand` — bool, default `True`

When `True` and `expand_ratio=0`, automatically expands the training set with
synthetic samples if `n_reduced < min_train_samples`. The expansion amount is
scaled by the noise modulation factor. Disabled automatically at large n (when
the real training set already exceeds `min_train_samples`).

**Leave at `True`** for small to medium datasets.

---

### `min_train_samples` — int, default `5 000`

Target training set size for `auto_expand`. If the reduced real set is smaller
than this, synthetic samples are generated to bring the total toward this value.
Capped at `5 × n_orig` to prevent overwhelming a very small dataset.

- **Increase** if you are on a large dataset and find that geometric partitions
  are under-populated (check `partition_trace()`).
- **Decrease** on very small datasets (n < 300) to prevent excessive synthetic
  expansion.

---

## Convergence and early stopping

### `convergence_tol` — float or None, default `None`

When set, GeoXGB monitors the mean absolute gradient (MAG) at each refit
interval. If the relative improvement over the last two intervals falls below
`convergence_tol`, training stops early and `model.convergence_round_` records
the round.

- **Enabling the Python path:** setting `convergence_tol` to any value forces
  the pure-Python backend (the C++ backend does not yet support this).
- **Typical values:** 1e-4 to 1e-3.
- **Leave at `None`** and simply run `n_rounds` to completion — with a low
  `learning_rate` there is no overfitting cost to running extra rounds.

---

### `adaptive_reduce_ratio` — bool, default `False`

When `True`, dynamically increases `reduce_ratio` when the gradient distribution
has a heavy tail (90th percentile / median > 1.5), retaining more samples during
rounds with large outlier gradients. Enabled by default in `GeoXGBMAERegressor`
where L1 gradients are more prone to tail effects.

- **Set to `True`** for MAE regression or datasets with heavy-tailed targets.
- **Leave at `False`** for standard squared-error regression.

---

## Geometry caching

### `cache_geometry` — bool, default `False`

When `True`, the initial HVRT geometry (partition tree, z-scores) is cached and
reused for all subsequent refits instead of being recomputed. Saves memory and
refit time on large datasets.

- **Set to `True`** when `n` is large and refit cost is noticeable.
- **Trade-off:** cached geometry cannot adapt to the evolving residual
  distribution. Best combined with a high `refit_interval` or `refit_interval=None`.

---

## Tree construction

### `tree_splitter` — str, default `'random'`

Splitting strategy for the weak learner decision trees. `'random'` selects the
best threshold among a random subset of features at each node (faster, implicit
regularisation). `'best'` evaluates all features exhaustively (slower, can
overfit at high depth).

- **Leave at `'random'`** for most tasks.
- **Switch to `'best'`** only if you observe systematically under-fitting on
  low-dimensional data (p < 5).

---

### `tree_criterion` — str, default `'squared_error'`

Splitting criterion for the weak learner. `'squared_error'` (L2) is the correct
choice for squared-error gradient boosting. `'friedman_mse'` applies a variance
improvement correction that can occasionally improve performance on small datasets.

**Leave at `'squared_error'`.**

---

### `variance_weighted` — bool, default `False`

When `True`, partition budgets during reduction are weighted by within-partition
gradient variance — high-variance partitions get proportionally more samples.
When `False`, budgets are proportional to partition size only (uniform).

- **Regression default `False`:** size-proportional is more stable.
- **Consider `True`** when you have a small number of high-signal partitions
  embedded in a large low-signal background.

---

## Feature weighting

### `feature_weights` — array-like of shape `(n_features,)` or None, default `None`

Per-feature scaling applied to `X` *before* the partition tree sees it. Features
with weight > 1 dominate the geometry; weight < 1 de-emphasises them. Tree
training always uses the original unscaled `X`.

Use `Gardener.recommend_feature_weights()` to derive weights from the divergence
between boosting and partition importances.

---

## Random state

### `random_state` — int, default `42`

Controls all stochastic elements: tree splits, partition tree construction,
synthetic sample generation, and FPS tie-breaking.

- Set a fixed value for reproducibility.
- HPO should use a fixed `random_state` per trial for fair comparison.
