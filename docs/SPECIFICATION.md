# GeoXGB Technical Specification

*Version 0.3.2 -- March 2026*

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The HVRT Subsystem](#2-the-hvrt-subsystem)
   - 2.1 [Whitening (StandardScaler)](#21-whitening-standardscaler)
   - 2.2 [Geometry Target Functions](#22-geometry-target-functions)
   - 2.3 [Target Blending (y_weight)](#23-target-blending-y_weight)
   - 2.4 [Quantile Binning](#24-quantile-binning)
   - 2.5 [Partition Tree Construction](#25-partition-tree-construction)
   - 2.6 [Reduction Strategies](#26-reduction-strategies)
   - 2.7 [Expansion / Synthetic Generation](#27-expansion--synthetic-generation)
   - 2.8 [HVRT Orchestration and Refit Fast Path](#28-hvrt-orchestration-and-refit-fast-path)
3. [The GeoXGB Boosting Loop](#3-the-geoxgb-boosting-loop)
   - 3.1 [Block Cycling](#31-block-cycling)
   - 3.2 [GBT Weak Learner Trees](#32-gbt-weak-learner-trees)
   - 3.3 [Resampling Integration](#33-resampling-integration)
   - 3.4 [Noise Estimation](#34-noise-estimation)
   - 3.5 [Adaptive y_weight Scheduling](#35-adaptive-y_weight-scheduling)
   - 3.6 [Convergence Detection](#36-convergence-detection)
   - 3.7 [kNN Target Assignment for Synthetic Samples](#37-knn-target-assignment-for-synthetic-samples)
   - 3.8 [Prediction Accumulation](#38-prediction-accumulation)
4. [Classification](#4-classification)
   - 4.1 [Binary Classification](#41-binary-classification)
   - 4.2 [Shared-Geometry Multiclass](#42-shared-geometry-multiclass)
5. [Interpretability API](#5-interpretability-api)
   - 5.1 [Geometry Accessors](#51-geometry-accessors)
   - 5.2 [Cooperation Matrix](#52-cooperation-matrix)
   - 5.3 [Cooperation Score](#53-cooperation-score)
   - 5.4 [Cooperation Tensor](#54-cooperation-tensor)
   - 5.5 [Local Model](#55-local-model)
   - 5.6 [Contributions (ContributionFrame)](#56-contributions-contributionframe)
   - 5.7 [Feature Importances](#57-feature-importances)
   - 5.8 [Noise Estimate and Sample Provenance](#58-noise-estimate-and-sample-provenance)
   - 5.9 [Multiclass Interpretability](#59-multiclass-interpretability)
6. [Performance Engineering](#6-performance-engineering)
   - 6.1 [BLAS-Accelerated Distance Matrices](#61-blas-accelerated-distance-matrices)
   - 6.2 [Cache-Friendly Memory Layout](#62-cache-friendly-memory-layout)
   - 6.3 [Pre-Allocated Scratch Buffers](#63-pre-allocated-scratch-buffers)
   - 6.4 [Histogram-Based Tree Splits](#64-histogram-based-tree-splits)
   - 6.5 [Bin Edge Injection](#65-bin-edge-injection)
   - 6.6 [Thread Pool Persistence](#66-thread-pool-persistence)
   - 6.7 [Partition-Level Batching in Interpretability](#67-partition-level-batching-in-interpretability)
7. [Configuration Reference](#7-configuration-reference)
8. [Experimental Features](#8-experimental-features)

---

## 1. Architecture Overview

GeoXGB is a geometry-aware gradient boosting framework.  It augments standard
gradient-boosted trees with a spatial resampling layer called **HVRT**
(Heteroscedasticity-Variance Resampling Transform) that partitions the
whitened feature space into geometrically coherent regions, selectively
retains informative samples, and optionally generates synthetic training
points to stabilize learning in small-data regimes.

```
                        fit(X, y)
                            |
                 +----------v-----------+
                 | Block Cycling (opt.)  |
                 | LCG permutation       |
                 +----------+-----------+
                            |
              +-------------v--------------+
              |       HVRT Subsystem       |
              |  Whiten -> Target -> Tree  |
              |  -> Reduce -> Expand       |
              +-------------+--------------+
                            |
              +-------------v--------------+
              |     Boosting Loop          |
              |  For each round:           |
              |   1. Compute gradients     |
              |   2. Bin reduced set       |
              |   3. Fit GBT weak learner  |
              |   4. Accumulate preds      |
              |                            |
              |  Every refit_interval:     |
              |   -> Re-run HVRT (fast)    |
              |   -> Noise guard check     |
              |   -> Convergence check     |
              +----------------------------+
```

**Language split.** The fitting engine is implemented entirely in C++17 with
Eigen3 for linear algebra and pybind11 for Python bindings.  Python handles
label encoding, multiclass orchestration, hyperparameter optimization
(Optuna), and the interpretability API surface.  There is no Python fallback
path; the C++ backend is the sole fitting engine.

**Build system.** scikit-build-core with CMake.  Eigen3 and pybind11 are
auto-fetched via FetchContent if not found system-wide.

---

## 2. The HVRT Subsystem

HVRT is a modular pipeline: **Whiten -> Target -> Bin -> Tree -> Reduce -> Expand**.
Each stage is a separate C++ class with a clean interface.

### 2.1 Whitening (StandardScaler)

**File:** `cpp/hvrt/src/whitener.cpp`

The whitener applies a per-feature z-score standardization (mean-center,
divide by population standard deviation), identical to sklearn's
`StandardScaler(with_std=True)`.

```
z_j = (x_j - mu_j) / sigma_j
```

where `sigma_j = sqrt(E[(x_j - mu_j)^2])` (population std, ddof=0).

**Mathematical justification.** Standardization is required so that the
pairwise product `z_i * z_j` is unit-free and comparable across feature
pairs.  Without it, a feature measured in thousands would dominate all
cooperation targets.  We use population std (not sample std) to match
sklearn conventions and ensure invertibility.

**Guard:** Features with `sigma_j < 1e-8` (effectively constant) are
assigned `sigma_j = 1.0`, producing a zero-mean but un-scaled column.
This prevents division-by-zero while preserving dimensionality.

**Complexity:** O(n * d) for both fit and transform.

### 2.2 Geometry Target Functions

**File:** `cpp/hvrt/src/target.cpp`

The geometry target is a scalar per sample that drives the partition tree's
splits.  GeoXGB supports four target functions, each capturing a different
notion of "feature cooperation" in z-space.

#### 2.2.1 Pairwise Cooperation (HVRT, default for d <= 50)

```
T_HVRT(x) = zscore( sum_{i<j} zscore(z_i * z_j) )
```

For each pair of features (i, j), compute the element-wise product of their
z-scores across all samples, z-score that product column, then sum all
d*(d-1)/2 z-scored products.  The final result is z-scored for
normalization.

**Geometric meaning.** High T_HVRT indicates a sample where many feature
pairs have aligned (same-sign) or anti-aligned (opposite-sign) z-scores.
The partition tree groups samples with similar cooperation patterns.

**Mathematical justification.**  The pairwise product `z_i * z_j` captures
second-order feature interactions.  Z-scoring each product before summation
ensures equal contribution regardless of the marginal variance of each pair.

**Complexity:** O(n * d^2).  Falls back to `compute_sum_target` (O(n*d))
when d > 50 to avoid quadratic scaling.

**Implementation optimization.** The inner loop is vectorized via Eigen:
```cpp
Eigen::MatrixXd prods = X_z.rightCols(d - i - 1).array().colwise()
                        * X_z.col(i).array();
```
This computes all products for column i against columns i+1..d-1 in one
Eigen expression, enabling SIMD auto-vectorization.

#### 2.2.2 HART (Absolute Pairwise Cooperation)

```
T_HART(x) = zscore( 0.5 * (||z||_1^2 - ||z||_2^2) )
```

This is algebraically equivalent to `sum_{i<j} |z_i| * |z_j|`, but
computed in O(n*d) using the L1/L2 norm identity rather than O(n*d^2)
explicit pair enumeration.

**Geometric meaning.** HART captures the total magnitude of cooperation
regardless of sign.  It is always non-negative.

#### 2.2.3 PyramidHART (Axis-Aligned Level Sets)

```
T_Pyramid(x) = zscore( |sum(z_i)| - ||z||_1 )
```

**Geometric meaning.** `A = |S| - L1` is always <= 0 (by triangle
inequality).  A = 0 exactly when all z-scores have the same sign (pure
cooperation).  The pyramid target partitions the feature space into
concentric level sets of this cooperation deficit.

**Mathematical justification.** Degree-1 homogeneous in z, so a single
50-sigma outlier feature shifts A minimally -- unlike pairwise products
which are degree-2.  This makes PyramidHART robust to heavy-tailed feature
distributions.

**Empirical status.** PyramidHART is the GeoXGBRegressor default
(`partitioner='pyramid_hart'`) based on benchmark results showing it
outperforms HVRT on most regression tasks.  The mathematical robustness
argument above is supported by the empirical data but the default was
ultimately chosen by benchmark, not by proof.

#### 2.2.4 FastHART (Row-Sum)

```
T_Fast(x) = zscore( sum(z_i) )
```

O(n*d).  Captures first-order directional alignment only.  Used as a fast
fallback when d > 50 (where pairwise targets are O(n*d^2)).

#### 2.2.5 Selective Pairwise Target (Approach 1)

**File:** `cpp/hvrt/src/target.cpp`, function `compute_selective_target`

At each HVRT refit, instead of using all d*(d-1)/2 pair products, only the
top-k pairs ranked by `|Pearson(zscore(z_a * z_b), zscore(residuals))|`
are included.  Default k = max(5, d*(d-1)/4).

**Rationale (empirical).** When d is small, most pairs are orthogonal to
the gradient direction.  Including them dilutes the signal.  Selective
targeting focuses HVRT splits on feature interactions that actually predict
the current residuals.

**Implementation.** Uses `std::partial_sort` to find the top-k pairs in
O(n_pairs * log(k)), avoiding a full sort.

### 2.3 Target Blending (y_weight)

**File:** `cpp/hvrt/src/target.cpp`, function `blend_target`

The partition tree's splitting target is a weighted combination of the
pure geometry target and a target-variable signal:

```
blended = (1 - y_weight) * zscore(geom_target)
         + y_weight * zscore(|y_norm - median(y_norm)|)
```

where `y_norm = (y - y_min) / (y_max - y_min)` maps y to [0,1], and the
absolute deviation from the median captures "extremeness" of y.

**Optional cross-term (S1):** When `blend_cross_term=True`:
```
blended += y_weight * zscore(geom_z * y_comp)
```
This biases splits toward regions where both geometric cooperation and
y-extremality co-occur.

**Mathematical justification.** The median-deviation transform makes the
y-component insensitive to location shift and robust to outliers (unlike
mean-deviation).  The [0,1] normalization prevents y's scale from
dominating the geometry signal.

### 2.4 Quantile Binning

**File:** `cpp/hvrt/src/binner.cpp`

The binner discretizes continuous features into `n_bins` quantile-based
bins for histogram-accelerated tree splitting.

**Algorithm:**
1. For each feature, compute b+1 quantile positions using
   `std::nth_element` (O(n) each, O(n*b) total per feature).
2. Ensure strict monotonicity by deduplicating edges.
3. Features with <= 2 unique values are flagged as "binary" and handled
   separately (simple threshold at midpoint).

**Transform:** `std::upper_bound` binary search maps each value to its
bin index in O(log b) per element.

**Output format:** `Eigen::Matrix<uint8_t, Dynamic, Dynamic, RowMajor>`.
Row-major layout ensures that `X_binned.row(i)` is contiguous in memory
-- critical for the scatter stage of histogram-based tree splitting
(Section 6.4).

**Two binning instances exist:**
- **HVRT binner** (16 bins): used for the partition tree.  16 bins halves
  the BFS scan cost vs 32 with negligible quality loss for the geometry
  partitioning task.
- **GBT binner** (64 bins, default): used for the gradient-boosted weak
  learner trees.  Higher resolution for accurate split thresholds on the
  prediction task.

### 2.5 Partition Tree Construction

**File:** `cpp/hvrt/src/tree.cpp`

The partition tree is a BFS-grown decision tree that splits on variance
reduction of the blended target.

#### Split Evaluation

**Variance reduction gain:**
```
gain = var(parent) * n_parent
     - var(left) * n_left
     - var(right) * n_right
```

Computed via running sums (Welford's method):
```
var_unnorm = sum_sq - sum^2 / n
gain = var_p - var_l - var_r
```

This avoids explicitly computing per-child variance, using only prefix
sums of target values and squared target values.

**Two-stage histogram algorithm:**

1. **Stage A (Scatter):** For each sample in the node, scatter its target
   value into per-bin accumulators (`bin_sum`, `bin_sum_sq`, `bin_cnt`).
   The inner loop iterates over features for each sample (sample-outer,
   feature-inner), exploiting the row-major layout of `X_binned` for
   sequential memory access.

2. **Stage B (Prefix scan):** For each feature, sweep bins left-to-right,
   accumulating `cum_sum`, `cum_sum_sq`, `cum_cnt`.  At each bin boundary,
   evaluate `variance_gain()` and track the best split.

**Split strategies:**
- `Best`: evaluates all valid thresholds per feature, selects the globally
  best (maximum gain).
- `Random`: collects all valid split positions per feature, picks one
  uniformly at random via an LCG.  This matches sklearn's
  `splitter="random"` and is the default for GBT weak learners.

**Auto-tuning (HVRT partition tree only):**
```cpp
min_samples_leaf = max(5, (d * 40 * 2) / 3)  // ~27*d for the "for_reduction" path
max_leaves = max(30, min(1500, 3*n / (2*msl)))
```
The `27*d` rule ensures each leaf contains enough samples for stable
per-feature z-score statistics.  The 1500-leaf cap prevents over-fragmentation.

**Mathematical origin of 27d:** With d features, a Ridge regression on
the additive + pairwise model in each partition has 1 + d + d*(d-1)/2
parameters.  For d=10, that is 56.  A minimum of ~2x parameters (112
samples) is needed for stable estimates.  The `(d*40*2)/3 ~ 27d`
approximation is a simplified, empirically tuned version of this argument.

**GBT weak learner tree config:**
```
n_partitions = 2 * 2^max_depth   (never constrains splitting)
min_samples_leaf = 5              (user-configurable)
max_depth = max_depth             (user-configurable, default 3-4)
auto_tune = false
split_strategy = Random           (ExtraTrees style)
```

### 2.6 Reduction Strategies

**File:** `cpp/hvrt/src/reduce.cpp`

Reduction selects a subset of training samples that preserves geometric
coverage.  Each strategy operates per-partition, with budgets allocated
proportionally (optionally variance-weighted).

#### 2.6.1 Budget Allocation

```
weight[p] = mean( mean(|X_z[p]|, axis=1) )   // if var_weighted
          = count(p)                           // otherwise
budget[p] = floor(weight[p] / total_weight * n_target)
```

Greedy +/-1 correction via fractional-remainder ranking ensures the total
exactly equals `n_target`.  Budgets are clamped to partition sizes for
reduction (can't select more than exist) but not for expansion.

**Rationale for variance weighting (empirical):**  Partitions with higher
mean |z| contain samples in the tails of the feature distribution.  These
regions are underrepresented by count but geometrically important.
Upweighting them improves coverage of the feature space boundary.  This was
validated empirically across multiple datasets.

#### 2.6.2 Variance-Ordered Reduction (Default)

Selects samples with the highest local k-NN distance variance within each
partition.

**Algorithm (for n <= 400):**
1. Compute pairwise squared-distance matrix via BLAS GEMM (Section 6.1).
2. For each sample, find k=5 nearest neighbors via `std::nth_element`.
3. Compute population variance of the k NN distances.
4. `std::partial_sort` to select the `budget` samples with highest local
   variance.

**Geometric meaning.** Samples with high local k-NN distance variance sit
near boundaries between dense and sparse regions -- exactly where the
model needs training signal.  Uniform-density regions have low variance
and are safely down-sampled.

**Mathematical justification.**  In a partition with piecewise-constant
density, the k-NN distance variance is zero in the interior and positive
at density transitions.  By selecting high-variance points, we
preferentially retain boundary samples.

**Empirical guards:**
- **High-retention shortcut (>= 65%):** When keeping >= 65% of a
  partition, the quality difference between variance-ordered and random
  selection is marginal.  An O(n) Fisher-Yates shuffle replaces the
  O(n^2) GEMM.  *Justification: empirically validated.*
- **Large-partition guard (N_CAP=400):** For partitions > 400 samples,
  the O(n^2) GEMM becomes expensive (n=4597 -> 168MB matrix, 320ms).
  Two strategies:
  - `budget >= 400`: stratified random selection (O(n)).
  - `budget < 400`: subsample 400 candidates via Fisher-Yates, run exact
    variance-ordered on the subsample.
  *Justification: N_CAP=400 keeps worst-case matrix at 1.25MB.*

#### 2.6.3 CentroidFPS (Farthest Point Sampling)

1. Seed: point closest to partition centroid.
2. Greedy: repeatedly select the point maximizing `min_dist_to_selected`.
3. Update `min_sq[i]` incrementally after each selection.

**Complexity:** O(n * budget) per partition.

**Geometric meaning.** Produces a maximal spread subset -- a discrete
approximation to the minimax facility location problem.

#### 2.6.4 MedoidFPS

Same as CentroidFPS but seeded from the medoid (point minimizing total
distance to all others).  Exact medoid for n <= 200, approximate
(sqrt(n)-nearest-to-centroid subset) for larger partitions.

#### 2.6.5 Orthant-Stratified Reduction

**Purpose:** y-aware geometric reduction for L1 (MAE) boosting.

1. Assign each sample to a z-space orthant via `sign(z_j - median_j)`.
2. Weight each orthant by `count * MAD(y_in_orthant)` (high-MAD orthants
   need more representation).
3. Within each orthant: sort by L1 distance from centroid, select at
   linearly-spaced positions (covers near-to-far evenly).

**Mathematical justification.** L1 gradients are {-1, 0, +1} -- the
standard variance-ordered approach cannot distinguish meaningful gradient
structure.  MAD-weighting orthants allocates budget to the geometric
regions where target variability is highest.

### 2.7 Expansion / Synthetic Generation

**File:** `cpp/hvrt/src/expand.cpp`

Expansion generates synthetic training samples to augment small datasets.
Generation is per-partition, preserving the geometric structure discovered
by the partition tree.

#### 2.7.1 Scott's Bandwidth

```
h_scott = n_partition ^ (-1 / (d + 4))
```

This is the multivariate Scott's rule for kernel density estimation.  It
appears in the Epanechnikov, Multivariate KDE, and Bootstrap strategies.

#### 2.7.2 Epanechnikov KDE (Default)

For each synthetic sample:
1. Pick a random training point from the partition.
2. For each feature j, draw three uniform samples `u1, u2, u3` in
   `[-h*sigma_j, +h*sigma_j]`.
3. The noise is: `u2` if `|u3| >= |u2|` and `|u3| >= |u1|`, else `u3`.

This rejection sampling produces the Epanechnikov kernel shape (parabolic),
which has the optimal asymptotic MSE among all second-order kernels.

#### 2.7.3 Simplex Mixup

```
x_syn = x_a + lambda * (x_b - x_a),   lambda ~ Uniform(0, 1)
```

Convex combination of two random partition members.  Produces samples that
lie on line segments between existing points.

**Mathematical justification.** Mixup is a provably consistent
regularization technique.  For partition-local generation, it preserves the
convex hull of each partition while smoothly interpolating the interior.
*This is the GeoXGBRegressor default* (`generation_strategy='simplex_mixup'`),
chosen empirically.

#### 2.7.4 Multivariate KDE

Sample from `N(x_base, h^2 * Sigma_partition)` where `Sigma_partition` is
the within-partition sample covariance.  Uses Cholesky decomposition
`L * z` (z ~ N(0,I)) for efficient sampling.

**Auto-selection:** Used when `n_partition >= max(15, 2*d)` (enough
samples for a stable covariance estimate).

#### 2.7.5 Laplace KDE

Per-feature Laplace distribution centered on the partition centroid with
scale = `1.4826 * MAD` (consistent estimator of sigma for Gaussian data).
Heavier tails than Gaussian KDE.

#### 2.7.6 Univariate Copula

1. Compute per-feature empirical CDF on a 2000-point grid.
2. Compute rank-based Pearson correlation matrix -> Cholesky.
3. Sample correlated normals -> map through Phi -> map through inverse
   empirical CDF.

Preserves marginal distributions and pairwise rank correlations.

### 2.8 HVRT Orchestration and Refit Fast Path

**File:** `cpp/hvrt/src/hvrt.cpp`

The HVRT class orchestrates the full pipeline.  On `fit()`:
1. Whiten X -> X_z.
2. Detect binary columns (via Binner).
3. Compute geometry target and cache it (`geom_target_cache_`).
4. Blend with y if `y_weight > 0`.
5. Build partition tree.
6. Prepare per-partition KDE parameters (Expander).

On `refit()` (called at every `refit_interval` during boosting):
1. **Skip** whitening (X_z unchanged).
2. **Skip** binning (X_binned_cache_ reused).
3. **Skip** geometry target (geom_target_cache_ reused).
4. Re-blend with new y (current gradients).
5. Re-build partition tree (bin edges reused from previous build).
6. Re-prepare Expander **only if partition assignments changed**.

**Cost savings per refit:**
- Whitening: saves O(n*d)
- Binning: saves O(n*d*log(n))  [nth_element per feature]
- Geometry target: saves O(n*d^2)  [pairwise] or O(n*d) [pyramid]
- Bin edges in tree: saves O(n*d*log(n))  [sort per feature]
- Expander (when stable): saves O(n_partitions * d * n_partition) [KDE fitting]

---

## 3. The GeoXGB Boosting Loop

**File:** `cpp/src/geoxgb_base.cpp`

### 3.1 Block Cycling

For large datasets (n > sample_block_n), GeoXGB divides the training set
into non-overlapping blocks of `sample_block_n` rows and cycles through
them at each refit boundary.

**Block size formula (auto):**
```python
if n <= 5000:
    return None   # disabled
ri_scale = clamp(refit_interval / 50, 0.5, 2.0)
block_size = max(2000, int(sqrt(n) * 15 * ri_scale))
```

**Justification (empirical).**  The sqrt(n) scaling was validated across 5
datasets, 6 block coefficients, and n up to 500k (Studies 1-4).  A
dimensionality-based formula (540*d) was tested but rejected because the
regularization benefit of smaller blocks (more geometric diversity from
cycling) outweighs the partition-quality gain of larger blocks at high d.

The `ri_scale` factor adjusts for refit frequency: fewer refits means fewer
block switches, so each block should be larger to ensure adequate data
coverage per cycle.

**LCG Permutation:**
```
seed = (random_state + epoch_seed) * 6364136223846793005 + 1442695040888963407
```
Knuth's LCG constants (period 2^64).  The Fisher-Yates shuffle using
`(lcg >> 33) % range` produces an unbiased permutation.  At epoch
boundaries, the permutation is regenerated with a new seed, ensuring
different block orderings across epochs.

**Block slicing:** Row copies from `X_arg` into contiguous `X_cur` matrix.
This ensures `X_cur` has contiguous memory for cache-efficient GEMM and
tree traversal, rather than working with strided views into X_arg.

### 3.2 GBT Weak Learner Trees

Each boosting round fits a single decision tree (PartitionTree) on the
current gradients of the reduced training set.

**Configuration:**
- `split_strategy = Random` (ExtraTrees style).  For each feature,
  collects all valid thresholds, picks one uniformly at random.
- `n_partitions = 2 * 2^max_depth` (never constrains splitting).
- `n_bins = 64` (GBT binner, higher resolution than HVRT's 16).
- Bin edges are injected from the pre-fitted GBT binner, skipping the
  O(n*d*log(n)) per-round sort.

**Why Random splits (empirical):** Random splits add regularization and
match the behavior of sklearn's `DecisionTreeRegressor(splitter="random")`.
In the GeoXGB setting, the reduced training set is already geometrically
curated by HVRT, so the marginal benefit of exhaustive splitting is small
relative to the regularization benefit of randomness.

### 3.3 Resampling Integration

At each refit boundary (every `refit_interval` rounds):

1. Compute gradients on full block: `grads = gradients(y_cur, preds_on_X)`.
2. Call `do_resample(X_cur, grads, i, last_hvrt)`:
   a. If `last_hvrt` exists: HVRT fast refit (Section 2.8).
   b. Else: fresh HVRT fit.
3. Compute noise modulation (Section 3.4).
4. Reduce: select `n_keep = n * eff_reduce` samples.
5. Expand (optional): generate synthetic samples up to `min_train_samples`.
6. Assign y-targets to synthetic samples via kNN (Section 3.7).
7. Return `ResampleResult{X_combined, y_combined, red_idx, noise_mod}`.

**Noise-modulated reduce ratio:**
```
eff_reduce = reduce_ratio + (1 - noise_mod) * (1 - reduce_ratio)
```
Noisier data (low `noise_mod`) -> higher effective reduce ratio -> keep
more samples.

**Mathematical justification.** When noise dominates, aggressive reduction
would discard signal-bearing samples randomly.  The modulation smoothly
interpolates between the configured ratio (clean data) and ratio=1.0
(all-noise: keep everything, let the tree ensemble average out noise).

### 3.4 Noise Estimation

**File:** `cpp/src/noise.cpp`

Estimates the signal-to-noise ratio of the current y-signal (gradients or
raw targets) using a k-NN local mean test.

**Algorithm:**
1. Subsample m = min(500, n) probe points.
2. Compute pairwise squared-distance matrix (m x n) via BLAS GEMM.
3. For each probe, find k=10 nearest neighbors via partial sort.
4. Compute local mean of y over the k neighbors.
5. Compute variance of local means vs. total y variance.
6. SNR = Var(local_means) / Var(y).
7. Noise modulation = clamp((SNR - 0.15) / 0.45, 0, 1).

**Mathematical justification.**  If y is pure noise, the local means
converge to the global mean (Var(local_means) -> 0, SNR -> 0).  If y has
spatial structure, nearby points have similar y-values (Var(local_means) ~
Var(y), SNR -> 1).  The linear mapping from [0.15, 0.60] to [0, 1]
was empirically calibrated.

**Thresholds (empirical):**
- 0.15: below this SNR, the data is effectively noise.  *Empirically tuned.*
- 0.45: the dynamic range of the mapping (0.60 - 0.15).  *Empirically tuned.*

### 3.5 Adaptive y_weight Scheduling

At each refit, the effective y_weight is scaled by the absolute Pearson
correlation between the cached geometry target and the current gradients:

```
rho = Pearson(geom_target_cache, gradients)
yw_eff = y_weight * |rho|
```

**Mathematical justification.** When gradients are uncorrelated with the
geometry target (rho ~ 0), the gradient signal is spatially unstructured
(likely noise).  Setting yw_eff -> 0 makes the partition tree
geometry-driven, avoiding over-fitting to noise.  When gradients align
with the geometry (high |rho|), the full y_weight is applied, letting the
tree focus on gradient-informative regions.

**Implementation.** Computed in one pass with no heap allocation:
```cpp
double rho = sum(geom[i] * (y[i] - y_mean)) / (y_std * (n-1))
```
The geometry target is already z-scored (~N(0,1)), so no normalization
is needed on the geom side.

### 3.6 Convergence Detection

At each refit boundary, computes `loss_now = mean(|gradients|)` and
tracks it in `convergence_losses_`.  When three consecutive measurements
exist:

```
rel_improvement = (loss_3_ago - loss_now) / (loss_3_ago + 1e-12)
if rel_improvement < convergence_tol:
    converge at round i
```

**Design choice.** Using 3-step lookback (not 1-step) avoids premature
convergence from single-round fluctuations caused by block cycling or
stochastic tree splits.  The 3-step window was chosen empirically.

### 3.7 kNN Target Assignment for Synthetic Samples

**File:** `cpp/src/geoxgb_base.cpp`, method `knn_assign_y`

Synthetic samples need y-targets.  GeoXGB uses k=3 inverse-distance
weighted (IDW) interpolation in HVRT z-space.

**Algorithm:**
1. Compute pairwise squared-distance matrix D (n_syn x n_red) via BLAS
   GEMM.
2. For each synthetic sample, find k=3 nearest real samples using a
   linear scan with insertion-sort on a 3-element buffer.
3. IDW: `y_syn[i] = sum(w_j * y_red[j]) / sum(w_j)`,
   where `w_j = 1 / (sqrt(d_j) + 1e-10)`.

**Why k=3 (empirical):** k=3 balances interpolation smoothness against
locality.  k=1 produces nearest-neighbor copies (no smoothing); k=5+
over-smooths in small partitions.

**Why linear scan instead of partial_sort (mathematical):** For k=3,
insertion-sort on a fixed 3-element buffer requires at most 2 comparisons
per candidate.  This eliminates the `O(n_red * log(k))` overhead of
`nth_element` and the `O(n_red)` index array allocation.  The 3-element
buffer fits in registers.

**Large n_red guard (KNN_RED_CAP = 5000):** When n_red > 5000, the
distance matrix exceeds practical memory bounds (10k x 70k = 5.6GB).
Fisher-Yates subsampling to 5000 candidates preserves approximate NN
quality: the HVRT-reduced set is geometrically dense, so a 5k subsample
of 70k still covers all partition neighborhoods.

**Persistent scratch buffer:** The distance matrix `knn_D_` is declared
as a `mutable` class member.  On subsequent calls with the same
dimensions, `Eigen::resize()` is a no-op, eliminating ~14MB of
VirtualAlloc/VirtualFree per call on Windows.

### 3.8 Prediction Accumulation

GeoXGB maintains two prediction vectors updated every round:

- `preds_on_X` (n-vector): predictions on the full current block `X_cur`.
- `preds` (n_reduced-vector): predictions on the reduced set `Xr`.

Each round:
```cpp
Eigen::VectorXd lp_Xr = wl.tree.predict(Xr);   // reduced set
Eigen::VectorXd lp_X  = wl.tree.predict(X_cur); // full block
preds       += learning_rate * lp_Xr;
preds_on_X  += learning_rate * lp_X;
```

**Sync at refit:** Real-sample predictions are copied from `preds_on_X`
via `preds[s] = preds_on_X[red_idx[s]]`.  Synthetic samples (which aren't
in X_cur) require `predict_from_trees()` -- but only for the synthetic
tail, not the full reduced set, cutting workload from ~4000 to ~500
predictions.

**Geometry persistence:** At the end of `fit_boosting()`:
```cpp
X_z_               = last_hvrt->X_z();           // (n_block, d)
partition_ids_     = last_hvrt->partition_ids();  // (n_block,)
train_predictions_ = predict_from_trees(X_arg, -1);  // (n_full,)
```
`train_predictions_` covers the full training set (not just the last
block) by running all accumulated trees over the complete `X_arg`.  This
ensures interpretability APIs always have predictions for all training
samples.

---

## 4. Classification

### 4.1 Binary Classification

**File:** `cpp/include/geoxgb/geoxgb_classifier.h`

Binary classification operates on log-odds.  A single tree ensemble is
trained using cross-entropy gradients:

```
init_pred = log(p / (1-p))     // log-odds of class frequency
p_hat = sigmoid(raw_prediction)
gradient = y - p_hat           // negative gradient of log-loss
```

**Class weighting:** When `pos_class_weight != 1.0`, gradients for
positive samples (y=1) are scaled by the weight.

**Predict:**
```
predict_proba(X) -> [1-sigmoid(raw), sigmoid(raw)]
predict(X) -> argmax(proba)
```

### 4.2 Shared-Geometry Multiclass

**File:** `cpp/src/geoxgb_multiclass.cpp`

For K > 2 classes, GeoXGB trains K class-specific tree ensembles that
share a single HVRT geometry.

**Key innovation:** Instead of K independent OvR binary classifiers
(each with its own HVRT), the multiclass classifier uses a single HVRT
whose y-signal is the L2 norm of the K-dimensional gradient vector:

```
y_combined[j] = sqrt( sum_k grad_k[j]^2 )
```

This focuses the partition tree on samples where the model struggles most
across all classes simultaneously.

**Initial HVRT:** Uses class 0's binary target (before gradients exist).
Subsequent refits use the combined gradient magnitude.

**Per-round training:**
```
for each round i:
    for each class k:
        grad_k = Y_onehot[:, k] - sigmoid(logits[:, k])
        fit tree_k on grad_k using shared reduced set Xr
        accumulate predictions
```

**Prediction:** Softmax over K raw logit vectors:
```
logits = init_preds + sum(lr * tree_k.predict(X)) for each k
proba = softmax(logits)    // row-wise: subtract max, exp, normalize
```

**Benefits over K independent OvR:**
- 1 HVRT fit/refit per cycle instead of K (dominant cost saving).
- Coherent partition structure across all classes.
- Combined gradient signal drives geometry toward the hardest boundaries.
- Empirically: +0.003 to +0.011 AUC vs XGBoost on multiclass benchmarks,
  compared to -0.015 AUC with the old OvR approach.

---

## 5. Interpretability API

All interpretability methods operate on the persisted geometry state
from the last HVRT fit: `X_z_`, `partition_ids_`, and (for
prediction-dependent methods) `train_predictions_`.

### 5.1 Geometry Accessors

The `_get_geometry()` Python method returns the C++ model handle and
training data.  It checks both `_cpp_model` (regressor/binary classifier)
and `_mc_cpp_model` (multiclass classifier).

C++ accessors exposed via pybind11:
- `X_z()`: whitened training data (n_block x d).
- `partition_ids()`: per-sample partition assignment (n_block,).
- `train_predictions()`: raw predictions on training data (n_block,).
  For multiclass: `train_predictions_multi()` returns (n_block x K).
- `to_z(X_new)`: whiten new query points using the fitted whitener.
- `apply(X_new)`: assign query points to partition IDs via tree traversal.

### 5.2 Cooperation Matrix

**Method:** `cooperation_matrix(X, feature_names=None)`

Returns per-sample and global feature cooperation matrices.

**Algorithm:**
1. Assign each query sample to a partition: `apply(X)`.
2. For each unique partition, compute Pearson correlation of z-scores:
   ```
   Z_centered = Z_partition - mean(Z_partition, axis=0)
   Z_normalized = Z_centered / std(Z_centered, axis=0)
   C = (Z_normalized^T @ Z_normalized) / n_partition
   ```
3. Map each query sample to its partition's correlation matrix.
4. Global matrix: partition-size-weighted average of all partition matrices.

**What it tells you.** `C[i,j] = +1` means features i and j are perfectly
correlated in the local neighborhood.  `C[i,j] = -1` means anti-correlated.
`C[i,j] = 0` means locally independent.  The key insight is that global
correlation may be zero while local correlation is strong (Simpson's
paradox).

**Properties guaranteed:**
- Diagonal entries = 1 (self-correlation).
- Symmetric: `C[i,j] = C[j,i]`.
- Bounded: `C[i,j] in [-1, 1]`.

### 5.3 Cooperation Score

**Method:** `cooperation_score(X)`

Returns a scalar per sample quantifying the partitioner-native cooperation
level.

**Formulas by partitioner:**

| Partitioner   | Formula                      | Range     |
|---------------|------------------------------|-----------|
| pyramid_hart  | `|S| - L1`                   | (-inf, 0] |
| hart          | `0.5 * (L1^2 - L2^2)`       | [0, inf)  |
| fasthvrt      | `L1`                         | [0, inf)  |
| hvrt (default)| `0.5 * (S^2 - L2^2)`        | R         |

Where `S = sum(z)`, `L1 = sum(|z|)`, `L2^2 = sum(z^2)`.

**What it tells you.** For PyramidHART: scores near 0 indicate high
feature cooperation (all z-scores same sign); large negative values
indicate competition (mixed signs).  Useful for identifying samples
where the model's geometric structure is strong vs. weak.

### 5.4 Cooperation Tensor

**Method:** `cooperation_tensor(X, feature_names=None)`

Returns per-sample three-way feature cooperation tensors.

**Formula:**
```
T[i,j,k] = E[z_i * z_j * z_k]   within partition
          = (Z^T @ Z @ Z) / n_partition  (using einsum)
```

**What it tells you.** `T[i,j,k]` identifies three-way feature
interactions.  A large `|T[a,b,c]|` means that feature c modulates the
a-b interaction.  This goes beyond pairwise SHAP interaction values --
GeoXGB can directly name the modulator variable.

**Properties:** Fully symmetric under index permutation.

### 5.5 Local Model

**Method:** `local_model(x, feature_names=None, min_pair_coop=0.10, alpha=1e-3, target_class=None)`

Fits a per-sample local additive + multiplicative polynomial explanation
in z-space.

**Algorithm:**
1. Identify x's partition via `apply(x)`.
2. Collect all training points in that partition: `Z_p`, `y_hat_p`
   (ensemble predictions, not raw targets).
3. Compute local Pearson correlation matrix `C_p`.
4. Select active pairs: `{(i,j) : |C_p[i,j]| >= min_pair_coop}`.
5. Build design matrix:
   ```
   M = [1 | Z_p | Z_p[:,i]*Z_p[:,j] for active (i,j)]
   ```
6. Solve Ridge regression: `theta = (M^T M + alpha*I)^{-1} M^T y_hat_p`.
7. Evaluate at x's z-scores: `prediction = theta^T @ m_x`.

**What it tells you:**
- `intercept`: baseline prediction in this partition.
- `additive[i]`: per-feature linear contribution in z-space.  Positive =
  increasing z_i increases prediction.
- `pairwise[(i,j)]`: interaction contribution.  Nonzero only for feature
  pairs with sufficient local correlation.
- `local_r2`: quality of the local polynomial fit (0-1).  Low R^2 means
  the ensemble's behavior in this partition is too complex for a linear +
  pairwise polynomial.

**Why Ridge on ensemble predictions (not raw y):**
The local model explains the *model's* behavior, not the data's structure.
Using `train_predictions()` as the target means the polynomial approximates
the ensemble's decision surface, giving a faithful local explanation
regardless of label noise.

**Multiclass:** Requires `target_class=k`.  Uses
`train_predictions_multi()[:, k]` as the Ridge target, explaining class k's
logit surface.

### 5.6 Contributions (ContributionFrame)

**File:** `src/geoxgb/contributions.py`

Batch version of `local_model()` that processes all samples efficiently.

**Key optimization:** Instead of fitting one Ridge per sample, fits one
Ridge per unique partition among the query samples.  For n_query=10000
and ~50 partitions, this reduces from 10000 to 50 Ridge solves.

**ContributionFrame fields:**
- `main: dict[str, ndarray(n,)]` -- per-feature additive contributions
  in prediction units (alpha_i * z_i for each sample).
- `interaction: dict[tuple[str,str], ndarray(n,)]` -- pairwise interaction
  contributions (beta_ij * z_i * z_j).  Only pairs with |C_p[i,j]| >=
  min_pair_coop in at least one partition are included.
- `intercepts: ndarray(n,)` -- per-sample polynomial intercept.
- `local_r2: ndarray(n,)` -- per-sample fit quality (used as reliability
  weight in plots).

**Plotting:**

`ContributionFrame.plot(feature, ...)`:
- 1D Nadaraya-Watson smoothed main-effect curve.
- Bandwidth: `range / 20` ("auto") or user-specified fraction of range.
- Kernel: Gaussian.  `w_i = exp(-0.5 * ((x_i - g) / h)^2)`.
- Weighted by `local_r2` so that high-quality partitions dominate.
- Optional: overlay interaction terms or other main effects.

`ContributionFrame.plot_interaction(feat_a, feat_b, ...)`:
- 2D Nadaraya-Watson smoothed interaction heatmap.
- Diverging colormap (RdBu_r) with 0-contour line marking the
  cooperation/competition boundary.

### 5.7 Feature Importances

**Method:** `feature_importances(feature_names=None)`

Aggregates impurity-based (variance reduction) importance across all
weak learner trees:
```
total[f] = sum( tree.importance[f] for all trees )
normalized[f] = total[f] / sum(total)
```

For multiclass: aggregates across all K * n_rounds trees.

### 5.8 Noise Estimate and Sample Provenance

**`noise_estimate()`:** Returns the initial noise modulation factor
(Section 3.4).  Range [0, 1]: 1.0 = clean data, 0.0 = pure noise.

**`sample_provenance()`:** Returns:
- `original_n`: samples passed to `fit()`.
- `reduced_n`: samples after initial HVRT reduction.
- `expanded_n`: always 0 (C++ path doesn't track cumulative expansion).
- `total_training`: equals `reduced_n`.
- `reduction_ratio`: `reduced_n / original_n`.

### 5.9 Multiclass Interpretability

All geometry-only methods (cooperation_matrix, cooperation_score,
cooperation_tensor) work identically for multiclass classifiers --
the shared HVRT geometry is class-agnostic.

Prediction-dependent methods (local_model, contributions) require
`target_class=k`:
- Uses `train_predictions_multi()[:, k]` (per-class raw logits) as
  the Ridge target.
- Each class gets its own set of contributions, explaining how features
  drive that class's logit relative to its partition baseline.
- Without `target_class`, a `ValueError` is raised for multiclass models.

---

## 6. Performance Engineering

### 6.1 BLAS-Accelerated Distance Matrices

The most performance-critical operation is pairwise squared-distance
computation, used in:
- Variance-ordered reduction (per-partition).
- Noise estimation (probe x full dataset).
- kNN target assignment (synthetic x reduced).

All three use the BLAS GEMM decomposition:
```
D[i,j] = ||X_i||^2 - 2 * X_i . X_j^T + ||X_j||^2
```

A single `D.noalias() = X * Y.transpose()` call delegates to Eigen's
BLAS backend (MKL, OpenBLAS, or Eigen's own GEMM), which is
auto-vectorized with AVX/SSE.  This replaces scalar `O(n^2*d)` loops
with a single vectorized matrix multiply, typically 10-20x faster for
the partition sizes GeoXGB encounters (n ~ 50-300, d ~ 8-15).

### 6.2 Cache-Friendly Memory Layout

**RowMajor distance matrices:** All pairwise distance matrices are stored
`Eigen::RowMajor`.  This ensures `D.row(i)` is contiguous in memory,
which is critical for:
- `std::nth_element` / `std::partial_sort` on D rows (sequential reads).
- The k=3 linear scan in kNN (streaming access pattern).
- Cache line utilization: a 64-byte cache line holds 8 doubles, covering
  8 columns of D per fetch.

**RowMajor binned matrices:** `X_binned` (uint8_t) is RowMajor so that
the scatter stage of histogram splitting reads `X_binned.row(idx)` as a
contiguous byte array across features -- one cache line per sample for
d <= 64.

**ColMajor Eigen default for feature matrices:** `X_z` (MatrixXd)
uses Eigen's default ColMajor, which is optimal for column-wise
operations (computing per-feature means, stds, products in target.cpp).

### 6.3 Pre-Allocated Scratch Buffers

**kNN distance matrix (`knn_D_`):** Declared as a `mutable` class member.
`Eigen::resize()` is a no-op when dimensions match, eliminating ~14MB
of VirtualAlloc/VirtualFree per call on Windows (n_syn=500, n_red=3500).

**predict_into():** The tree prediction method writes into a pre-allocated
`Eigen::VectorXd& out` buffer, avoiding a heap allocation per tree per
round.  In the boosting loop, a single `tmp(n)` buffer is reused across
all trees:
```cpp
Eigen::VectorXd tmp(n);
for (int t = 0; t < n_trees; ++t) {
    trees_[t].tree.predict_into(X, tmp);
    p.noalias() += trees_[t].lr * tmp;
}
```

**Histogram arrays:** In `evaluate_continuous_splits`, flat arrays
`bin_sum`, `bin_sum_sq`, `bin_cnt` of size `d_cont * nb_max` are
allocated once per node split, not per feature.

### 6.4 Histogram-Based Tree Splits

Both HVRT partition trees and GBT weak learner trees use histogram-based
splitting, which reduces per-split cost from O(n * d * log(n)) (sorted
scan) to O(n * d + d * n_bins) (scatter + sweep).

**Scatter phase:** O(n * d) -- each sample contributes to exactly one
bin per feature.  The inner loop accesses `X_binned` row-by-row
(RowMajor), ensuring sequential memory reads.

**Sweep phase:** O(d * n_bins) -- a prefix scan per feature with
constant-time variance gain evaluation at each bin boundary.

**OpenMP parallel scatter (when available):**
```cpp
#pragma omp parallel
{
    // thread-local histogram arrays
    #pragma omp for schedule(static)
    for (int si = 0; si < n_node; ++si) { ... }
    #pragma omp critical { merge; }
}
```
Each thread accumulates into local histograms, merging under a single
critical section.  Allocation overhead is proportional to d * n_bins per
thread (typically a few KB).

### 6.5 Bin Edge Injection

GBT weak learner trees receive pre-computed bin edges from the GBT
binner via `inject_bin_edges()`.  This skips the O(n*d*log(n))
per-feature sort that `PartitionTree::build()` would otherwise perform
to compute its own bin edges.

For a 1000-round model with d=10 features and n_reduced=300 samples,
this saves 1000 * 10 * 300 * log(300) ~ 24M comparisons.

### 6.6 Thread Pool Persistence

**File:** `cpp/hvrt/include/hvrt/threadpool.h`

Both the Expander and reduce() functions use persistent thread pools:
```cpp
static thread_local std::unique_ptr<ThreadPool> tl_pool;
if (!tl_pool || tl_pool->size() != n_threads)
    tl_pool = std::make_unique<ThreadPool>(n_threads);
```

Thread construction on Windows costs ~1ms per thread.  With 200+ resamples
per fit and 4 threads, this saves ~800ms of startup overhead.

### 6.7 Partition-Level Batching in Interpretability

`compute_contributions()` processes query samples in partition batches:
1. Assign all queries to partitions: `test_leaves = cpp.apply(X)`.
2. For each unique partition: fit one Ridge, evaluate all queries in
   that partition.

For n_query=10000 and ~50 unique partitions, this reduces from 10000
individual Ridge solves to 50 -- a 200x reduction in linear algebra
operations.

---

## 7. Configuration Reference

### GeoXGBConfig Defaults

| Parameter              | Default        | Origin     | Notes |
|------------------------|----------------|------------|-------|
| n_rounds               | 1000           | Empirical  | |
| learning_rate          | 0.2            | Empirical  | GeoXGBRegressor overrides to 0.02 |
| max_depth              | 3              | Empirical  | |
| min_samples_leaf       | 5              | Empirical  | For GBT weak learners |
| reduce_ratio           | 0.7            | Empirical  | GeoXGBRegressor uses 0.8 |
| expand_ratio           | 0.0            | Empirical  | GeoXGBRegressor uses 0.1 |
| y_weight               | 0.2            | Empirical  | Upper bound when adaptive=true |
| adaptive_y_weight      | true           | Mathematical | Scales by |rho| |
| refit_interval         | 20             | Empirical  | GeoXGBRegressor uses 50 |
| auto_noise             | true           | Empirical  | |
| noise_guard            | true           | Empirical  | |
| refit_noise_floor      | 0.05           | Empirical  | |
| auto_expand            | true           | Empirical  | |
| min_train_samples      | 5000           | Empirical  | |
| bandwidth              | -1.0 (auto)    | Mathematical | Scott's rule |
| n_bins                 | 64             | Empirical  | |
| variance_weighted      | true           | Empirical  | |
| convergence_tol        | 0.0 (disabled) | -- | |
| pos_class_weight       | 1.0            | -- | |
| sample_block_n         | -1 (disabled)  | Empirical  | Python 'auto' resolves to sqrt formula |
| loss                   | squared_error  | -- | |

### GeoXGBRegressor Defaults (overrides)

| Parameter              | Default          | Origin     |
|------------------------|------------------|------------|
| learning_rate          | 0.02             | Empirical  |
| max_depth              | 3                | Empirical  |
| reduce_ratio           | 0.8              | Empirical  |
| expand_ratio           | 0.1              | Empirical  |
| y_weight               | 0.25             | Empirical  |
| refit_interval         | 50               | Empirical  |
| partitioner            | pyramid_hart     | Empirical  |
| method                 | variance_ordered | Empirical  |
| generation_strategy    | simplex_mixup    | Empirical  |
| sample_block_n         | 'auto'           | Empirical  |

"Origin: Empirical" means the default was chosen based on benchmark
results across multiple datasets.  "Origin: Mathematical" means there is
a derivation (though the specific constant may be empirically tuned).

---

## 8. Experimental Features

### 8.1 Selective Pairwise Target (selective_target)

Replaces the static full-pairwise geometry target with the top-k pairs
most correlated with current residuals.  Disabled by default.

### 8.2 d_geom_threshold (Pure GBT Fast Path)

When `d <= d_geom_threshold`, skips HVRT entirely and runs pure GBT on
the full training set.  Disabled by default (threshold=0).

**Rationale:** For low-d data (d <= 8-12), the pairwise cooperation
target has too few feature pairs to align with the gradient direction.

### 8.3 Residual Correction Lambda

Per-partition z-space sub-split of residuals, applied to synthetic y
targets.  Shrunk correction: `delta = mean(resid[child]) * n / (n + lambda)`.
Disabled by default (lambda=0.0).

### 8.4 Y-Coupling Strategies

- **S1 (blend_cross_term):** Adds `x_z * y_comp` interaction to the blend
  target.  Disabled by default.
- **S2 (syn_partition_correct):** After kNN assignment, shifts synthetic y
  by the per-partition mean difference (real mean - kNN mean).  Disabled.
- **S3 (y_geom_coupling):** Blends residuals with geometry target before
  HVRT refit.  Disabled by default (alpha=0.0).

### 8.5 Adaptive Reduce Ratio

When `adaptive_reduce_ratio=True`, increase reduce_ratio for
heavy-tailed gradient distributions:
```
tail_ratio = p90(|gradients|) / median(|gradients|)
adapt_delta = clamp((tail_ratio - 1.5) / 20, 0, 0.15)
eff_reduce += adapt_delta
```

Recommended for L1 (MAE) boosting where sign gradients produce bimodal
distributions.
