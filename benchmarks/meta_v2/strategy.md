# GeoXGB Regression Meta-Analysis v2 — Strategy

## Why a full restart

The previous analysis (`meta_analysis.py`) had three fundamental problems:

1. **Stale API** — swept parameters no longer exist in the C++ backend
   (`tree_criterion`, `tree_splitter`, `method`, `assignment_strategy`,
   `convergence_tol`, `hvrt_auto_reduce_threshold`, `cache_geometry`,
   `feature_weights`, `generation_strategy`). Results were meaningless.

2. **No noise normalisation** — raw R² cannot be compared across datasets
   with different irreducible noise. A model achieving R²=0.75 on a dataset
   where the theoretical ceiling is 0.80 is far better than one achieving
   R²=0.85 on a noiseless dataset. Comparing them raw is misleading.

3. **No residual geometry** — results were single summary numbers with no
   understanding of *where* errors occur or *why* certain configurations
   perform differently. The geometric prior that makes GeoXGB distinct was
   not being interrogated at all.

---

## Primary metric: Noise-adjusted R²

For any dataset with known additive Gaussian noise σ applied to a signal
with variance Var(f(X)):

```
R²_ceiling = 1 − σ² / Var(y_noisy)
R²_adj     = R²_model / R²_ceiling
```

- R²_adj = 1.0 → model captures 100% of all explainable variance
- R²_adj = 0.5 → model explains half of what is theoretically possible
- R²_adj > 1.0 → impossible; indicates a ceiling estimation error

For friedman2 (noise=0), R²_ceiling = 1.0 and R²_adj = R²_model.

This metric is the primary comparison target throughout all phases.
Raw R², RMSE, and MAE are also recorded for secondary analysis.

---

## Baseline configuration

Uses current C++ `GeoXGBConfig` defaults (post-optimisation):

| Parameter           | Value  | Notes                                  |
|---------------------|--------|----------------------------------------|
| n_rounds            | 3000   | Established as fair vs XGBoost         |
| learning_rate       | 0.2    | C++ default; swept in Phase 2          |
| max_depth           | 3      | Updated default (was 4)                |
| min_samples_leaf    | 5      | Weak learner leaf size                 |
| reduce_ratio        | 0.7    | Fraction of real samples retained      |
| expand_ratio        | 0.0    | No synthetic expansion (baseline)      |
| y_weight            | 0.5    | Geometry / label signal blend          |
| refit_interval      | 20     | HVRT re-partitioned every 20 rounds    |
| auto_noise          | True   | SNR-modulated resampling volume        |
| noise_guard         | True   | Suppresses refits when SNR is low      |
| refit_noise_floor   | 0.05   | Minimum noise_mod to permit refit      |
| auto_expand         | True   | Synthetic expansion when n < threshold |
| min_train_samples   | 5000   | Expansion trigger                      |
| bandwidth           | auto   | Scott's rule per partition             |
| variance_weighted   | True   | Budget by geometric variance           |
| hvrt_min_samples_leaf | -1   | HVRT auto-tune                         |
| n_partitions        | -1     | HVRT auto-tune                         |
| n_bins              | 64     | GBT histogram bins                     |

---

## Datasets

All regression, y z-score normalised for comparable RMSE. Noise σ is the
exact value passed to `make_friedman1` / `make_regression`; R²_ceiling is
computed as `1 − σ² / Var(y_generated)`.

| Name           | n    | d  | n_informative | σ   | Characteristics               |
|----------------|------|----|---------------|-----|-------------------------------|
| friedman1      | 1000 | 10 | 10            | 1.0 | Nonlinear interactions        |
| friedman2      | 1000 | 4  | 4             | 0.0 | Zero noise, highly nonlinear  |
| reg_sparse     | 2000 | 30 | 8             | 0.5 | Sparse signal, low noise      |
| reg_large      | 5000 | 20 | 15            | 1.0 | Larger n, moderate noise      |

---

## Phase 1 — Noise robustness sweep

**Purpose**: Establish how GeoXGB degrades relative to the theoretical
ceiling as noise increases, compared with XGBoost at equal settings.

**Method**: Fix friedman1 dataset structure; vary noise σ ∈
{0, 0.5, 1.0, 2.0, 4.0, 8.0}. For each σ, fit both models with baseline
config on 5 folds × 10 seeds. Report mean ± std of R²_adj for each.

**Output**: `results/phase1_noise.csv`

**Key question**: Does GeoXGB's resampling mechanism help, hurt, or remain
neutral as noise increases? A model that degrades gracefully (R²_adj
constant across σ) is more robust than one that collapses.

---

## Phase 2 — OAT Primary (6 parameters)

**Purpose**: Identify which of the six highest-expected-impact parameters
actually drive performance on regression tasks with the C++ backend.

**Parameters swept**:

| Parameter      | Values                          | Rationale                              |
|----------------|---------------------------------|----------------------------------------|
| learning_rate  | 0.02, 0.05, 0.1, 0.2, 0.3      | Most sensitive in any boosting system  |
| max_depth      | 2, 3, 4, 5                      | Weak learner expressivity              |
| reduce_ratio   | 0.4, 0.5, 0.6, 0.7, 0.8, 0.9   | Core GeoXGB param; data volume retained|
| y_weight       | 0.2, 0.3, 0.5, 0.7, 0.8        | Geometry vs label signal blend         |
| refit_interval | 0, 5, 10, 20, 50                | 0 = disabled; geometry update frequency|
| auto_noise     | True, False                     | SNR-modulated volume                   |

**Method**: One at a time against baseline. 10 seeds × 4-fold CV per config,
all 4 datasets. Importance = mean |ΔR²_adj| from baseline across all
datasets and seeds.

**Output**: `results/phase2_oat_primary.csv`

---

## Phase 3 — Auto-Edge Extension

**Purpose**: If the best value of any numeric parameter lies at the boundary
of the tested range, extend the sweep automatically.

**Rules**:
- Best value == minimum of sweep → try ÷2, ÷4 (e.g. lr=0.02 → try 0.01, 0.005)
- Best value == maximum of sweep → try ×1.5, ×2 (e.g. lr=0.3 → try 0.45, 0.6)
- Integer parameters (max_depth): extend by ±1 step
- Parameters with natural bounds (reduce_ratio ∈ (0,1]): clamp accordingly
- Binary parameters (auto_noise): no extension

Extensions use same 10 seeds × 4-fold × 4 datasets as Phase 2.

**Output**: `results/phase3_oat_edge.csv`

---

## Phase 4 — OAT Secondary (6 parameters)

**Purpose**: Measure impact of secondary parameters independently, using
baseline for the primary parameters (not the HPO-optimal values, to avoid
confounding).

**Parameters swept**:

| Parameter          | Values               | Rationale                           |
|--------------------|----------------------|-------------------------------------|
| min_samples_leaf   | 3, 5, 10, 20         | Leaf size of GBT weak learner       |
| variance_weighted  | True, False          | Whether budget scales with variance |
| n_bins             | 32, 64, 128          | Histogram resolution                |
| n_partitions       | -1, 30, 60, 100      | -1 = auto-tune                      |
| expand_ratio       | 0.0, 0.1, 0.2        | Synthetic expansion fraction        |
| noise_guard        | True, False          | Refit suppression gate              |

**Output**: `results/phase4_oat_secondary.csv`

---

## Phase 5 — Pairwise (top-3 OAT parameters)

**Purpose**: Expose interaction effects between the three highest-importance
parameters found in Phases 2–3. Pure OAT misses cases where parameter A
is only beneficial when parameter B is also tuned.

**Method**: Read combined OAT importance from Phases 2 + 3. Pick top-3
numeric parameters by mean |ΔR²_adj|. Run full Cartesian product of their
sweep values (with edge extensions included). 5 seeds × 3-fold × 4 datasets.

Interaction score = std(ΔR²_adj) across all (v1, v2) combos — flat surface
= pure main effects, high spread = genuine interaction.

**Output**: `results/phase5_pairwise.csv`

---

## Phase 6 — Residual Geometry Analysis

**Purpose**: Understand *where* errors concentrate geometrically and *why*.
Three hypotheses tested:

1. **Geometric failure** — high residuals in partitions with degenerate cone
   (T-statistic collapsed, frac_in_cone < 0.05 or > 0.95). Remedy: adjust
   y_weight or increase HVRT min_samples_leaf.

2. **Data poverty** — high residuals in very small partitions. Remedy:
   reduce n_partitions or increase reduce_ratio.

3. **Residual signal** — high residuals in geometrically healthy, large
   partitions. Remedy: more rounds, deeper trees, or surgical re-splitting.

**Method**: Using best config from Phase 5, fit on each dataset (80/20
split, 5 seeds). Independently fit Python HVRT with matching y_weight to get
partition assignments and geometry_stats(). For each partition, compute:

- n_real: real training samples in partition
- mean_resid, std_resid, skew_resid: residual distribution
- abs_mean_resid: |mean| (signed bias)
- mean_abs_z, T_value, frac_in_cone: geometry metrics
- cone_degenerate: bool flag

Report Spearman r between residual severity and each geometric property.

**Output**: `results/phase6_residual.csv`

---

## Statistical rigour

- **Multiple seeds**: 10 seeds per OAT config avoids single-seed artifacts
  (the Opt#2 outlier that triggered this redesign was a single-seed artifact)
- **Cross-validation**: 4-fold for OAT, 3-fold for pairwise (speed trade-off)
- **Reporting**: mean ± std across seeds × folds, not just point estimate
- **Significance**: configs must beat baseline by > 1 std to be considered
  meaningfully better (informal threshold)

---

## Parallelism

Uses `multiprocessing.Pool` (Python stdlib). Each worker process has a
local copy of all datasets (generated from fixed seeds in the initialiser).
Tasks are tiny dicts; no large arrays are pickled per-task. Wall time
estimate at 12 workers: ~5 hours total.
