# GeoXGB: Large-N Imbalanced Classification Gap

## Problem

GeoXGB underperforms XGBoost on a 594k-sample telecom churn dataset (19 features, 73% majority class). After tuning with partition_feature + global_geometry + HVRT partitioner: GeoXGB best = **0.8983**, XGBoost HPO = 0.9163, CatBoost HPO = 0.9163. Gap: **0.018**.

The gap is **not** due to irreducible error. The 100 worst GeoXGB predictions are genuinely hard (1% base churn rate for two-year contracts; XGBoost also misses them). The gap comes from **feature breadth**: GeoXGB uses 8–9/19 features while XGBoost uses 13/19. The AUC gap is largest in minority contract types where secondary features matter most.

## Architecture Constraints

- **Block cycling is non-negotiable.** At 594k samples, fitting HVRT on the full dataset each refit is too expensive. Block size ~4,700 samples.
- **Shallow trees (depth 3–5)** are the GeoXGB regime. XGBoost uses depth 6–8 on this dataset.
- **Random splitter** (ExtraTrees-style) is used for GBT weak learners, not best-split.

## Root Cause Analysis

1. **Block cycling + imbalanced classes**: Block size ~4,700 → only ~47 two-year churners per block → depth-3 trees can't find weak secondary features in such small minority slices.
2. **Feature breadth**: GeoXGB concentrates importance on Contract (dominant) + a few others. Features like StreamingTV, OnlineSecurity, DeviceProtection get zero importance despite having univariate AUC 0.55–0.62.
3. **Geometry is NOT the problem**: All 19 features are geometrically active in z-space with rich cooperation structure. The HVRT partitioning works correctly — the issue is downstream GBT tree fitting on small blocks.

## Solutions Attempted

### 1. Partition Feature (`partition_feature=True`)
**Idea**: Append the HVRT partition_id as an extra column for GBT trees. A single split on partition_id encodes multi-feature geometric neighborhood, letting shallow trees access interactions they'd otherwise miss.

**Result alone**: AUC = 0.8735 (WORSE than baseline 0.8841). With block-local HVRT, only 5 partitions exist — too coarse, just adds noise.

### 2. Global Geometry (`global_geometry_n=N`)
**Idea**: Fit HVRT once on a large subsample of the full dataset (not just the current block) before training begins. This gives stable, granular partitions (27–133 parts vs 5) that persist across all blocks.

**Result alone**: AUC = 0.8841 (neutral). Geometry is better but GBT trees can't see it without partition_feature.

### 3. Partition Feature + Global Geometry (COMBINED)
**Result**: This is the winning combination.

| Config | AUC | Parts | Feats | Time |
|---|---|---|---|---|
| baseline | 0.8841 | 5 | 8/19 | 7.6s |
| partition_feature only | 0.8735 | 44 | 8/19 | 9.1s |
| global_geom_50k only | 0.8841 | 24 | 8/19 | 7.5s |
| both_50k | 0.8852 | 24 | 8/19 | 9.9s |
| both_100k, d4, 2k rounds | 0.8927 | 27 | 8/19 | 19s |
| both_100k, d4, 4k rounds | 0.8980 | 27 | 9/19 | 40s |
| both_100k, d4, 5k rounds | 0.8992 | 27 | 9/19 | 56s |
| XGBoost HPO | 0.9163 | — | 13/19 | — |

**Why it works**: Global HVRT gives 27 stable partitions from 100k samples. Partition_feature lets GBT trees split on partition_id, encoding geometric neighborhood. Together, shallow trees can access partition-level interactions without needing deep splits.

**What's still missing**: Feature breadth (9/19 vs 13/19). More rounds help but with diminishing returns. The remaining 0.017 gap likely requires structural changes, not just hyperparameter tuning.

## Implementation Details

### C++ Changes (`cpp/src/geoxgb_base.cpp`)
- `global_geometry_n` config field: when > 0 and block cycling active, fits HVRT on a subsample of the full X_arg before training begins.
- `partition_feature` config field: when true, appends partition_id as column d+1 to all GBT training/prediction matrices.
- `part_hvrt()` lambda: returns `global_hvrt` when available, else `last_hvrt` — ensures partition_id assignments always use the most granular HVRT.
- `augment_with_part()` lambda: appends partition_id column using `part_hvrt()`.
- `predict_raw()`: augments test X with partition_id from persisted `last_hvrt_` (which is `global_hvrt` when it was active during training).
- `feature_importances()`: returns only the original d features (excludes partition_id column).

### Python Changes
- `partition_feature` and `global_geometry_n` added to `_PARAM_NAMES`, `__init__` signatures (regressor, classifier), and `_PYTHON_TO_CPP` mapping.
- Resolved in `make_cpp_config()` and passed through to C++ backend.

## Remaining Gap: 0.017 AUC

### Hypotheses for the remaining gap

1. **Feature breadth ceiling**: GBT with random splitter on ~4,700 samples per block may structurally limit how many weak features get discovered. XGBoost sees all 594k samples with best-split.

2. **Block cycling data efficiency**: Each block sees only ~0.8% of the data. Even with global geometry, the GBT trees within each block only learn from 4,700 samples. XGBoost learns from all samples simultaneously.

3. **Imbalanced Y representation per block**: ~47 churners per block (after reduce_ratio=0.8 → ~38). Weak features for minority class need more samples to be statistically significant.

4. **Imbalanced X representation per block**: Feature categories are also imbalanced — e.g., "Two-year" contracts are ~20% of data, "Fiber optic" internet ~44%, certain PaymentMethod values ~15%. In a block of ~4,700, rare X categories get very few representatives (~940 for a 20% category, ~700 for 15%). The *intersection* of rare X and rare Y is especially sparse: ~9 two-year churners per block. A random splitter cannot discover secondary feature effects (e.g., StreamingTV signal *within* two-year churners) from 9 samples. This X-imbalance compounds the Y-imbalance — it's not just that churners are rare in each block, it's that churners *of specific types* are vanishingly rare. XGBoost sees all 594k samples at once and can find these cross-category signals globally.

5. **Partition granularity**: 27 partitions may not be enough. With 200k global_geometry_n we get 133 partitions but AUC drops (too many partition_id values for shallow trees to split effectively).

### Possible directions to explore

- **Stratified block cycling**: Instead of random permutation blocks, stratify by key categorical features (or Y) so each block has representative coverage of rare X×Y combinations. E.g., ensure each block has proportional two-year churners.
- **X-aware block construction**: Use HVRT partition_ids from the global geometry to construct blocks that each contain samples from every partition — guaranteeing all geometric neighborhoods are represented.
- **Larger block sizes for imbalanced data**: Scale block size with class imbalance ratio. Current auto formula doesn't account for how many effective samples exist per X×Y stratum.
- **Best-split GBT**: Switch from random to best-split for weak learners (at speed cost). Random splits are especially wasteful when the informative splits involve rare category boundaries.
- **Feature subsampling / colsample**: Force trees to explore beyond the dominant features. Currently Contract absorbs most splits; colsample_bytree would force exploration of StreamingTV, OnlineSecurity, etc.
- **Stacking / residual boost**: Use partition_id predictions as a first stage, then boost residuals.
- **Partition-aware gradient weighting**: Weight gradients by partition minority enrichment.
- **Histogram-based splitting**: Use pre-binned histogram splits (like LightGBM) instead of random splits — more data-efficient for rare categories.
- **Synthetic minority oversampling within blocks**: When a block has very few minority-class samples, generate synthetic churners (via HVRT's expand mechanism) specifically for underrepresented X strata.

## Validated: HVRT-Guided Feature Selection

**Independent validation** (`benchmarks/hvrt_feature_selection_validation.py`) confirms HVRT geometry can deterministically select feature subsets per partition, and this improves performance on the churn dataset.

### Q1: Do partitions activate different features?
Yes. Mean pairwise Jaccard similarity of top-5 feature sets across partitions:
- Churn (166 partitions): **0.244** (76% different between partition pairs)
- Synthetic 50k (173 partitions): **0.159** (84% different)

With just top-3 features per partition, 17/19 churn features are already covered. At top-5, all 19/19. Each partition focuses on the features that are geometrically active in its neighborhood:
- Part 46 (69% churn, high-risk): StreamingTV, StreamingMovies, MultipleLines, OnlineBackup
- Part 57 (4% churn, low-risk): SeniorCitizen, Partner, Contract, PaymentMethod
- Part 13 (37% churn): PhoneService, SeniorCitizen, PaperlessBilling

### Q2: Does this give broader total feature coverage?
On churn, HVRT-selected trees achieve **19/19 features used** (same as full-feature). But the mechanism is different: each tree uses a different 8, collectively covering all 19.

### Q3: Does it improve predictions?
On churn (per-partition depth-4 trees, 3-fold CV):
- Full-feature per-partition: AUC = **0.8502**
- HVRT-selected (top-8) per-partition: AUC = **0.8622** (+0.012)

The feature selection acts as regularization — it prevents Contract from dominating every tree. Different partitions get different feature focus, matching their local geometry.

### Why this matters for the main gap
The core issue is GeoXGB uses 8-9/19 features while XGBoost uses 13/19. HVRT-guided feature selection is a fully deterministic, interpretable way to force broader feature exploration *that leverages block cycling rather than fighting it*. Different blocks → different dominant partitions → different feature subsets → broader coverage across the ensemble.

### Implementation direction
At each refit_interval, compute z-score variance per feature in the current block's HVRT partitions. Select top-k features for that block's trees. This is:
- **Deterministic** (z-score variance is a fixed function of the partition geometry)
- **Interpretable** (you can explain exactly why each feature was selected)
- **Complementary to block cycling** (different blocks naturally get different selections)
- **No new hyperparameters needed** (k can be derived from partition geometry, e.g., features with above-median variance)

## Dataset Details

- **Source**: Kaggle telecom churn competition (`data/train.csv`)
- **N**: 594,285 samples (train split: 396,190)
- **Features**: 19 (mix of categorical and continuous)
- **Target**: Binary churn (Yes/No), ~27% positive rate overall
- **Contract types**: Month-to-month (high churn ~42%), One-year (~11%), Two-year (~1%)
- **XGBoost HPO**: 0.9163 (30 trials, Optuna)
- **CatBoost HPO**: 0.9163 (identical to XGBoost)

## Benchmark Script

`benchmarks/kaggle_churn_part_feat_bench.py` — runs each config in a subprocess to avoid memory accumulation. Requires `data/train.csv`.
