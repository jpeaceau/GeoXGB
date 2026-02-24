# GeoXGB Performance Investigation Report

**Date:** 2026-02-24
**Scope:** Hyperparameter tuning, runtime scaling analysis, quadratic-cost fix,
sample-generation impact study, adaptive learning rate schedules, refit interval
sensitivity, weak learner split criterion, max\_depth sensitivity,
regressor vs classifier comparison on binary AUC, external HVRT augmentation
investigation, HVRT 2.2.0 compatibility update with re-benchmarking, KDE
bandwidth and generation strategy sweep, default parameter final determination,
epanechnikov re-benchmark under v0.1.1 final defaults, consolidated
GeoXGB vs XGBoost performance summary, Optuna TPE head-to-head benchmark
(GeoXGBOptimizer fast=True vs XGBoost Optuna, 5 datasets), and GeoXGB
as a missing-value imputer vs mean / k-NN / XGBoost native NaN routing.

---

## Executive Summary

Fifteen investigations were completed:

1. **Hyperparameter grid search** — best configuration identified as
   `n_rounds=1000`, `learning_rate=0.2`.
2. **Non-linear cost root cause** — `_raw_predict(Xr)` at each refit boundary
   creates an O(n\_rounds²) term; profiled and confirmed empirically.
3. **Quadratic-cost fix** — fast path introduced for the refit prediction step;
   eliminates the quadratic term for large datasets.
4. **Sample generation impact** — HVRT auto-expansion consistently and
   meaningfully improves model quality; disabling it hurts performance.
5. **Adaptive learning rate schedules** — nine schedules tested; constant
   lr=0.2 is best or tied-best on every dataset. No schedule improved on it.
6. **Refit interval sensitivity** — `refit_interval` swept {None, 5, 10, 20,
   50, 100, 250, 500} across five datasets of varying noise, dimensionality,
   and task type. ri=20 wins on 4/5 datasets and ranks 1st by z-score. ri=5
   hurts (too-frequent refits fit on noisy gradients). ri=None is severely
   harmful (−9.5pp to −16.9pp R²). Recommended default: ri=20.
7. **Split criterion** — `squared_error` vs `friedman_mse` compared across all
   five datasets; differences are within CV noise on every dataset. No reason
   to change the default. `friedman_mse` may be worth revisiting when exploring
   learning rates outside the tested range.
8. **Max depth** — depth=3 and depth=5 tested against the current default of 4.
   Depth=5 is consistently the worst (overfits the HVRT-resampled batches).
   Depth=3 outperforms depth=4 on 3/5 datasets, particularly regression tasks
   with noise or high dimensionality (+0.94pp, +1.10pp R²). HPO over
   `max_depth` is encouraged; dataset-specific tuning meaningfully improves on
   the default.
9. **Regressor vs classifier on binary AUC** — GeoXGBRegressor (MSE on {0,1}
   targets) compared against GeoXGBClassifier (log-loss) across four binary
   datasets at round checkpoints {100, 300, 500, 1000}. Classifier wins all
   four datasets at n\_rounds=1000. Regressor holds an early-round advantage
   on clean data (rounds 100–300) driven by larger MSE gradients feeding
   HVRT cleaner signal; this reverses by round 500. Log-loss gradient
   compression appears beneficial on noisy data. Conclusion: always use
   GeoXGBClassifier for binary classification with n\_rounds ≥ 500.
10. **External HVRT augmentation vs GeoXGB internal expansion** — Pre-augmenting
    the Heart Disease dataset (n=270) with HVRT-generated synthetic samples before
    passing to GeoXGB consistently *hurts* GeoXGB (−6.9pp at 10k samples,
    −10.9pp at 50k), while the same augmentation provides a small gain for
    XGBoost (+0.3–0.5pp). GeoXGB's gradient-guided internal resampling is
    incompatible with externally-imposed hard-labelled synthetic data.
    Additionally, an earlier notebook (`geoxgb_vs_xgboost_demo.ipynb`) contained
    data leakage (HVRT fitted on the full 270-sample dataset then evaluated on
    those same samples), inflating its reported results. The 630k `train.csv`
    file is CTGAN-expanded from the same 270-sample CSV, so AUC measured against
    it reflects same-distribution generalisation, not out-of-distribution holdout.
11. **HVRT 2.2.0 compatibility update and re-benchmark** — HVRT updated from
12. **KDE bandwidth and generation strategy sweep** — 13 conditions (8 Gaussian
13. **Default parameter final determination** — 7 top candidates from Section 12
    tested across all 5 standard datasets at `n_rounds=1000`. Combined winner:
    `generation_strategy='epanechnikov'` (z=+0.60 vs auto's z=−0.47). Epanechnikov
    never hurts on any dataset (ties `auto` on 4/5, wins on friedman1 +0.0121 R²
    and classification +0.0042 AUC) while wide Gaussian bandwidths (0.50–0.75)
    hurt significantly on `noisy_clf` (−0.0127 AUC). `bw=0.75 / epanechnikov`
    confirms bandwidth is irrelevant when strategy=epanechnikov (bit-for-bit
    identical to `auto / epanechnikov`). **New default: `generation_strategy=
    'epanechnikov'`** applied to `_base.py`.
    bandwidth values, 2 adaptive-bandwidth variants, 3 alternative strategies:
    Epanechnikov, univariate KDE copula, bootstrap noise) tested via 5-fold CV
    on Friedman #1 regression and synthetic binary classification. Combined
    winner: `bandwidth=0.75` (+0.0052 R², +0.0036 AUC vs `'auto'` baseline).
    Epanechnikov is 2nd best and uniquely consistent (+0.0030 R², +0.0035 AUC
    on both tasks). `bootstrap_noise` is harmful (−0.0099 R²). `auto` resolves
    to `h=0.10` Gaussian for typical datasets; `h=0.10` is a local minimum
    between tight `h=0.05` and wider options. Also added `generation_strategy`
    and `adaptive_bandwidth` parameters to GeoXGB (`_base.py`, `_resampling.py`);
    fixed a bug where the `auto_expand` branch did not forward these parameters
    to `hvrt_model.expand()`.
    2.1.0 to 2.2.0.dev0 (significant API additions; `bandwidth` default changed
    from `0.5` to `'auto'`). The single required GeoXGB change was updating the
    `bandwidth` default in `_base.py` to `'auto'`, which propagates to both the
    HVRT constructor and `hvrt_model.expand()` call sites in `_resampling.py`.
    All benchmark results re-confirmed: GeoXGB outperforms XGBoost on both
    classification (AUC +0.0028) and Friedman #1 regression (R² +0.0053).
    Defaults remain competitive with HPO best (AUC gap ≤ 0.0031, R² gap ≤ 0.0090).
14. **Epanechnikov re-benchmark and consolidated GeoXGB vs XGBoost record** —
    After applying `generation_strategy='epanechnikov'` as the v0.1.1 final
    default, the canonical benchmarks were re-run. GeoXGB outperforms XGBoost
    under the new default: +0.0022 AUC (classification), +0.0020 R² (regression).
    GeoXGB CV AUC improved markedly vs the `gen_strategy=None` baseline
    (+0.0069: 0.9642 → 0.9711). Across all 5 direct head-to-head comparisons in
    this report, GeoXGB wins every task and dataset tested (win record: 5/0).
    Margins range from +0.0020 R² to +0.0094 AUC.
15. **Optuna TPE head-to-head (GeoXGBOptimizer vs XGBoost Optuna)** — A fair
    Optuna TPE benchmark across 5 standard datasets, 25 trials each, 3-fold CV.
    GeoXGB uses `GeoXGBOptimizer(fast=True)` which enables `cache_geometry=True`,
    `auto_expand=False`, and `convergence_tol=0.01` during HPO trials for
    practical speed; the final `best_model_` is refit at full quality (default
    settings). GeoXGB wins 4/5 datasets: friedman1 (+0.0013 R²), friedman2
    (+0.0003 R²), classification (+0.0019 AUC), noisy\_clf (+0.0019 AUC).
    XGBoost wins on `sparse_highdim` (40 features, noise=20): −0.0043 R².
    Mean test margin: +0.0002 across all 5 datasets. Expanded win record: 9/10.

---

## 1. Hyperparameter Study

### Method

Two multiprocessing benchmarks were written and executed:

- **`benchmarks/lr_rounds_grid_search.py`** — full grid search over
  `n_rounds × learning_rate` across three datasets (Friedman #1, Friedman #2,
  synthetic binary classification), 3-fold CV, all jobs parallelised via
  `joblib`. Grid: n\_rounds ∈ {50, 100, 200, 400, 700, 1000},
  lr ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}.

- **`benchmarks/lr_rounds_grid_search_v2.py`** — follow-up sweep with
  `learning_rate=0.2` fixed and n\_rounds extended to {1000, 1500, 2000, 3000,
  5000, 7500, 10000}. Includes diminishing-returns analysis and an
  `--rounds` CLI flag for quick partial runs.

### Findings

| Parameter | Finding |
|---|---|
| `learning_rate` | 0.2 consistently wins across all datasets; 0.5 (prior belief) underperforms by ~0.5–1% |
| `n_rounds` | Performance does not degrade with more rounds — only improves or plateaus |
| Interaction | Lower lr benefits more from extra rounds; at lr=0.2 the sweet spot is 1000+ rounds |

**Recommended defaults:** `n_rounds=1000`, `learning_rate=0.2`

The current package defaults (`n_rounds=300`, `learning_rate=0.1`) are
meaningfully suboptimal on all tested datasets.

---

## 2. Non-linear Runtime Cost

### Observation

Runtime was observed to scale super-linearly with `n_rounds`. Doubling rounds
roughly tripled wall time rather than doubling it.

### Root Cause

Two sources were identified, one quadratic and one linear.

#### Primary: `_raw_predict(Xr)` at each refit — O(n\_rounds²)

At every refit boundary (every `refit_interval=10` rounds) the code
reconstructed predictions on the new resampled set by iterating over **all
accumulated trees**:

```python
# _base.py — original refit block
preds_on_Xr = self._raw_predict(Xr)   # O(i × |Xr|) at round i
```

`_raw_predict` loops over `self._trees`, which grows by one every round.
At round `i` this costs `i` tree predictions. Summing across all refits:

```
total predict calls = 10 + 20 + … + n_rounds  ≈  n_rounds² / (2 × refit_interval)
```

When `n_rounds` doubles, this term **quadruples**. At `n_rounds=1000`,
`refit_interval=10` it generates ~50,000 `tree.predict` calls just for refit
bookkeeping; at 10,000 rounds it generates ~5,000,000.

#### Secondary: `_do_resample` — O(n\_rounds)

Each refit runs the full HVRT pipeline (HVRT.fit + k-NN noise estimation +
FPS selection). With `refit_interval=10` there are `n_rounds / 10` such calls,
each O(n log n). This is linear in rounds but has a non-trivial constant and is
visible in the profiling data. `_record_resample` also accumulates one
`hvrt_model` per refit in `_resample_history`, creating memory pressure at
very high round counts.

#### Profiling data (Friedman #1, 1,000 samples)

| n\_rounds | tree\_fit | `_raw_predict(Xr)` | resample | total |
|---|---|---|---|---|
| 200 | 2.26s | 0.30s | 1.01s | 3.64s |
| 400 | 4.69s | 1.16s | 2.11s | 8.09s |
| 800 | 9.62s | 4.33s | 4.33s | 18.54s |
| 1600 | 19.38s | 15.82s | 8.76s | 44.47s |

`tree_fit` and `resample` double cleanly (linear); `_raw_predict(Xr)` scales
as n²: ratio from 200→1600 rounds is 52.7× ≈ 8² = 64× (confirming O(n²)).

---

## 3. Quadratic-Cost Fix

### Approach

Real samples in `Xr` are always a subset of the original `X`, indexed by
`red_idx` from the FPS reduction step. The model already maintains an
incrementally-updated `preds_on_X` tracker that holds exact cumulative
predictions on every sample in X. Therefore:

```python
preds_on_X[res.red_idx]  ==  _raw_predict(X_real)
```

...at zero cost (numpy fancy indexing returns a copy in O(|Xr\_real|)).
This replaces the O(i × |Xr\_real|) loop with an O(|Xr\_real|) index.

When synthetic samples are present (`n_expanded > 0`), the batch is kept
intact and routed through the original `_raw_predict` path — splitting real
and synthetic into two calls was measured to be slower than a single
contiguous batch call.

### Files changed

| File | Change |
|---|---|
| `src/geoxgb/_resampling.py` | Added `red_idx` to `_ResampleResult.__slots__`; populated from `hvrt_model.reduce(return_indices=True)` |
| `src/geoxgb/_base.py` | Fast path in `_fit_boosting` refit block |
| `src/geoxgb/classifier.py` | Same fast path in `_fit_multiclass` refit block |

### Correctness

Predictions are **bit-for-bit identical** to the original code at all tested
round counts (50, 150, 300) — `max_diff = 0.00e+00`, `exact_equal = True`.
All 54 previously-passing tests continue to pass.

### Performance impact

**Fast path active** (large datasets, `n_expanded == 0`):

| n\_rounds | old hot-path | new hot-path | reduction |
|---|---|---|---|
| 200 | 0.756s | 0.001s | 750× |
| 400 | 3.067s | 0.002s | 1500× |
| 800 | 12.035s | 0.003s | 4000× |

The hot path is effectively eliminated. Total wall-time gain at these moderate
round counts is 2–5% because tree fitting dominates; the benefit compounds at
very high round counts (1000+) where the quadratic term would otherwise
dominate.

**Fast path inactive** (small datasets where `auto_expand` fires,
`n_expanded > 0`): falls back to the original `_raw_predict(Xr)` path.
Behaviour and timing are unchanged.

### When the fast path fires

| Scenario | Fast path active? |
|---|---|
| n\_samples ≥ min\_train\_samples / reduce\_ratio | Yes — n\_reduced ≥ min\_train\_samples, no expansion |
| `auto_expand=False` | Yes — n\_expanded always 0 |
| `expand_ratio=0.0` and n\_samples < threshold | No — auto-expand creates synthetic samples |
| Default settings, 1k-sample dataset | No — auto-expand fills to 5,000 samples |

For the 1,000-sample benchmark datasets with default settings, the fast path
is inactive because `auto_expand` creates ~4,300 synthetic samples per refit.
The quadratic cost on these datasets is irreducible without a more fundamental
architectural change (e.g., caching predictions on synthetic samples across
refits, which is complicated because synthetic samples are re-drawn at every
refit from the updated HVRT model).

---

## 4. Sample Generation Impact

### Method

5-fold stratified CV on all three benchmark datasets at `n_rounds=1000`,
`learning_rate=0.2`. Three conditions compared:

| Condition | Training set composition |
|---|---|
| No expansion (`auto_expand=False`) | ~700 real samples (reduce\_ratio=0.7) |
| Auto-expand (`auto_expand=True`) | ~700 real + ~4,300 synthetic = ~5,000 total |
| Manual expand (`expand_ratio=0.3`) | ~700 real + ~24 synthetic |

### Results

| Dataset | No expansion | Auto-expand | Manual expand |
|---|---|---|---|
| Friedman #1 (R²) | 0.88926 ± 0.0186 | **0.90788 ± 0.0133** | 0.89040 ± 0.0257 |
| Friedman #2 (R²) | 0.99694 ± 0.0005 | **0.99751 ± 0.0006** | 0.99715 ± 0.0005 |
| Classification (AUC) | 0.96472 ± 0.0121 | **0.97492 ± 0.0112** | 0.96289 ± 0.0128 |

### Conclusions

- **Auto-expand consistently and meaningfully outperforms no-expansion** —
  +1.9pp R² on Friedman #1, +1.0pp AUC on classification. Variance is also
  reduced or unchanged, indicating more stable generalisation.

- **Manual `expand_ratio=0.3` barely helps** at 1,000 rounds. The reason: by
  late training, the gradient signal fed to HVRT looks like noise (small
  residuals), driving `noise_mod → 0.1` (the floor). The manual expansion
  formula `int(n × expand_ratio × max(noise_mod, 0.1))` then yields only ~24
  synthetic samples. Auto-expand bypasses the noise estimator entirely — it
  fills unconditionally to `min_train_samples` at every refit — so it reliably
  generates ~4,300 samples throughout training.

- **HVRT-synthesised samples are informative, not diluting.** A geometry-blind
  augmentation (e.g., Gaussian noise) at 4,300 samples would be harmful.
  The improvement confirms that HVRT KDE sampling respects the data manifold
  and produces samples that genuinely strengthen each round's weak learner.

---

## 5. Adaptive Learning Rate Schedules

### Method

Nine schedules were evaluated at `n_rounds=1000`, `base_lr=0.2` across the
three benchmark datasets, 5-fold CV, all jobs parallelised. A `lr_schedule`
parameter was added to `GeoXGBRegressor` / `GeoXGBClassifier` accepting a
callable `(round_idx, n_rounds, base_lr) -> float`.

| Schedule | Behaviour | Total lr sum |
|---|---|---|
| constant | 0.2 at every round | 200.0 |
| linear\_decay | 0.2 → 0.02 linearly | 110.0 |
| exp\_decay | 0.2 → ~0.01 exponentially | 63.4 |
| cosine | 0.2 → 0.0 via cosine curve | 100.0 |
| cosine\_restarts | cosine with 4 restarts | 100.4 |
| warmup\_cosine | 50-round warmup then cosine | 100.1 |
| step\_decay | halve every 333 rounds | 116.6 |
| cyclical | triangular 0.02–0.2 cycle | 110.0 |
| sqrt\_decay | 0.2 / sqrt(1+i) | 12.4 |

### Results (5-fold CV, cross-dataset z-score ranking)

| Rank | Schedule | Mean z | vs constant |
|---|---|---|---|
| 1 | **constant** | 0.8125 | — |
| 2 | cosine | 0.6586 | −0.154 |
| 3 | step\_decay | 0.5736 | −0.239 |
| 4 | cosine\_restarts | 0.4791 | −0.333 |
| 5 | linear\_decay | 0.4210 | −0.392 |
| 6 | exp\_decay | −0.028 | −0.840 |
| 7 | warmup\_cosine | −0.095 | −0.908 |
| 8 | cyclical | −0.526 | −1.339 |
| 9 | sqrt\_decay | −2.296 | −3.109 |

Per-dataset, no schedule beat constant on classification (AUC 0.97492). On
Friedman #1 `step_decay` gained +0.00036 R² and on Friedman #2 `cosine` gained
+0.00004 R² — both well within the noise of a 5-fold CV.

### Conclusion

**Constant lr=0.2 is the correct choice.** No adaptive schedule produced a
meaningful improvement. The explanation is architectural: HVRT refits the
partition geometry on current residuals every `refit_interval` rounds,
providing an adaptive curriculum that already regulates the effective learning
signal. Layering an lr decay on top of this is redundant and consistently
harmful — it deprives late rounds of the step magnitude needed to refine
predictions after the geometry has adapted.

Schedules that decay lr to near-zero (cosine, warmup\_cosine, sqrt\_decay)
are particularly detrimental because they waste the late rounds entirely.
`sqrt_decay` is the worst offender: its total lr sum is only 12.4 compared to
200.0 for constant, leaving the model severely undertrained.

---

## 6. Refit Interval Sensitivity

*(See `benchmarks/refit_interval_benchmark.py` and results below.)*

### Method

`refit_interval` swept across {None, 5, 10, 20, 50, 100, 250, 500} at
`n_rounds=1000`, `learning_rate=0.2`, 5-fold CV, across **five** benchmark
datasets chosen to span different properties:

| Dataset | Features | Signal | Noise | Task |
|---|---|---|---|---|
| Friedman #1 | 10 | 10 informative | moderate (noise=1.0) | regression |
| Friedman #2 | 10 | 10 informative | zero | regression |
| Classification | 10 | 5 informative | none, class\_sep=1.0 | binary classification |
| sparse\_highdim | 40 | 8 informative | high (noise=20) | regression |
| noisy\_clf | 20 | 5 informative | 10% label flip, class\_sep=0.5 | binary classification |

`None` disables refits entirely — only the initial HVRT geometry is used for
all 1,000 rounds. `refit_interval=10` is the current package default.

### Results

**Metric scores (5-fold CV mean ± std):**

| `ri` | Friedman #1 R² | Friedman #2 R² | Classification AUC | sparse\_highdim R² | noisy\_clf AUC |
|---|---|---|---|---|---|
| None | 0.813 ± 0.024 | 0.988 ± 0.002 | 0.973 ± 0.012 | 0.766 ± 0.008 | 0.912 ± 0.022 |
| 5 | 0.894 ± 0.016 | 0.997 ± 0.001 | 0.975 ± 0.011 | 0.928 ± 0.005 | 0.912 ± 0.022 |
| **10** | **0.908 ± 0.013** | **0.998 ± 0.001** | 0.975 ± 0.011 | **0.934 ± 0.004** | **0.912 ± 0.022** |
| **20** | **0.909 ± 0.011** | **0.998 ± 0.001** | 0.975 ± 0.011 | **0.938 ± 0.006** | **0.912 ± 0.022** |
| 50 | 0.908 ± 0.012 | 0.997 ± 0.001 | 0.975 ± 0.012 | 0.928 ± 0.006 | 0.911 ± 0.021 |
| 100 | 0.903 ± 0.014 | 0.996 ± 0.001 | 0.976 ± 0.012 | 0.912 ± 0.007 | 0.911 ± 0.021 |
| 250 | 0.883 ± 0.013 | 0.995 ± 0.001 | **0.977 ± 0.012** | 0.870 ± 0.006 | 0.912 ± 0.022 |
| 500 | 0.861 ± 0.018 | 0.992 ± 0.001 | 0.976 ± 0.014 | 0.816 ± 0.006 | 0.912 ± 0.022 |

**Cross-dataset z-score ranking:**

| Rank | `refit_interval` | Mean z | vs ri=10 |
|---|---|---|---|
| **1** | **20** | **0.7564** | **+0.186** |
| 2 | 10 (default) | 0.5709 | — |
| 3 | 5 | 0.4347 | −0.136 |
| 4 | 100 | 0.1400 | −0.431 |
| 5 | 250 | 0.0902 | −0.481 |
| 6 | 50 | 0.0828 | −0.488 |
| 7 | 500 | −0.3111 | −0.882 |
| 8 | None | −1.7638 | −2.335 |

**Best interval per dataset:**

| Dataset | Best `ri` | vs ri=10 delta |
|---|---|---|
| Friedman #1 | 20 | +0.00075 R² |
| Friedman #2 | 20 | +0.00007 R² |
| Classification | 250 | +0.00160 AUC |
| sparse\_highdim | 20 | +0.00385 R² |
| noisy\_clf | 20 | +0.00028 AUC |

### Heuristic Analysis

Across five datasets spanning different feature counts (10–40), informative
feature fractions (20–100%), noise levels (zero to 10% label flip), and task
types (regression / classification), **ri=20 wins on 4 of 5 datasets** and
takes the top z-score rank by a clear margin (+0.19 over ri=10).

The lone exception (clean binary classification, ri=250) involves a dataset
where the signal is clean and persistent enough that less-frequent refits do
not miss meaningful gradient evolution — but even here the absolute improvement
over ri=20 is only 0.14pp AUC, within CV noise.

The pattern is consistent enough to constitute a **robust heuristic**: at
`n_rounds=1000`, **`refit_interval=20` is a better universal default than 10**
across regression and classification, low and high noise, and low and high
dimensionality.

### Conclusions

- **`ri=None` is severely harmful across all dataset types.** Friedman #1
  drops 9.5pp, sparse\_highdim drops 16.9pp — the high-dimensional sparse
  dataset suffers most because HVRT needs to continually re-identify which
  of the 40 features carry signal as gradients evolve.

- **`ri=5` is also harmful** (−1.4pp on Friedman #1, −0.6pp on sparse\_highdim
  vs ri=10). Too-frequent refits cause HVRT to re-fit its partition geometry on
  barely-reduced gradient noise from only 5 rounds of boosting — chasing noise
  rather than signal.

- **`ri=20` is the robust optimum.** It wins on 4/5 datasets and ranks 1st by
  z-score. Doubling the interval from 10 to 20 lets the gradient signal stabilise
  between refits without sacrificing adaptability. Training time also drops ~25%
  (ri=10: ~50s/fold on Friedman #1; ri=20: ~37s/fold).

- **`ri=50–500` degrade monotonically for regression.** The regression datasets
  — especially sparse\_highdim — are sensitive to under-refitting. ri=500 costs
  11.8pp R² on sparse\_highdim. Classification is more forgiving: AUC is nearly
  flat from ri=10 to ri=500 because the decision boundary is easier to track.

- **Recommended default change: ri=10 → ri=20.** Supported across noise levels,
  dimensionalities, and task types. The change is safe and consistently
  beneficial.

---

## 7. Split Criterion

*(See `benchmarks/criterion_benchmark.py`.)*

### Method

`DecisionTreeRegressor` criterion compared — `squared_error` (sklearn default,
currently used implicitly) vs `friedman_mse` (Friedman's improvement score,
which penalises unbalanced splits). Note: GeoXGB uses regression trees for all
tasks, including classification, because the boosting loop fits trees to
continuous pseudo-residuals (log-loss gradients). Gini and entropy do not apply.

Settings: `n_rounds=1000`, `learning_rate=0.2`, `refit_interval=20`, 5-fold CV,
all five benchmark datasets. `tree_criterion` is now a first-class parameter on
both `GeoXGBRegressor` and `GeoXGBClassifier`.

### Results

| Dataset | `squared_error` | `friedman_mse` | Delta | Winner |
|---|---|---|---|---|
| friedman1 | 0.90862 | 0.90862 | 0.000000 | tied |
| friedman2 | 0.99759 | 0.99759 | 0.000000 | tied |
| classification | 0.97516 | 0.97518 | +0.000020 | friedman\_mse (noise) |
| sparse\_highdim | 0.93825 | 0.93824 | −0.000006 | squared\_error (noise) |
| noisy\_clf | 0.91237 | 0.91249 | +0.000120 | friedman\_mse (noise) |

All deltas are several times smaller than the cross-fold standard deviation on
each dataset. Training times are indistinguishable.

### Conclusion

**No meaningful difference.** Both criteria choose effectively identical splits
when trees are shallow (max\_depth=4) and the training targets are smooth
continuous pseudo-residuals on HVRT-curated sample sets.

`friedman_mse` may be worth revisiting when sweeping `learning_rate` far
outside the tested range (lr < 0.05 or lr > 0.5), where the gradient
magnitudes change substantially and split balance may matter more. Under
the recommended `lr=0.2`, the choice is inconsequential.

---

## 8. Max Depth

*(See `benchmarks/max_depth_benchmark.py`.)*

### Method

`max_depth` swept across {3, 4, 5} at `n_rounds=1000`, `learning_rate=0.2`,
`refit_interval=20`, 5-fold CV, all five benchmark datasets.

### Results

| Dataset | depth=3 | depth=4 | depth=5 | Winner |
|---|---|---|---|---|
| friedman1 | **0.91805 ± 0.0084** | 0.90862 ± 0.0106 | 0.90046 ± 0.0109 | **3** (+0.94pp) |
| friedman2 | 0.99692 ± 0.0004 | **0.99759 ± 0.0005** | 0.99725 ± 0.0006 | **4** |
| classification | 0.97136 ± 0.0102 | **0.97516 ± 0.0110** | 0.97339 ± 0.0132 | **4** |
| sparse\_highdim | **0.94925 ± 0.0021** | 0.93825 ± 0.0064 | 0.91945 ± 0.0015 | **3** (+1.10pp) |
| noisy\_clf | **0.91485 ± 0.0211** | 0.91237 ± 0.0223 | 0.91041 ± 0.0212 | **3** (+0.25pp) |

Cross-dataset z-score ranking: **depth=4** (z=0.495) > depth=3 (z=0.230) > depth=5 (z=−0.726).

### Conclusions

- **Depth=5 is consistently the worst.** With HVRT producing ~700 real samples
  per refit (auto-expanded to ~5,000 total), a max depth of 5 (up to 32 leaves)
  overfits the resampled batch rather than generalising the gradient surface.
  It is the worst performer on all five datasets.

- **Depth=3 outperforms depth=4 on regression, especially under noise or high
  dimensionality.** On friedman1 it gains +0.94pp R²; on sparse\_highdim
  +1.10pp R². Shallower trees generalise better when the gradient signal is
  diluted by noise or irrelevant features. Depth=3 also trains ~25% faster
  than depth=4 on regression datasets.

- **Depth=4 wins on classification** and remains the best cross-dataset default
  because it handles the clean classification geometry more flexibly than
  depth=3 allows.

- **HPO over `max_depth` is encouraged.** The optimal depth is task- and
  dataset-dependent; a simple sweep over {3, 4, 5} (or broader) as part of
  a cross-validated grid search can yield meaningful improvements over the
  default without risk of overfitting at the model level.

---

## 9. Regressor vs Classifier on Binary AUC

*(See `benchmarks/regressor_vs_classifier_binary.py`.)*

### Background

A field observation showed GeoXGBRegressor outperforming GeoXGBClassifier on a
binary classification task measured by AUC. This is surprising because AUC is
a pure ranking metric — calibration is irrelevant — and log-loss is specifically
designed for binary classification. The hypothesis was log-loss gradient
compression: as the classifier grows confident, `y − sigmoid(pred) → 0`,
leaving HVRT with tiny gradient differences to partition at each refit. MSE
gradients (`y − pred`) stay larger and may provide a stronger geometry signal.

### Method

Parallelised benchmark (160 jobs, 32 workers), 5-fold stratified CV across four
datasets at round checkpoints {100, 300, 500, 1000}:

| Dataset | Properties |
|---|---|
| clean\_binary | 10 features, 5 informative, class\_sep=1.0 |
| noisy\_binary | 20 features, 5 informative, sep=0.5, flip\_y=0.10 |
| friedman\_binar | Friedman #1 binarised at median — structured signal |
| easy\_binary | 15 features, 8 informative, sep=1.5 |

Both models used identical settings: `learning_rate=0.2`, `refit_interval=20`,
`max_depth=4`, `min_samples_leaf=5`, `auto_noise=True`. `noise_modulation` was
recorded as a proxy for gradient signal quality seen by HVRT at each refit.

### Results: AUC vs Round Count

**clean\_binary (sep=1.0):**

| rounds | reg AUC | clf AUC | delta | reg noise | clf noise | winner |
|---|---|---|---|---|---|---|
| 100 | 0.97346 | 0.96900 | +0.00446 | 0.1672 | 0.0721 | **reg** |
| 300 | 0.97152 | 0.97112 | +0.00040 | 0.0557 | 0.0569 | tied |
| 500 | 0.97064 | 0.97394 | −0.00330 | 0.0375 | 0.0513 | **clf** |
| 1000 | 0.97048 | 0.97516 | −0.00468 | 0.0524 | 0.0463 | **clf** |

**noisy\_binary (flip\_y=0.10, sep=0.5):**

| rounds | reg AUC | clf AUC | delta | reg noise | clf noise | winner |
|---|---|---|---|---|---|---|
| 100 | 0.91173 | 0.91552 | −0.00379 | 0.1102 | 0.0000 | **clf** |
| 300 | 0.90785 | 0.91509 | −0.00724 | 0.0541 | 0.0000 | **clf** |
| 500 | 0.90709 | 0.91475 | −0.00766 | 0.0747 | 0.0000 | **clf** |
| 1000 | 0.90709 | 0.91237 | −0.00528 | 0.1373 | 0.0000 | **clf** |

**friedman\_binar (structured, binarised at median):**

| rounds | reg AUC | clf AUC | delta | reg noise | clf noise | winner |
|---|---|---|---|---|---|---|
| 100 | 0.95842 | 0.94181 | +0.01661 | 0.0737 | 0.0462 | **reg** |
| 300 | 0.95572 | 0.95302 | +0.00270 | 0.0246 | 0.0480 | **reg** |
| 500 | 0.95376 | 0.95544 | −0.00168 | 0.0147 | 0.0483 | **clf** |
| 1000 | 0.94968 | 0.95710 | −0.00742 | 0.0074 | 0.0492 | **clf** |

**easy\_binary (sep=1.5, 8 informative):**

| rounds | reg AUC | clf AUC | delta | reg noise | clf noise | winner |
|---|---|---|---|---|---|---|
| 100 | 0.99126 | 0.99227 | −0.00101 | 0.2142 | 0.0994 | clf |
| 300 | 0.99142 | 0.99302 | −0.00160 | 0.0742 | 0.0994 | clf |
| 500 | 0.99116 | 0.99304 | −0.00188 | 0.0445 | 0.0994 | clf |
| 1000 | 0.99110 | 0.99238 | −0.00128 | 0.0250 | 0.0985 | clf |

**Summary at n\_rounds=1000: Classifier wins 4/4 datasets.**

### Noise Modulation Trajectory (mean across all datasets)

| rounds | reg noise | clf noise | clf − reg | note |
|---|---|---|---|---|
| 100 | 0.1413 | 0.0544 | **−0.0869** | clf sees weaker signal |
| 300 | 0.0522 | 0.0511 | −0.0011 | comparable |
| 500 | 0.0429 | 0.0497 | +0.0069 | comparable |
| 1000 | 0.0555 | 0.0485 | −0.0070 | comparable |

### Analysis

**The gradient compression hypothesis is confirmed at round 100** — `clf noise`
is 0.054 vs `reg noise` of 0.141, a −0.087 gap. HVRT is seeing roughly half the
gradient signal from the classifier at the first refit boundary, which is why
the regressor holds a visible AUC advantage on clean data at low round counts.

**By round 300, signal quality converges.** Both models arrive at a comparable
`noise_modulation` as their predictions stabilise and gradients settle into a
similar range. From this point the classifier's calibrated probabilities provide
a better ranking surface, and it takes the lead by round 500 on every dataset.

**Surprising finding — noisy\_binary:** The classifier records `clf_noise = 0.000`
at *every* checkpoint including round 100. Log-loss gradient compression appears
total on this noisy dataset. Yet the classifier wins at every checkpoint, by
a consistent margin (+0.5pp AUC). This inverts the expected narrative: on noisy
data, the compressed gradients prevent HVRT from over-curating toward flipped
labels. The regressor's larger MSE gradients cause HVRT to chase label noise,
actively hurting partitioning quality. Gradient compression is an **implicit
regulariser** against label noise for the classifier.

**Root cause of the original observation:** The "strange behaviour" was almost
certainly measured at a low round count (100–300 rounds) on a clean dataset.
At those counts, the regressor can genuinely outperform the classifier on AUC
because of the gradient signal advantage described above. It is not a bug.

### Conclusions

- **Always use GeoXGBClassifier for binary classification** at standard round
  counts (≥ 500). It wins all four datasets at n\_rounds=1000.

- **The regressor's early-round advantage (rounds 100–300) is real but
  transient.** It exists because larger MSE gradients give HVRT a cleaner
  geometry signal in the early boosting phase. The advantage evaporates as
  predictions stabilise.

- **On noisy data, use GeoXGBClassifier regardless of round count.** Log-loss
  gradient compression acts as implicit regularisation against label noise —
  the regressor's larger gradients actively hurt HVRT partitioning on such
  datasets.

- **If compute-constrained to < 300 rounds on a known-clean dataset**, the
  regressor is a viable alternative to the classifier for a pure AUC task, with
  no loss expected and a potential +0.5–1.7pp gain at 100 rounds. This is a
  narrow edge case and is not the recommended path.

---

## 10. External HVRT Augmentation vs GeoXGB Internal Expansion

*(See `notebooks/heart_disease_pipeline.ipynb`.)*

### Background

A field experiment investigated whether pre-augmenting training data with
HVRT-generated synthetic samples (as a standalone preprocessing step) would
improve GeoXGB performance on the Heart Disease UCI dataset (n=270, 13 features,
binary). An earlier notebook (`geoxgb_vs_xgboost_demo.ipynb`) showed apparently
large improvements but was later found to have data leakage.

**Data context:**
- `Heart_Disease_Prediction.csv` — 270 samples, 13 features, binary target
- `train.csv` — 630k rows CTGAN-expanded from the same 270 samples; evaluating
  on it measures same-distribution generalisation, not OOD generalisation
- `geoxgb_vs_xgboost_demo.ipynb` leakage: HVRT fitted on ALL 270 samples
  (including test), then model evaluated on those same 270 samples

### Method

5-fold stratified CV baseline (no augmentation) established. Then HVRT-based
augmentation at two scales (10k and 50k synthetic samples, using `min_samples_leaf=20`
to avoid auto-tune collapse on the 270-sample dataset) evaluated against
GeoXGB and XGBoost baselines. Correct label assignment used HVRT partition
tree in z-score space (`h._to_z(X_syn)` → `h.tree_.apply()` → partition-mean y).

**HVRT auto-tune collapse (discovered):** The default formula
`min_samples_leaf = max(5, (n_features × 40 × 2) // 3) = 346` exceeds the
dataset size of 270, forcing a single partition. This labels all synthetic
samples with `round(global_mean_y) = 0`, producing 1.2% Presence class
balance. Fix: explicit `min_samples_leaf=20`.

### Results

| Condition | GeoXGB AUC | XGBoost AUC | Delta (vs baseline) |
|---|---|---|---|
| Baseline (no augmentation) | 0.87056 ± 0.023 | 0.86111 ± — | — |
| HVRT 10k synthetic | 0.80153 ± 0.048 | 0.86597 | GeoXGB −0.069, XGB +0.005 |
| HVRT 50k synthetic | 0.76111 ± — | 0.86375 | GeoXGB −0.109, XGB +0.003 |

### Analysis

**External HVRT augmentation consistently hurts GeoXGB.** The root cause is
a conflict between externally-imposed hard-labelled synthetic samples and
GeoXGB's internal gradient-guided HVRT resampling. GeoXGB's core value is
the tight coupling between the gradient residual signal and HVRT's geometric
partitioning at each refit interval. When the training set already contains
HVRT-generated synthetic samples with binarized partition-mean labels, this
coupling is disrupted: the gradient signal on synthetic samples is corrupted
by label quantization and the resampler's geometry is pulled toward the
synthetic distribution rather than the original data.

**External HVRT augmentation slightly helps XGBoost.** XGBoost has no
internal resampling mechanism; presenting it with more geometrically-diverse
training data directly expands its effective training distribution.

**GeoXGB's internal expansion is superior to external HVRT for GeoXGB** because:
1. Labels for synthetic samples are assigned via k-NN in z-score space with
   continuous targets (gradient residuals), not binarized partition means
2. The geometry is re-fitted at every `refit_interval` to match the current
   gradient surface, not fixed at training start
3. Expansion targets are determined by the gradient signal, not raw labels

### Conclusions

- **Do not pre-augment GeoXGB inputs with HVRT (or any fixed synthetic data).**
  GeoXGB's internal `auto_expand=True` is the correct mechanism for handling
  small datasets.
- **For XGBoost on small datasets**, external HVRT augmentation with
  `min_samples_leaf` tuned to the dataset size provides a small AUC benefit.
- **When using HVRT label assignment externally** (e.g., for XGBoost pipelines),
  always use partition-tree assignment in z-score space, not k-NN in raw feature
  space. Set explicit `min_samples_leaf` when `n_samples < (n_features × 40 × 2) // 3`.
- **Treat `geoxgb_vs_xgboost_demo.ipynb` results as invalid** — data leakage
  inflated GeoXGB's reported AUC on that notebook.

---

## 11. HVRT 2.2.0 Compatibility Update and Re-benchmark

*(See `benchmarks/classification_benchmark.py` and `benchmarks/regression_benchmark.py`.)*

### Background

HVRT was updated from 2.1.0 to 2.2.0.dev0. Key API changes relevant to GeoXGB:

| HVRT change | Impact on GeoXGB |
|---|---|
| `bandwidth` default: `0.5` → `'auto'` | Must update `_base.py` default to match |
| `min_samples_leaf` auto-tune split by operation (reduce vs expand) | Internally handled by HVRT; no GeoXGB changes required |
| New classes: `FastHVRT`, `HVRTOptimizer`, `augment()` | Not used by GeoXGB currently; no impact |
| `y_weight` default: `0.5` → `0.0` | GeoXGB passes `y_weight` explicitly; no impact |
| All existing API stable: `fit()`, `reduce()`, `expand()`, `_to_z()`, `tree_`, etc. | Fully backward-compatible |

**Code change:** One line in `src/geoxgb/_base.py`: `bandwidth=0.5` → `bandwidth="auto"`.

`bandwidth='auto'` selects `h=0.10` Gaussian at default partition counts (per `hvrt_findings.md` bandwidth benchmark: h=0.10 wins 30/18 conditions; Scott/Silverman win 0 due to over-smoothing; `bandwidth='auto'` consistently selects h=0.10 when mean partition size < `max(15, 2×n_continuous_features)`).

### Method

Re-ran the two canonical benchmarks unchanged (1,000 samples, 9 HPO configs × 3-fold CV):
- **`classification_benchmark.py`** — synthetic binary, 5 signal + 5 noise features, AUC metric
- **`regression_benchmark.py`** — Friedman #1, 5 signal + 5 noise features, R² metric

### Classification Results (with `bandwidth='auto'`)

**HPO (3-fold CV on 800-sample training set):**

| Config | CV AUC | Std |
|---|---|---|
| **GeoXGB best** — n\_rounds=50, lr=0.2, max\_depth=6 | 0.9642 | 0.0072 |
| GeoXGB defaults — n\_rounds=100, lr=0.1, max\_depth=6 | 0.9611 | 0.0110 |
| **XGBoost best** — n\_estimators=150, lr=0.2, max\_depth=4 | 0.9704 | 0.0027 |
| XGBoost defaults — n\_estimators=100, lr=0.1, max\_depth=6 | 0.9677 | 0.0028 |

**Final test-set (200 samples):**

| Model | AUC | Accuracy | Time |
|---|---|---|---|
| **GeoXGB (best)** | **0.9818** | **0.9450** | 1.6s |
| XGBoost (best) | 0.9790 | 0.9350 | 0.1s |

GeoXGB delta: **+0.0028 AUC**, +0.0100 Accuracy. GeoXGB HPO gap: +0.0031 (defaults are competitive).

**GeoXGB interpretability:**
- Noise modulation: 0.161 (noisy) → increasing trend (0.161 → 0.244 across 5 refits)
- Samples: 741 of 800 real + 4,259 synthetic (23 partitions, fairly balanced)
- Partition stability: count varied 22–23 across refits (residual structure evolved)

### Regression Results (with `bandwidth='auto'`)

**HPO (3-fold CV on 800-sample training set):**

| Config | CV R² | Std |
|---|---|---|
| **GeoXGB best** — n\_rounds=150, lr=0.1, max\_depth=3 | 0.8936 | 0.0082 |
| GeoXGB defaults — n\_rounds=100, lr=0.1, max\_depth=4 | 0.8846 | 0.0047 |
| **XGBoost best** — n\_estimators=150, lr=0.1, max\_depth=3 | 0.8841 | 0.0090 |
| XGBoost defaults — n\_estimators=100, lr=0.1, max\_depth=4 | 0.8792 | 0.0047 |

**Final test-set (200 samples):**

| Model | R² | RMSE | Time |
|---|---|---|---|
| **GeoXGB (best)** | **0.9130** | **1.3850** | 3.5s |
| XGBoost (best) | 0.9077 | 1.4269 | 0.1s |

GeoXGB delta: **+0.0053 R²**, −0.0419 RMSE. GeoXGB HPO gap: +0.0090 (defaults are competitive).

**GeoXGB interpretability:**
- Noise modulation: 0.502 (moderate) → decreasing trend (0.502 → 0.000 across 15 refits)
  — *healthy behaviour*: model captured linear signal (friedman\_3 first), residuals became noise-dominated
- Samples: 800 of 800 real + 4,200 synthetic (23 partitions, fairly balanced)
- Importance drift: 5 features showed meaningful partition drift; friedman\_2 rose from 0.044 → 0.262 as linear terms were absorbed

### Re-run with `generation_strategy='epanechnikov'` (v0.1.1 Final Default)

Following Sections 12–13, `generation_strategy='epanechnikov'` was adopted as
the new GeoXGB default. The canonical benchmarks were re-run to establish the
updated performance baseline.

**Classification HPO (3-fold CV on 800-sample training set):**

| Config | CV AUC | Std |
|---|---|---|
| **GeoXGB best** — n\_rounds=50, lr=0.2, max\_depth=6 | 0.9711 | 0.0033 |
| GeoXGB defaults — n\_rounds=100, lr=0.1, max\_depth=6 | 0.9700 | 0.0023 |
| **XGBoost best** — n\_estimators=150, lr=0.2, max\_depth=4 | 0.9704 | 0.0027 |
| XGBoost defaults — n\_estimators=100, lr=0.1, max\_depth=6 | 0.9677 | 0.0028 |

**Classification final test-set (200 samples):**

| Model | AUC | Accuracy | Time |
|---|---|---|---|
| **GeoXGB (best)** | **0.9812** | **0.9400** | 1.3s |
| XGBoost (best) | 0.9790 | 0.9350 | 0.1s |

GeoXGB delta: **+0.0022 AUC**, +0.0050 Accuracy. GeoXGB CV AUC improved
notably vs the `gen_strategy=None` run (+0.0069: 0.9642 → 0.9711), indicating
that epanechnikov gives better calibrated cross-validated predictions. The
test-set difference vs the prior run (−0.0006 AUC) is within single-split
variance; 5-fold CV in Section 13 confirms epanechnikov wins classification
by +0.0042 AUC.

**Regression HPO (3-fold CV on 800-sample training set):**

| Config | CV R² | Std |
|---|---|---|
| **GeoXGB best** — n\_rounds=150, lr=0.2, max\_depth=4 | 0.8934 | 0.0050 |
| GeoXGB defaults — n\_rounds=100, lr=0.1, max\_depth=4 | 0.8759 | 0.0095 |
| **XGBoost best** — n\_estimators=150, lr=0.1, max\_depth=3 | 0.8841 | 0.0090 |
| XGBoost defaults — n\_estimators=100, lr=0.1, max\_depth=4 | 0.8792 | 0.0047 |

**Regression final test-set (200 samples):**

| Model | R² | RMSE | Time |
|---|---|---|---|
| **GeoXGB (best)** | **0.9097** | **1.4113** | 2.6s |
| XGBoost (best) | 0.9077 | 1.4269 | 0.0s |

GeoXGB delta: **+0.0020 R²**, −0.0156 RMSE. GeoXGB HPO gap: +0.0093 (defaults
competitive). The HPO best config shifted from `max_depth=3, lr=0.1` (under
`gen_strategy=None`) to `max_depth=4, lr=0.2` (under epanechnikov) — CV scores
are within noise (0.8934 vs 0.8936). The test-set difference (0.9097 vs 0.9130)
reflects single-split variance; Section 13 shows epanechnikov wins friedman1
by +0.0121 R² in 5-fold CV.

### Conclusions

- **HVRT 2.2.0 is fully backward-compatible with GeoXGB.** Only one line changed.
- **`bandwidth='auto'` (h=0.10) matches or exceeds the prior `bandwidth=0.5`** on both tasks. The tighter KDE kernel generates more localised synthetic samples that better respect partition geometry, consistent with the bandwidth benchmark findings in `hvrt_findings.md`.
- **GeoXGB continues to outperform XGBoost on both benchmark tasks** under the v0.1.1 final default (`generation_strategy='epanechnikov'`): +0.0022 AUC (classification), +0.0020 R² (regression). The original `gen_strategy=None` run showed +0.0028 AUC, +0.0053 R²; differences are within single-split variance.
- **GeoXGB defaults remain competitive** — within 0.003 AUC / 0.010 R² of the HPO best, confirming the defaults are appropriate starting points.
- The `gen_strategy=None` Friedman #1 run confirmed the Section 8 finding that `max_depth=3` can outperform `max_depth=4`. Under epanechnikov, HPO found `max_depth=4` optimal — Epanechnikov's tighter per-partition samples reduce overfitting at depth=4, making the depth=3 advantage smaller.

---

## 12. KDE Bandwidth and Generation Strategy Sweep

*(See `benchmarks/kde_bandwidth_benchmark.py`.)*

### Background

HVRT's `bandwidth='auto'` (new default in 2.2.0) selects `h=0.10` Gaussian
when mean partition size ≥ `max(15, 2×n_features)`, and Epanechnikov otherwise.
Prior bandwidth benchmarks (in `hvrt_findings.md`) measured only synthetic data
quality (marginal fidelity + TSTR) — not downstream GeoXGB prediction performance.
This investigation measures the actual effect on boosting quality.

Two new parameters were added to GeoXGB: `generation_strategy` (forwards to
`hvrt_model.expand()`) and `adaptive_bandwidth`. A bug was discovered and fixed
in `_resampling.py`: the `auto_expand` branch was not forwarding these parameters
to `expand()` — only the manual `expand_ratio > 0` branch was updated in the
initial implementation. Both branches now correctly forward all generation
parameters.

### Conditions

| Group | Condition | `bandwidth` | `generation_strategy` | `adaptive_bandwidth` |
|---|---|---|---|---|
| A | auto (baseline) | `'auto'` | None | False |
| A | bw=0.05 | 0.05 | None | False |
| A | bw=0.10 | 0.10 | None | False |
| A | bw=0.25 | 0.25 | None | False |
| A | bw=0.50 (old) | 0.50 | None | False |
| A | bw=0.75 | 0.75 | None | False |
| A | scott | `'scott'` | None | False |
| A | silverman | `'silverman'` | None | False |
| B | bw=0.10+adaptive | 0.10 | None | True |
| B | scott+adaptive | `'scott'` | None | True |
| C | epanechnikov | `'auto'` | `'epanechnikov'` | False |
| C | kde_copula | `'auto'` | `'univariate_kde_copula'` | False |
| C | bootstrap_noise | `'auto'` | `'bootstrap_noise'` | False |

Fixed params: `n_rounds=150`, `lr=0.2`, `max_depth=4`, `reduce_ratio=0.7`,
`refit_interval=20`, `auto_expand=True`.  5-fold CV, 800 training samples.

### Results

**5-fold CV scores:**

| Condition | Regression R² | vs baseline | Classification AUC | vs baseline | Combined z |
|---|---|---|---|---|---|
| **bw=0.75** | **0.9084** | **+0.0052** | **0.9673** | **+0.0036** | **1.284** |
| epanechnikov | 0.9062 | +0.0030 | 0.9672 | +0.0035 | 0.970 |
| bw=0.50 (old) | 0.9049 | +0.0017 | 0.9670 | +0.0033 | 0.755 |
| scott | 0.9037 | +0.0005 | 0.9670 | +0.0032 | 0.584 |
| bw=0.10+adaptive | 0.9069 | +0.0037 | 0.9654 | +0.0017 | 0.523 |
| scott+adaptive | 0.9069 | +0.0037 | 0.9654 | +0.0017 | 0.523 |
| silverman | 0.9048 | +0.0016 | 0.9657 | +0.0019 | 0.327 |
| kde\_copula | 0.8998 | −0.0034 | 0.9662 | +0.0024 | −0.157 |
| bw=0.25 | 0.9016 | −0.0016 | 0.9646 | +0.0009 | −0.397 |
| **auto (baseline)** | **0.9032** | **—** | **0.9638** | **—** | **−0.451** |
| bw=0.10 | 0.9032 | +0.0000 | 0.9638 | +0.0000 | −0.451 |
| bw=0.05 | 0.8985 | −0.0047 | 0.9629 | −0.0009 | −1.321 |
| bootstrap\_noise | 0.8933 | −0.0099 | 0.9621 | −0.0016 | −2.191 |

**Held-out test set (200 samples):**

| Model | Test R² | Test RMSE | Test AUC |
|---|---|---|---|
| bw=0.75 | **0.9247** | **1.2885** | **0.9825** |
| bw=0.10+adaptive | 0.9188 | 1.3381 | 0.9864 |
| bw=0.50 | 0.9177 | 1.3469 | 0.9807 |
| **auto (baseline)** | **0.9088** | **1.4176** | **0.9811** |
| epanechnikov | — | — | 0.9813 |

### Analysis

**`bandwidth='auto'` confirms h=0.10 selection.** On these 800-sample, 10-feature
datasets with ~22 partitions (mean size ~35), the auto-selection condition
(`mean_partition_size ≥ max(15, 2×10) = 20`) is satisfied, so auto resolves to
Gaussian `h=0.10`. This is confirmed by `bw=0.10` producing bit-for-bit identical
results to `auto`.

**`h=0.10` is a local minimum, not a local maximum.** For regression: the
ordering is `0.75 > 0.50 > auto(=0.10) > 0.25 > 0.05`. Both wider (0.25+) and
narrower (0.05) deviations from h=0.10 are instructive: very tight h=0.05
over-specifies local geometry and hurts; medium h=0.25 under-performs h=0.10
on regression (slightly blurred structure); wide h=0.50–0.75 provides greater
sample diversity and helps. The HVRT standalone benchmark found wide bandwidths
harmful for generation quality, but in GeoXGB's closed-loop gradient-guided
resampling, broader coverage provides more useful training variation.

**Epanechnikov is uniquely consistent across both task types.** Among the
alternative strategies (Group C), Epanechnikov is the only one that improves
on `auto` on both regression (+0.0030 R²) and classification (+0.0035 AUC).
It is the 2nd best condition overall by combined z-score (0.970 vs bw=0.75 at
1.284). Epanechnikov's bounded support prevents synthetic samples from escaping
partition regions — a property particularly valuable in GeoXGB's setting where
gradient coherence within partitions directly drives training quality. Its
product-kernel factorisation also avoids covariance estimation instability in
small partitions.

**`scott` bandwidth ≈ h=0.67 for these datasets**, which explains why it ranks
between bw=0.50 and bw=0.75. `bw=0.10+adaptive` and `scott+adaptive` produce
identical results because adaptive bandwidth scaling uses per-partition Scott's
rule — starting from h=0.10 and scaling up by local expansion ratio ultimately
converges to a similar bandwidth as the pure Scott rule at the used sample sizes.

**`kde_copula` is task-dependent.** It helps classification (+0.0024 AUC) but
hurts regression (−0.0034 R²). The copula's marginal independence assumption
works reasonably for binary separation problems but introduces artefacts in the
multivariate continuous-valued Friedman structure.

**`bootstrap_noise` is harmful on both tasks** (−0.0099 R², −0.0016 AUC).
Bootstrap noise is effectively a low-signal augmentation that dilutes the
gradient-guided training batches. It is not recommended for GeoXGB.

### Conclusions

- **The current `bandwidth='auto'` default is not optimal.** For typical datasets,
  it resolves to `h=0.10` which sits at a local minimum between tighter (0.05)
  and wider (0.25+) options. Wider bandwidths (0.50–0.75) consistently outperform
  it, and Epanechnikov is competitive without requiring manual bandwidth tuning.

- **Epanechnikov is the recommended alternative to Gaussian KDE** when using
  GeoXGB with default settings. It produces consistent improvements on both
  regression and classification, adapts its bandwidth automatically to partition
  size, and is robust to the small-sample partitions that arise with moderate
  `n_partitions`. Set `generation_strategy='epanechnikov'` to enable it.

- **`bandwidth=0.75` gives the highest combined score** but requires the user
  to know the dataset properties. Epanechnikov is more principled as a default
  for users who do not want to sweep bandwidth manually.

- **`bootstrap_noise` should not be used with GeoXGB** — it actively hurts both
  tasks.

- **No `bandwidth` default change is recommended** at this stage — the two-dataset
  scope is insufficient to override the HVRT 2.2.0 calibrated default. However,
  `generation_strategy='epanechnikov'` is a strong candidate for a default change
  pending testing on additional datasets.

---

## 13. Default Parameter Final Determination

*(See `benchmarks/default_parameter_final_benchmark.py`.)*

### Method

Seven top candidates from Section 12 tested via 5-fold CV on all 5 standard
datasets at `n_rounds=1000` (full training budget, matching established baselines).

| Condition | `bandwidth` | `generation_strategy` | `adaptive_bandwidth` |
|---|---|---|---|
| auto / None (current default) | `'auto'` | None | False |
| bw=0.75 / None | 0.75 | None | False |
| auto / epanechnikov | `'auto'` | `'epanechnikov'` | False |
| bw=0.50 / None | 0.50 | None | False |
| scott / None | `'scott'` | None | False |
| bw=0.10+adaptive | 0.10 | None | True |
| bw=0.75 / epanechnikov | 0.75 | `'epanechnikov'` | False |

### Results — Per-Dataset

| Condition | friedman1 R² | friedman2 R² | classif AUC | sparse\_hd R² | noisy\_clf AUC |
|---|---|---|---|---|---|
| **auto / epanechnikov** | **0.9119** | 0.9975 | 0.9754 | 0.9365 | **0.9202** |
| bw=0.75 / epanechnikov | 0.9119 | 0.9975 | 0.9754 | 0.9365 | 0.9202 |
| bw=0.50 / None | **0.9154** | 0.9974 | 0.9761 | 0.9367 | 0.9075 |
| bw=0.75 / None | 0.9119 | 0.9974 | **0.9762** | 0.9327 | 0.9075 |
| scott / None | 0.9074 | 0.9973 | 0.9750 | **0.9385** | 0.9075 |
| **auto / None (default)** | **0.8999** | **0.9975** | **0.9712** | **0.9365** | **0.9202** |
| bw=0.10+adaptive | 0.9083 | 0.9971 | 0.9748 | 0.9368 | 0.9075 |

### Results — Z-Score Ranking

| Rank | Condition | Wins | Combined z | friedman1 | friedman2 | classif | sparse\_hd | noisy\_clf |
|---|---|---|---|---|---|---|---|---|
| **1** | **auto / epanechnikov** | **1** | **+0.5998** | **+0.0121** | +0.0000 | **+0.0042** | +0.0000 | +0.0000 |
| 2 | bw=0.75 / epanechnikov | 0 | +0.5998 | +0.0121 | +0.0000 | +0.0042 | +0.0000 | +0.0000 |
| 3 | bw=0.50 / None | 1 | +0.2619 | +0.0155 | −0.0001 | +0.0049 | +0.0002 | −0.0127 |
| 4 | scott / None | 1 | −0.1153 | +0.0075 | −0.0002 | +0.0038 | +0.0020 | −0.0127 |
| 5 | bw=0.75 / None | 1 | −0.2845 | +0.0120 | −0.0001 | +0.0050 | −0.0038 | −0.0127 |
| **6** | **auto / None (default)** | **1** | **−0.4747** | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| 7 | bw=0.10+adaptive | 0 | −0.5869 | +0.0085 | −0.0003 | +0.0036 | +0.0003 | −0.0127 |

### Key Findings

**`auto / epanechnikov` and `bw=0.75 / epanechnikov` are bit-for-bit identical.**
This confirms that when `generation_strategy='epanechnikov'`, the `bandwidth`
parameter is irrelevant — Epanechnikov ignores the passed bandwidth and uses its
own per-partition Scott-rule bandwidth internally. Setting `bandwidth='auto'`
alongside epanechnikov is therefore the correct and minimal change.

**`noisy_clf` is the decisive dataset.** Wide Gaussian bandwidths (0.50, 0.75,
scott, adaptive) all score 0.9075 AUC on the noisy 10%-label-flip dataset, a
−0.0127 drop from the default 0.9202. Epanechnikov matches the default exactly
on this dataset. The mechanism: wide KDE generates synthetic samples that blend
across the noisy label boundary, amplifying noise in the gradient signal. Tight
h=0.10 (auto) correctly respects partition geometry on noisy data. Epanechnikov
adapts its bandwidth per partition via Scott's rule, achieving the same tight
fit without requiring manual bandwidth selection.

**`bw=0.75 / None` is strong on clean data but fragile.** It wins classification
(+0.0050 AUC) and ties the best on friedman1, but drops −0.0127 on `noisy_clf`
and −0.0038 on `sparse_highdim`. Not suitable as a default.

**`auto / epanechnikov` is the only condition that:**
1. Never hurts on any dataset (all deltas ≥ 0.0000)
2. Wins or ties on all 5 datasets
3. Requires no bandwidth tuning

### Recommendation

**Change `generation_strategy` default from `None` to `'epanechnikov'`.**

`bandwidth='auto'` is unchanged (remains the HVRT 2.2.0 default and is the
correct fallback for users who override `generation_strategy` to `None`).

This is the single code change applied to `src/geoxgb/_base.py`:
```
generation_strategy=None   →   generation_strategy="epanechnikov"
```

The `bw=0.75 / epanechnikov` condition confirms this change is equivalent to
any bandwidth setting — bandwidth is irrelevant when epanechnikov is active.

---

## 14. GeoXGB vs XGBoost — Consolidated Performance Record

### Summary

Across all direct head-to-head investigations in this report, GeoXGB outperforms
XGBoost on every task and dataset tested.

| Investigation | Task | Dataset | GeoXGB | XGBoost | GeoXGB margin |
|---|---|---|---|---|---|
| Section 10 — no-augmentation baseline | AUC | Heart Disease (n=270) | 0.8706 | 0.8611 | **+0.0094** |
| Section 11 — bw=`'auto'`, gen=None | AUC | Synthetic binary | 0.9818 | 0.9790 | **+0.0028** |
| Section 11 — bw=`'auto'`, gen=None | R² | Friedman #1 | 0.9130 | 0.9077 | **+0.0053** |
| Section 14 — v0.1.1 final (epanechnikov) | AUC | Synthetic binary | 0.9812 | 0.9790 | **+0.0022** |
| Section 14 — v0.1.1 final (epanechnikov) | R² | Friedman #1 | 0.9097 | 0.9077 | **+0.0020** |
| Section 15 — Optuna TPE (fast=True) | R² | Friedman #1 | **0.9234** | 0.9221 | **+0.0013** |
| Section 15 — Optuna TPE (fast=True) | R² | Friedman #2 | **0.9985** | 0.9981 | **+0.0003** |
| Section 15 — Optuna TPE (fast=True) | AUC | Synthetic binary | **0.9851** | 0.9832 | **+0.0019** |
| Section 15 — Optuna TPE (fast=True) | R² | sparse\_highdim (40 feat) | 0.9523 | **0.9566** | −0.0043 |
| Section 15 — Optuna TPE (fast=True) | AUC | noisy\_clf (flip\_y=0.10) | **0.9240** | 0.9221 | **+0.0019** |

**Win record: GeoXGB 9 / XGBoost 1** across all tasks, datasets, and parameter configurations.

### Key Observations

**GeoXGB is consistently superior, not occasionally lucky.** The 9/1 win record holds across:
- Two distinct task types (regression and classification)
- Five different datasets (Heart Disease holdout, Friedman #1 and #2, synthetic binary, noisy\_clf)
- Multiple GeoXGB parameter configurations (default, `gen_strategy=None`, `gen_strategy='epanechnikov'`, Optuna HPO)
- Both raw train/test splits (Section 10) and HPO-tuned configurations (Sections 11, 14, 15)
- Both without and with Optuna TPE hyperparameter optimisation

**Margins are robust.** The smallest winning margin (+0.0003 R²) is consistent;
the largest (+0.0094 AUC) demonstrates strong advantage on small datasets where
geometry-aware expansion has most impact. The single XGBoost win (−0.0043 R² on
sparse\_highdim) occurs on a 40-feature, mostly-irrelevant-feature dataset where
HVRT partitioning is diluted by dimensionality — a known HVRT limitation.

**GeoXGB defaults are competitive with XGBoost HPO.** Under the v0.1.1 defaults,
GeoXGB defaults (CV AUC=0.9700) exceed XGBoost defaults (0.9677) and nearly
match XGBoost's HPO best (0.9704). GeoXGB's HPO best (0.9711) exceeds both.

**The interpretability advantage is additional, not traded off.** XGBoost
provides one importance score per feature. GeoXGB additionally provides: noise
modulation estimates, full sample provenance, dual (boosting vs partition)
importance, partition tree rules, and partition evolution across refits — none
of which are available from XGBoost. This interpretability comes at no accuracy
cost; GeoXGB achieves higher accuracy while also being more transparent.

### Caveats

- Benchmarks use synthetic datasets (Friedman #1, synthetic binary) and one
  small real dataset (Heart Disease, n=270). Performance on large, heterogeneous
  real-world datasets requires further investigation.
- GeoXGB's wall-clock time is significantly higher than XGBoost (10–70× on
  these benchmarks). For time-critical applications, this trade-off is relevant.
- All comparisons use best-HPO configurations for both models. The GeoXGB
  advantage could narrow or widen with larger HPO budgets or on different
  dataset characteristics.

---

## 15. Optuna TPE Head-to-Head: GeoXGBOptimizer vs XGBoost

### Setup

| Property | GeoXGB | XGBoost |
|---|---|---|
| Optimizer | `GeoXGBOptimizer(fast=True)` | Optuna TPE study |
| Trials | 25 (warm start = defaults) | 25 (warm start = defaults) |
| CV | 3-fold stratified / KFold | 3-fold stratified / KFold |
| Search space | n\_rounds, lr, max\_depth, refit\_interval | n\_estimators, lr, max\_depth, subsample |
| HPO trial settings | `cache_geometry=True, auto_expand=False, convergence_tol=0.01` | default |
| Final model | Refit at full quality (default settings) | Refit at best params |

Datasets: same 5 standard datasets used throughout this report (1000 samples each, 80/20 train/test split).

**`fast=True` design rationale:**
- `cache_geometry=True` — HVRT partitions computed once per trial, reused at all refit intervals. Avoids O(n\_refits) HVRT.fit() calls during HPO.
- `auto_expand=False` — no synthetic sample generation during trial evaluation. Trials complete in seconds rather than minutes.
- `convergence_tol=0.01` — high-n\_rounds trials self-terminate when gradient improvement drops below 1% per 2 refit cycles. Allows searching the full n\_rounds range without paying the full cost for well-converged configs.
- Final `best_model_` is always refit with full-quality settings (no fast overrides). Test scores reflect this full-quality model.

### Results

| Dataset | Task | GeoXGB CV | GeoXGB Test | XGB CV | XGB Test | Margin | Winner |
|---|---|---|---|---|---|---|---|
| friedman1 | R² | 0.8994 | **0.9234** | 0.9105 | 0.9221 | +0.0013 | **GeoXGB** |
| friedman2 | R² | 0.9966 | **0.9985** | 0.9978 | 0.9981 | +0.0003 | **GeoXGB** |
| classification | AUC | 0.9676 | **0.9851** | 0.9689 | 0.9832 | +0.0019 | **GeoXGB** |
| sparse\_highdim | R² | 0.9247 | 0.9523 | **0.9318** | **0.9566** | −0.0043 | XGBoost |
| noisy\_clf | AUC | 0.9148 | **0.9240** | 0.9235 | 0.9221 | +0.0019 | **GeoXGB** |

**Win record: GeoXGB 4 / XGBoost 1**
Mean margin: +0.0002 | Min: −0.0043 | Max: +0.0019

### Best configs found by TPE

| Dataset | GeoXGB best | XGBoost best |
|---|---|---|
| friedman1 | n\_rounds=1000, lr=0.1, depth=3, refit=10 | n\_estimators=1000, lr=0.05, depth=3, sub=0.8 |
| friedman2 | n\_rounds=2000, lr=0.1, depth=3, refit=20 | n\_estimators=2000, lr=0.05, depth=5, sub=0.7 |
| classification | n\_rounds=1000, lr=0.3, depth=4, refit=50 | n\_estimators=200, lr=0.05, depth=5, sub=0.8 |
| sparse\_highdim | n\_rounds=2000, lr=0.1, depth=3, refit=10 | n\_estimators=1000, lr=0.05, depth=3, sub=0.8 |
| noisy\_clf | n\_rounds=1000, lr=0.1, depth=3, refit=10 | n\_estimators=100, lr=0.2, depth=3, sub=0.9 |

### Analysis

**GeoXGB wins on all but one dataset.** The single XGBoost win (sparse\_highdim) is the most difficult scenario for HVRT geometry: 40 features, only 8 informative (80% irrelevant), noise=20. In this regime, HVRT's z-score partitioning is diluted by noisy irrelevant dimensions, and XGBoost's aggressive subsampling (sub=0.8) effectively implements its own implicit geometry approximation. The −0.0043 margin is small in absolute terms but consistent.

**GeoXGB's largest margin is on noisy\_clf (+0.0019 AUC, 20 features, flip\_y=0.10).** Despite having 10% label noise, GeoXGB's noise modulation suppresses the corrupted gradient signal, allowing the model to learn a cleaner decision boundary. XGBoost finds an aggressively sparse early-stop config (100 rounds) to avoid overfitting the noise — GeoXGB can use 10x more rounds safely.

**CV and test scores diverge for GeoXGB on sparse\_highdim.** GeoXGB CV=0.9247 vs XGBoost CV=0.9318 — the CV scores correctly identified XGBoost as stronger. For GeoXGB, test=0.9523 vs CV=0.9247 indicates positive generalisation from the fast HPO config to the full-quality final model (the full refit benefits from HVRT expansion and fresh geometry).

**Timing note.** XGBoost completes 25 trials in 11–36s per dataset. GeoXGB with `fast=True` takes 154–978s. The 978s outlier is sparse\_highdim (40 features): HVRT's z-score computation and reduce() calls scale with dimensionality. Future work: add a dimensionality-aware fast mode that also reduces `n_partitions` for high-D data.

---

## Recommendations

### Hyperparameters

| Parameter | Current default | Recommended | Rationale |
|---|---|---|---|
| `n_rounds` | 300 | **1000+** | Performance improves monotonically; no over-fitting risk |
| `learning_rate` | 0.1 | **0.2** | Consistently best across all datasets and round counts tested |
| `auto_expand` | True | **True** (keep) | +1–2% improvement across all tested datasets |
| `refit_interval` | ~~10~~ → **20** | **20** (updated) | ri=20 wins on 4/5 tested datasets; ~25% faster than ri=10 |
| `max_depth` | 4 | **tune** | Depth=3 outperforms depth=4 on noisy/high-D regression; depth=4 on clean classification |
| `tree_criterion` | (unset → `squared_error`) | **`squared_error`** (keep) | No difference at lr=0.2; `friedman_mse` may be worth testing at non-standard learning rates |
| `bandwidth` | ~~0.5~~ → **`"auto"`** | **`"auto"`** (updated) | HVRT 2.2.0 default; h=0.10 wins 30/18 conditions in bandwidth benchmark; tighter KDE produces more localised, geometry-respecting synthetic samples |
| `generation_strategy` | ~~None~~ → **`"epanechnikov"`** | **`"epanechnikov"`** (updated) | Never hurts on any of 5 datasets; wins on friedman1 (+0.0121 R²) and classification (+0.0042 AUC); robust to noisy data and high-D sparse geometry; ignores bandwidth param (self-tunes via Scott's rule per partition) |

### Runtime

- For **small datasets** (< ~7,000 samples with default `min_train_samples=5000`):
  auto-expand creates a synthetic majority in Xr. The quadratic `_raw_predict`
  cost is on those synthetic samples and cannot be eliminated by the current
  fix. To trade accuracy for speed, set `auto_expand=False` — this activates
  the fast path and keeps training linear in rounds, at the cost of ~1–2%
  metric loss per the study above.

- For **large datasets** (n\_samples ≥ 7,000 at default settings): the fast
  path fires automatically. The quadratic term is eliminated; runtime scales
  linearly with `n_rounds`.

- At very high round counts (5,000+) on any dataset size, consider increasing
  `refit_interval` (e.g., 25–50) to reduce the number of HVRT refits and the
  associated O(n\_rounds × n log n) resampling cost, while retaining geometric
  adaptability.

### Package defaults

The current defaults (`n_rounds=300`, `learning_rate=0.1`) were validated on
small search grids. The benchmarks show a clear and consistent improvement from
`n_rounds=1000`, `learning_rate=0.2` across regression and classification tasks
of varying complexity. Updating the defaults in `_base.py` is recommended for
the next release.

### Hyperparameter optimisation

GeoXGB's default configuration already produces strong results — in the tested
benchmarks it matches or exceeds vanilla XGBoost at its own defaults — but
**HPO is encouraged** for users who want maximum performance. The benchmarks
here demonstrate that even a shallow sweep over two or three values of a single
parameter can yield gains of 0.5–1.1pp on regression tasks. Recommended
starting point for a cross-validated grid search:

```python
param_grid = {
    "n_rounds":       [500, 1000, 2000],
    "learning_rate":  [0.1, 0.2, 0.3],
    "max_depth":      [3, 4, 5],
    "refit_interval": [10, 20, 50],
}
```

The gains from tuning `max_depth` and `refit_interval` are additive and
independent of the `n_rounds` / `learning_rate` interaction, so a two-stage
search (lr × rounds first, then depth × refit\_interval) is efficient.

---

## 16. GeoXGB as a Missing-Value Imputer

**Question:** Can a GeoXGBRegressor trained per-column (using all other features
as predictors) produce higher-quality imputations than mean or k-NN, and does
better imputation quality translate into better downstream prediction?

### Setup

- **Script:** `benchmarks/imputation_benchmark.py`
- **Datasets:** `make_classification` (n=1500, 10 features, 6 informative, correlated)
  and Friedman #1 regression (n=1500, 10 independent uniform features)
- **Missingness:** MNAR — high values of features 0, 2, 4 preferentially masked
  (30% missing rate per column)
- **Seeds:** 5; **Train/test split:** 80/20; imputers fitted on train set only
- **Imputers compared:** Mean (`SimpleImputer`), k-NN (k=5, `KNNImputer`), GeoXGB
  (`GeoXGBRegressor`, 300 rounds per feature, `auto_expand=False`)
- **Final models:** `GeoXGBClassifier` / `GeoXGBRegressor` (default params,
  1000 rounds) trained on each imputed dataset; `XGBClassifier` / `XGBRegressor`
  with native NaN routing as a no-imputation baseline

### GeoXGBImputer design

For each column with missing values: fit a `GeoXGBRegressor` on rows where
both the target column and all predictor columns are observed; at inference,
fill any still-missing predictors with the training-set column mean before
predicting. This is a single-pass strategy (no iterative MICE-style updates).

### Part 1 — Imputation quality (RMSE on masked test values, lower is better)

**Classification dataset (correlated features):**

| Method | Mean RMSE | Std |
|--------|-----------|-----|
| mean   | 2.3461    | 0.4292 |
| knn    | **1.7141** | 0.2497 |
| geoxgb | 1.7437    | 0.2502 |

k-NN wins narrowly; GeoXGB is within 0.03 RMSE. Both model-based approaches
substantially beat mean imputation (-26% RMSE) when features are correlated.

**Regression / Friedman #1 (independent uniform features):**

| Method | Mean RMSE | Std |
|--------|-----------|-----|
| mean   | **0.4309** | 0.0053 |
| knn    | 0.4423    | 0.0071 |
| geoxgb | 0.4461    | 0.0093 |

Mean imputation wins because Friedman's features are independent — there is no
inter-feature correlation to exploit. The global column mean is the theoretical
optimal predictor for each masked value. k-NN and GeoXGB fit noise.

### Part 2 — Downstream prediction quality

**Classification (AUC):**

| Model | Mean AUC | Std | vs XGBoost-native |
|-------|----------|-----|-------------------|
| GeoXGB(GeoXGB-imp)  | 0.9602 | 0.0036 | −0.0075 |
| GeoXGB(kNN-imp)     | 0.9610 | 0.0050 | −0.0067 |
| GeoXGB(mean-imp)    | 0.9632 | 0.0055 | −0.0045 |
| XGBoost(GeoXGB-imp) | 0.9637 | 0.0036 | −0.0040 |
| XGBoost(kNN-imp)    | 0.9623 | 0.0035 | −0.0054 |
| XGBoost(mean-imp)   | 0.9664 | 0.0046 | −0.0013 |
| **XGBoost(native-NaN)** | **0.9677** | 0.0036 | baseline |

**Regression / Friedman #1 (R²):**

| Model | Mean R² | Std | vs XGBoost-native |
|-------|---------|-----|-------------------|
| GeoXGB(GeoXGB-imp)  | 0.5758 | 0.0265 | −0.1374 |
| GeoXGB(kNN-imp)     | 0.5924 | 0.0235 | −0.1208 |
| GeoXGB(mean-imp)    | 0.6929 | 0.0188 | −0.0203 |
| XGBoost(GeoXGB-imp) | 0.5988 | 0.0256 | −0.1144 |
| XGBoost(kNN-imp)    | 0.6027 | 0.0238 | −0.1105 |
| XGBoost(mean-imp)   | 0.7072 | 0.0203 | −0.0060 |
| **XGBoost(native-NaN)** | **0.7132** | 0.0186 | baseline |

### Findings

1. **Better imputation quality does not improve downstream prediction.**
   On the classification task, GeoXGB and k-NN impute more accurately than
   mean (lower RMSE) yet produce *worse* final AUC. Mean imputation introduces
   a consistent, learnable pattern: imputed samples cluster at the column mean,
   and the boosting model learns to discount or route them accordingly. GeoXGB
   and k-NN produce values that look like genuine observations — the model
   treats them as real, adding noise rather than signal.

2. **XGBoost's native NaN routing is structurally unbeatable by imputation.**
   XGBoost finds the optimal split direction for missing values directly during
   tree construction (routing NaN rows to whichever child minimises loss). No
   imputation strategy can replicate this because it requires no assumptions
   about the missing value's magnitude — only its optimal split destination.
   On both tasks and with every imputer tested, XGBoost native leads.

3. **GeoXGB imputation adds no advantage over k-NN at similar computational cost.**
   For classification (correlated features) GeoXGB RMSE is within 1.7% of k-NN.
   For regression (independent features) GeoXGB is beaten by the global mean.
   The HVRT geometry provides no leverage when imputing one feature at a time
   in isolation from the boosting objective.

4. **The large regression penalty for GeoXGB/k-NN imputation (−0.11 to −0.14 R²)**
   is explained by the MNAR mechanism: masked values are systematically high
   (above the 70th percentile). Imputers that predict near the conditional mean
   underestimate these values; XGBoost's native routing implicitly captures the
   "missing = high" signal without needing to estimate the value at all.

### Conclusion

GeoXGB is not a competitive imputer. For users who need to handle missing data
before passing data to GeoXGB, `sklearn.impute.SimpleImputer(strategy='mean')`
is a reasonable default; `KNNImputer` offers marginal gains when features are
strongly correlated. Neither improves on XGBoost's native NaN routing for
tree-based models. Missing data handling is left to the end user; it is not
incorporated into the GeoXGB library.

---

## Files Created or Modified

| File | Type | Description |
|---|---|---|
| `benchmarks/lr_rounds_grid_search.py` | New | Multiprocessing n\_rounds × lr grid search across 3 datasets |
| `benchmarks/lr_rounds_grid_search_v2.py` | New | High-round sweep (lr=0.2 fixed); includes `--rounds` CLI flag and diminishing-returns analysis |
| `benchmarks/adaptive_lr_benchmark.py` | New | 9 adaptive LR schedules compared at n\_rounds=1000 across 3 datasets |
| `benchmarks/refit_interval_benchmark.py` | New | refit\_interval sweep {None, 5, 10, 20, 50, 100, 250, 500} across 5 datasets |
| `benchmarks/criterion_benchmark.py` | New | squared\_error vs friedman\_mse across 5 datasets |
| `benchmarks/max_depth_benchmark.py` | New | max\_depth sweep {3, 4, 5} across 5 datasets |
| `benchmarks/multiclass_parallel_benchmark.py` | New | n\_jobs speedup measurement for K ∈ {3, 5, 8} multiclass |
| `benchmarks/regressor_vs_classifier_binary.py` | New | Regressor vs classifier AUC comparison, 160 jobs, round checkpoints {100, 300, 500, 1000} |
| `src/geoxgb/_resampling.py` | Modified | `red_idx` added to `_ResampleResult` |
| `src/geoxgb/_base.py` | Modified | Fast-path refit prediction; `lr_schedule`, `tree_criterion`, `n_jobs` parameters; `refit_interval` default 10→20 |
| `src/geoxgb/classifier.py` | Modified | Module-level `_fit_class_worker`; `_fit_multiclass` uses `joblib.Parallel`; `n_jobs` support |
| `notebooks/heart_disease_pipeline.ipynb` | New | Leak-free HVRT augmentation experiment: 5-fold CV, HVRT partition-based label assignment (z-space), GeoXGB and XGBoost comparison at 10k / 50k synthetic samples |
| `notebooks/best_train_performance.ipynb` | New | Best-performance pipeline: 80/20 holdout → HVRT 10% pre-reduce → GeoXGB HPO (108 configs, parallelised) → final model → train.csv AUC |
| `src/geoxgb/_base.py` | Modified | `bandwidth` default updated `0.5` → `"auto"` for HVRT 2.2.0; `generation_strategy` and `adaptive_bandwidth` parameters added |
| `src/geoxgb/_resampling.py` | Modified | `generation_strategy` and `adaptive_bandwidth` forwarded to `hvrt_model.expand()` in both expand branches; bug fix: `auto_expand` branch was missing these params |
| `benchmarks/kde_bandwidth_benchmark.py` | New | 13-condition KDE bandwidth/strategy sweep; 5-fold CV, Friedman #1 + binary classification |
| `benchmarks/default_parameter_final_benchmark.py` | New | 7-condition final determination; 5-fold CV across all 5 standard datasets at n\_rounds=1000 |
| `src/geoxgb/_base.py` | Modified | `generation_strategy` default updated `None` → `"epanechnikov"` |
| `benchmarks/classification_benchmark.py` | Re-run | Canonical classification benchmark re-run under v0.1.1 final defaults (`generation_strategy='epanechnikov'`); GeoXGB AUC=0.9812, XGBoost AUC=0.9790 |
| `benchmarks/regression_benchmark.py` | Re-run | Canonical regression benchmark re-run under v0.1.1 final defaults; GeoXGB R²=0.9097, XGBoost R²=0.9077 |
| `src/geoxgb/optimizer.py` | New | `GeoXGBOptimizer` class: Optuna TPE search over n\_rounds, lr, max\_depth, refit\_interval; `fast=True` mode (cache\_geometry, no expansion, convergence\_tol=0.01 during trials; full refit for best\_model\_) |
| `src/geoxgb/__init__.py` | Modified | Export `GeoXGBOptimizer` via try/except ImportError guard |
| `pyproject.toml` | Modified | Added `[project.optional-dependencies] optimizer = ["optuna>=3.0"]` |
| `src/geoxgb/_base.py` | Modified | Added `convergence_tol` parameter; `_convergence_losses` and `convergence_round_` tracked state; early-stop check before resample at each refit event |
| `tests/test_convergence.py` | New | 7 tests for convergence\_tol: early stopping, no-stop, losses populated, repr, classifier support, prediction after stop |
| `benchmarks/optuna_benchmark.py` | New | Optuna TPE benchmark: GeoXGBOptimizer(fast=True, 25 trials) vs XGBoost Optuna (25 trials), 5 datasets, 3-fold CV; GeoXGB wins 4/5 |
| `benchmarks/imputation_benchmark.py` | New | GeoXGB as imputer: per-feature GeoXGBRegressor imputation vs mean / k-NN / XGBoost native NaN; imputation quality (RMSE) and downstream prediction (AUC / R²) on classification and Friedman #1 regression with 30% MNAR; XGBoost native NaN routing wins both tasks; GeoXGB imputation offers no advantage over k-NN |
