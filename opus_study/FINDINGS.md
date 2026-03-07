# Block-Size Calibration Study — Findings

## Study 1: Block-Size Formula Sweep (small-to-moderate n)

Tested 6 block-size formulas across 3 datasets, 5 sample sizes (2k-50k),
2 partitioners (pyramid_hart, hvrt), deterministic variance_ordered reduction.
500 rounds, 3-fold CV.

### Key Observations

**1. Three distinct dataset regimes exist:**

| Dataset | Character | GeoXGB vs XGBoost | Block-size preference |
|---|---|---|---|
| linreg_20d | Linear, 20d, high-d informative | GeoXGB WINS at all sizes (+0.025 to +0.015) | Smallest blocks best (current formula) |
| friedman1 | Non-linear interactions, 10d | GeoXGB wins at 2k, loses from 5k+ (-0.004 to -0.013) | Larger blocks help, plateaus around n/3-n/5 |
| california | Real-world, spatial, 8d | GeoXGB always loses (-0.009 to -0.037) | Larger blocks help slightly, gap persists |

**2. Block-size matters most at large n (>10k):**

At n=2k and n=5k, all formulas produce identical results (block cycling disabled below 5k threshold).
The formula only diverges at n=10k+ where block cycling activates.

**3. The "current" formula is too conservative for non-linear tasks:**

At n=50k, `current` gives block=1,400 → R2=0.914 (friedman1), while `tenth` (block=5,000) gives R2=0.922.
The current formula's small blocks discard too much per-block signal on complex tasks.

**4. "none" (full dataset) is NOT optimal — block cycling helps on linreg:**

On linreg_20d at n=50k: current(block=1400) R2=0.912 vs none(full) R2=0.895.
Block cycling acts as regularization on linear/high-d tasks — geometric diversity prevents overfitting.
But on friedman1/california, "none" doesn't help over moderate blocks.

**5. PyramidHART vs HVRT: negligible difference at these defaults**

Both partitioners produce nearly identical R2 across all configs.
Differences are < 0.003 R2. The block-size formula matters far more than partitioner choice.

### Cross-Formula Mean R2 Delta vs XGBoost

| Formula | Mean Delta | Mean Speed | Wins/Total |
|---|---|---|---|
| current | -0.0086 | 0.96x | 11/30 |
| sqrt | -0.0063 | 1.15x | 11/30 |
| log | -0.0076 | 1.22x | 11/30 |
| tenth | -0.0066 | 1.44x | 11/30 |
| fifth | -0.0066 | 2.19x | 11/30 |
| none | -0.0081 | 3.80x | 9/30 |

"sqrt" offers the best overall accuracy-speed tradeoff: close to best accuracy,
only 1.15x XGBoost speed.

## Study 2: Block-Size × Reduce-Ratio Interaction (partial — friedman1 + california)

### Key Findings from friedman1

**At n=10k (best config for accuracy):**
- blk=1000, rr=1.00, er=0.1 → R2=0.9201 (-0.0113 vs XGB) — **best overall**
- blk=full, rr=0.70, er=0.0 → R2=0.9193 (-0.0120)
- Expanding (er=0.1) consistently helps on small blocks but hurts on full data

**At n=20k (best config):**
- blk=n/3(6666), rr=0.80-0.90, er=0.0 → R2=0.9227 (-0.0091) — **best**
- blk=full, rr=1.00, er=0.0 → R2=0.9224 (-0.0095)
- At n/3 blocks, expand is neutral. Reduce ratio barely matters (0.80 ≈ 0.95)

**At n=50k (best config):**
- blk=n/10(5000), rr=0.95, er=0.0 → R2=0.9231 (-0.0103) — **best**
- blk=n/3(16666), rr=0.90, er=0.0 → R2=0.9221 (-0.0113)
- blk=full, rr=1.00, er=0.0 → R2=0.9206 (-0.0127)
- Optimal block IS NOT full data — n/10 beats full while being 3x faster

**Expand ratio at large n:**
- expand_ratio=0.0 is consistently BETTER than 0.1 when block_size >= n/3
- expand_ratio=0.1 helps when block_size is small (< n/5), compensating for data loss
- At large block sizes, synthetic samples are pure noise — real data is sufficient

**Reduce ratio effect:**
- Almost no effect when block >= n/3 (range: ~0.001 R2 across 0.7-1.0)
- Moderate effect when block is small (range: ~0.015 R2)
- rr=0.70 sometimes wins (regularization), rr=1.00 sometimes wins (data access)

### Key Findings from california

- Gap to XGBoost is structural: -0.025 to -0.055 regardless of configuration
- Larger blocks help monotonically (more data → less gap)
- expand_ratio=0.1 helps at small blocks, neutral at large blocks
- reduce_ratio barely matters (0.001 R2 range)

## Study 3: Large-N Coefficient Sweep (10k–500k)

Tested block coefficients [10, 15, 20, 25, 30, 40] for `sqrt(n)*C` formula
across 5 datasets (friedman1, friedman3, linreg_20d, reg_10d, california)
at n = [10k, 20k, 50k, 100k, 300k, 500k]. 3-fold CV, 500 rounds, ri=50.

### The critical finding: block coefficient barely matters at large n

| Coefficient | Mean Delta (n>=100k) | Mean Speed |
|---|---|---|
| C=10 | -0.01232 | 3.1x |
| C=15 | -0.01221 | 3.5x |
| C=20 | -0.01201 | 4.0x |
| C=25 | -0.01216 | 4.5x |
| C=30 | -0.01230 | 5.0x |
| C=40 | -0.01261 | 6.1x |

R2 range across all coefficients is < 0.001 at n>=100k. The ~0.012 gap vs
XGBoost is **structural**, not block-size dependent.

### Per-dataset patterns at large n

| Dataset | Best C | Behavior |
|---|---|---|
| linreg_20d | C=10 always | GeoXGB beats XGBoost up to 300k. Smallest blocks = strongest regularization. Advantage shrinks with n. |
| friedman3 | C=40 always | Larger blocks consistently better but gap grows with n regardless. |
| friedman1 | Varies (noise) | No consistent winner. Best C jumps between 10, 15, 25, 30 randomly across n. |
| reg_10d | C=10-15 | GeoXGB wins at 10k-20k, loses from 50k+. Coefficient barely matters (0.001 range). |
| california | C=25-30 | Structural gap ~0.03-0.04, grows with n. Block coefficient irrelevant (0.003 range). |

### Speed implication

Since accuracy is nearly identical across coefficients, **C=10 is optimal for speed** —
gives 2-3x XGBoost speed at large n vs 5-8x for C=40.

## Study 4: Refit-Interval × Block-Size Interaction

Tested block coefficients [10, 15, 25, 40] × refit_intervals [25, 50, 100, 200]
across 3 datasets (friedman1, linreg_20d, reg_10d) at n = [20k, 50k, 100k].
3-fold CV, 500 rounds.

### The interaction is massive

| refit_interval | Mean Best Delta | Wins/9 | Best Config |
|---|---|---|---|
| **ri=25** | **+0.0008** | **4/9** | Only setting with positive mean delta |
| ri=50 | -0.0000 | 4/9 | Neutral — current default |
| ri=100 | -0.0017 | 4/9 | Moderate degradation |
| ri=200 | -0.0047 | 2/9 | Significant degradation |

### Mechanism: data coverage

| ri | Switches (500 rounds) | Coverage at C=10 n=100k |
|---|---|---|
| 25 | 20 | 63% |
| 50 | 10 | 32% |
| 100 | 5 | 16% |
| 200 | 2 | 6% |

More refits = more block switches = more data coverage + more geometric diversity.

### Key finding: block coefficient and refit_interval are NOT independent

- At ri=25: C=10 wins (smallest blocks, most geometric diversity per refit)
- At ri=200: C=25 or C=40 wins (need larger blocks to compensate for fewer switches)

### Optimal combo

**ri=25, C=10**: mean_delta = **-0.0001** at **2.8x speed** — best accuracy-speed
tradeoff across all configurations tested. Achieves 84% average coverage while
keeping blocks small for speed.

### Per-dataset behavior

**friedman1 n=100k** — regularization-dominated:
- ri=100, C=10 (cov=16%) → R2=0.9240 (-0.0100) ← BEST in dataset
- ri=25, C=10 (cov=63%) → R2=0.9215 (-0.0125)
- Less coverage = better for friedman1 (regularization effect)

**linreg_20d n=100k** — coverage-dominated:
- ri=25, C=10 (cov=63%) → R2=0.9089 (+0.0109) ← BEST in dataset
- ri=200, C=10 (cov=6%) → R2=0.8966 (-0.0014)
- More coverage = monotonically better for linreg

**reg_10d n=100k** — coverage-dominated:
- ri=25, C=10 (cov=63%) → R2=0.9692 (-0.0027) ← BEST
- ri=200, C=10 (cov=6%) → R2=0.9652 (-0.0067)
- More coverage = monotonically better

## Final Auto Formula

### Implemented: `max(2000, int(sqrt(n) * 15 * ri_scale))`

Where `ri_scale = clamp(refit_interval / 50, 0.5, 2.0)`.

| ri | Effective C | Behavior |
|---|---|---|
| 25 | 7.5 | Smaller blocks — more refits compensate, faster execution |
| 50 | 15.0 | Baseline (same as Study 1 recommendation) |
| 100 | 30.0 | Larger blocks — fewer refits need more data per block |
| 200 | 30.0 | Capped at 2x — can't fully fix rare refits |

Rationale:
- Study 3 showed block coefficient barely matters (< 0.001 R2) within a given ri
- Study 4 showed ri is the dominant parameter (0.005+ R2 range)
- The ri_scale adjusts blocks proportionally to refit frequency
- Capped at [0.5, 2.0]x to prevent extreme values
- n_rounds accepted but unused — keeps formula simple; ri is the primary lever

### Optimizer changes

- HPO threshold raised: 5,000 → 30,000 (block cycling hurts HPO at moderate n)
- Removed `None` from HPO search space (full-dataset mode is both slower and worse)
- Default in HPO: `'auto'` (uses this formula)

### Remaining structural gap

~0.012 R2 gap vs XGBoost at large n persists regardless of block size or ri.
This is architectural: GeoXGB's resampling + geometric partitioning per-tree
trades some per-tree information for geometric interpretability. The gap does
NOT grow with n beyond a plateau.

Potential approaches to close (not yet tested):
1. Multi-block accumulation per tree
2. Residual warm-starting from prior blocks
3. Hybrid mode: full data for trees, HVRT for interpretability only

## Study 5: Dimensionality-Dependent Formula Validation

Tested a geometry-derived formula `max(2000, 20 * int(26.6 * d))` (≈ 540d)
against the ri-scaled `sqrt(n)*15` formula across 4 datasets with d=4,8,10,20
at n=10k,50k,100k. 3-fold CV, 500 rounds, ri=50.

### Hypothesis (rejected)

HVRT auto-tunes `min_samples_leaf ≈ 27d`. A block needs ~20 partitions →
block ≈ 540d. This should give better results at high-d where sqrt(n)*15
undersizes blocks.

### Results

| Dataset (d) | d-form Δ vs XGB | sqrt Δ vs XGB | d-form advantage |
|---|---|---|---|
| reg_4d (4) | -0.0045 | -0.0043 | -0.0003 (tie) |
| california (8) | -0.0350 | -0.0342 | -0.0008 (sqrt) |
| friedman1 (10) | -0.0119 | **-0.0139** | **+0.0020** (d-form) |
| linreg_20d (20) | +0.0039 | **+0.0121** | **-0.0082** (sqrt) |

Overall: sqrt wins 6/12, d-form wins 3/12, ties 3/12.
Mean delta: sqrt -0.010, d-form -0.012.

### Why the geometric formula fails

The 540d derivation correctly identifies the *minimum* for stable HVRT
partitioning, but:

1. **Regularisation through cycling dominates**: Smaller blocks → more block
   switches → more geometric diversity. This is the mechanism behind GeoXGB's
   advantage on linreg_20d (+0.012 with small blocks vs +0.004 with large).

2. **High-d inflation is harmful**: At d=20, 540×20=10,800 → at n=10k (CV
   train ~6,667) this disables cycling entirely. The regularisation benefit
   of cycling outweighs the partition quality gain of larger blocks.

3. **Block cycling compensates for undersized blocks**: Even if a single block
   has too few samples for perfect 20-partition resolution, subsequent blocks
   reveal different geometric facets. The cumulative coverage exceeds any
   single block.

### Conclusion

The ri-scaled sqrt(n) formula `max(2000, int(sqrt(n) * 15 * ri_scale))` is
retained. The geometric minimum 540d is a useful theoretical lower bound but
not an optimal operating point.
