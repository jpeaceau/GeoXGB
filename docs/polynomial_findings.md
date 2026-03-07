# Higher-Order Polynomial Features in GeoXGB

## Problem Statement

GeoXGB's geometry is built on T = S² − Q, which is exactly 2·e₂ — the second
elementary symmetric polynomial of the whitened features. HVRT partitions,
reduces, and expands samples based on T-level hyperboloids. This is degree-2
only.

The noise invariance theorem (Theorem 3 in the paper) relies on T being a
**matched-degree subtraction**: both S² and Q are degree-2 in the z-scores,
so the additive noise bias (dσ²) cancels exactly: E[T̃] = E[T].

**Question:** Can we extend this to degree 3 and 4? Higher-order interactions
(e.g., y = a·b·c or y = a·b·c·d) exist in real data. HVRT's T-only geometry
is blind to them — two samples with identical T but opposite degree-3 structure
land in the same partition.

### Why This Matters

- Tree-based models capture higher-order interactions implicitly through
  sequential splits, but this is approximate and not noise-invariant.
- Elementary symmetric polynomials e_k are **exactly** noise-invariant at
  every degree k (distinct-index products kill all noise cross-terms in
  expectation). This is not an approximation.
- If we could give GeoXGB explicit access to e₃ and e₄ structure, it could
  partition and reduce with awareness of triple and quadruple interactions.

### Mathematical Background

Newton's identities express e_k via power sums p_j = Σz_i^j, all O(n·d):

```
e₁ = S                                          (degree 1)
e₂ = (S² − Q) / 2 = T / 2                      (degree 2)
e₃ = (S³ + 2·p₃ − 3·Q·S) / 6                   (degree 3)
e₄ = (S⁴ − 6·S²·Q + 3·Q² + 8·S·p₃ − 6·p₄) / 24  (degree 4)
```

Each e_k is noise-invariant because it is a sum over products of DISTINCT
indices. When noise at each index is independent zero-mean, expectation kills
every term containing a noise factor that appears only once.

---

## Approaches Tested

### 1. Feature Augmentation (e_k as GBT input columns)

**Idea:** Compute e₂, e₃, e₄ scalar aggregates from whitened Z, append as
extra features, then re-fit GeoXGB on the augmented matrix.

**Implementation:** `src/geoxgb/experimental/_e3_augment.py`
- `AdaptiveEkGeoXGBRegressor`: two-stage fit (base → residuals → augment → re-fit)
- Scalar aggregates e₂, e₃, e₄ (always included, noise-invariant)
- Per-feature partials e₃^(j) = z_j · e₂(z_{-j}), e₄^(j) = z_j · e₃(z_{-j})
- Selective k-tuples ranked by residual correlation, noise-gated:
  budget_k = floor(top_k_max · noise_estimate^(α·(k−1)))

**Results across 20 datasets (12 synthetic, 8 real):**

| Config     | Mean Δ R² | Win/Neutral/Loss |
|------------|-----------|------------------|
| Ek d=3     | −0.017    | 2 / 8 / 9        |
| Ek d=4     | −0.021    | 3 / 7 / 9        |
| Ek d=4+p   | −0.059    | 4 / 5 / 10       |

**Why it fails:**
- Scalar e₃ averages ALL C(d,3) triples. When only one specific triple
  matters, the aggregate dilutes the signal below the noise floor.
- The two-stage re-fit disrupts HVRT geometry: the augmented feature space
  changes whitening and partition structure.
- Per-feature partials (`+p`) add d extra noisy dimensions per degree.
- The noise estimator reads ≈0 on clean synthetic data, gating off all
  selective tuples even with noise_floor=0.

**One exception:** Diabetes consistently gains +0.07–0.08 R² because d=10 is
small enough that the aggregates retain meaningful signal.

### 2. Reduction Alignment (joint e₂/e₃/e₄ importance for sample selection)

**Idea:** HVRT reduces by T-variance, which discards samples important for
e₃/e₄ structure. If we reduce using a combined criterion
(√(e₂² + e₃² + e₄²)), samples important for ANY degree survive.

**Implementation:** `src/geoxgb/experimental/_ek_aligned.py`
- `EkAlignedRegressor`: joint multi-degree reduction + HistGBT
- `TReducedEkRegressor`: T-only reduction + e_k augmentation (control)
- `PlainEkRegressor`: no reduction, e_k augmentation (isolates reduction variable)

**Ablation results (14 datasets):**

| Config           | Mean Δ R² | Interpretation                        |
|------------------|-----------|---------------------------------------|
| T-reduce + GBT   | −0.008    | Reduction alone costs ~0.8%           |
| HistGBT + e_k    | −0.014    | e_k features hurt even without reduction |
| T-reduce + e_k   | −0.018    | Combined: worse                       |
| Joint-reduce + e_k | −0.021  | Joint reduction: actually worse       |

**Key finding:** `HistGBT + e_k` (no reduction at all) still hurts. The
problem is NOT that HVRT discards the wrong samples — the scalar e_k features
themselves are not useful GBT features for most datasets. Reduction alignment
is not the bottleneck.

### 3. Multiplicative Composite HVRT Target

**Idea:** Instead of augmenting features, change what HVRT partitions BY.
Use |T| × |e₃| as the target statistic so partitions align with BOTH degree-2
and degree-3 structure.

**Implementation:** `src/geoxgb/experimental/_ek_target.py`
- `CompositeHVRTRegressor` with pluggable target functions
- Tested: T×e₃, rank-normalized product, additive blend

**Results (14 datasets, vs HVRT(T)):**

| Target     | Wins | Losses | Mean diff vs T |
|------------|------|--------|----------------|
| T×e₃       | 6    | 5      | −0.003         |
| Rank prod  | 5    | 6      | +0.006         |

**Conclusion:** Multiplicative composition is a wash. The product zeros out
when either factor is zero, losing too much information.

### 4. Blended HVRT Target: T + λ·e₃ ★

**Idea:** Additive blend of T and e₃ (each normalized to unit variance) as
the HVRT y-signal. This preserves information from both degrees. λ controls
the degree-3 weight.

**Implementation:** `benchmarks/ek_blended_target_bench.py`,
`benchmarks/ek_blended_comprehensive.py`

**Results (20 datasets, 16-config HPO with 3-fold CV):**

| Config          | Mean Δ R² | W/N/L vs HistGBT |
|-----------------|-----------|------------------|
| HVRT(T) rr=0.8  | −0.009    | 3 / 5 / 12       |
| T+2e₃ rr=0.8    | −0.003    | 4 / 9 / 7        |
| HPO Blend        | −0.003    | 4 / 13 / 3       |

**Head-to-head HPO Blend vs HVRT(T): 12 wins, 5 losses, 3 ties.**

HPO Blend cuts the reduction penalty by ~65% and wins 12/20 datasets
head-to-head vs T-only. The improvement is +0.006 mean Δ R² over T-only.

**What HPO selects:**
- Datasets with no higher-order interactions → λ₃=0, rr=1.0 (opts out correctly)
- Degree-3+ datasets → λ₃ ∈ {1.5, 2.0, 3.0} with reduction active
- The blend only matters when reduction is active (at rr=1.0, blended target
  matches HistGBT exactly across all 20 datasets)

**Why it works (modestly):**
- e₃ preserves sign — it tells HVRT whether triple cooperation is positive or
  negative, which T alone cannot see.
- The additive blend preserves information from both degrees (unlike the
  multiplicative product which zeros out).
- Higher λ₃ is needed because T has more natural variance (more features
  contribute to degree-2 than to degree-3).

---

## Conclusions

1. **Feature augmentation with e_k scalars does not work.** The aggregates
   dilute sparse interactions across all C(d,k) terms. Individual k-tuples
   could help but require noise gating, and the noise estimator is unreliable
   on clean/synthetic data.

2. **Reduction alignment is not the bottleneck.** Even with no reduction at
   all, e_k features hurt on average. The problem is the features, not which
   samples are kept.

3. **Blending e₃ into the HVRT target is a genuine but modest improvement.**
   It helps reduction preserve samples important for degree-3 structure (+0.006
   mean, 12/20 head-to-head wins). But it requires per-dataset λ₃ tuning and
   only matters when reduction is active.

4. **The fundamental limitation:** HVRT cannot simultaneously optimize for
   degree-2 AND degree-3 variance retention without a composite target. The
   blended target is the best compromise tested, but the effect is small
   because most real datasets are dominated by degree-1 and degree-2 structure.

5. **For inclusion in GeoXGB:** The blended target (T + λ·e₃) is the only
   approach that passes the "do no harm" test (HPO correctly opts out when
   e₃ signal is absent). A C++ implementation would require auto-tuning λ₃ —
   possible candidates: ratio of e₃ variance to T variance, or the noise
   estimate. The modest effect size (+0.006 mean) may not justify the
   complexity.

---

## Files

| File | Description |
|------|-------------|
| `src/geoxgb/experimental/_e3_augment.py` | e_k feature augmentation classes |
| `src/geoxgb/experimental/_ek_aligned.py` | Joint multi-degree reduction experiment |
| `src/geoxgb/experimental/_ek_target.py` | Composite HVRT target experiment |
| `benchmarks/e3_augment_bench.py` | Initial augmentation benchmark |
| `benchmarks/e3_augment_extended_bench.py` | Extended augmentation (20 datasets) |
| `benchmarks/ek_aligned_bench.py` | Reduction alignment ablation |
| `benchmarks/ek_target_bench.py` | Multiplicative target benchmark |
| `benchmarks/ek_blended_target_bench.py` | Blended T+λ·e₃ target benchmark |
| `benchmarks/ek_blended_comprehensive.py` | Comprehensive blended target with HPO |
