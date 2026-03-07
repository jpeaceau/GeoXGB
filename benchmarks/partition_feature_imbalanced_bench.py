"""
Synthetic benchmark: partition_feature utility under imbalanced X and Y.

Creates a dataset where:
- Y is imbalanced (10% positive class)
- X has imbalanced categorical features (rare categories matter for Y)
- Signal lives in interactions between rare X categories and secondary features
- N is large enough to trigger block cycling (50k samples)

The key question: does partition_feature help GBT trees discover signals
that live in rare X x Y intersections when block cycling limits per-block
sample counts?

Configs tested:
  1. GeoXGB default (hvrt, no partition_feature)
  2. GeoXGB + partition_feature=True
  3. GeoXGB + partition_feature=True + global_geometry_n
  4. XGBoost default (300 estimators)
"""
import sys, time
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def make_imbalanced_dataset(n=50000, d_noise=10, random_state=42):
    """
    Synthetic dataset mimicking large-n imbalanced classification:

    Features:
      x0: dominant categorical (3 levels: 60%, 30%, 10%)
      x1: secondary categorical (4 levels: 50%, 25%, 15%, 10%)
      x2-x5: continuous features with weak main effects
      x6+: pure noise

    Target:
      P(y=1) depends on:
        - Strong main effect from x0 (dominant, easy to find)
        - Moderate main effect from x2 (continuous, easy)
        - Weak interaction: x1_rare * x3 (only matters for rare x1 category)
        - Weak interaction: x0_rare * x1_rare * x4 (triple interaction in rarest X stratum)
      Base rate ~10% positive.

    The triple interaction x0_rare * x1_rare * x4 affects only ~1% of samples
    (10% x0_rare * 10% x1_rare). In a block of 5000, that's ~50 samples,
    of which ~5 are positive. This is the regime where partition_feature
    should help: the global HVRT sees all 50k samples and can identify this
    geometric neighborhood, encoding it as a partition_id that block-local
    trees can split on.
    """
    rng = np.random.default_rng(random_state)

    d_total = 6 + d_noise
    X = np.zeros((n, d_total))

    # x0: dominant categorical (encoded as 0, 1, 2)
    x0_probs = [0.60, 0.30, 0.10]
    X[:, 0] = rng.choice([0, 1, 2], size=n, p=x0_probs).astype(float)

    # x1: secondary categorical (encoded as 0, 1, 2, 3)
    x1_probs = [0.50, 0.25, 0.15, 0.10]
    X[:, 1] = rng.choice([0, 1, 2, 3], size=n, p=x1_probs).astype(float)

    # x2-x5: continuous features
    for j in range(2, 6):
        X[:, j] = rng.standard_normal(n)

    # x6+: noise features
    for j in range(6, d_total):
        X[:, j] = rng.standard_normal(n)

    # Build log-odds for P(y=1)
    logit = np.full(n, -2.2)  # base rate ~10% before adjustments

    # Strong main effect: x0 category
    logit += np.where(X[:, 0] == 0, -0.5, 0.0)   # majority: lower risk
    logit += np.where(X[:, 0] == 1,  0.5, 0.0)    # mid: moderate risk
    logit += np.where(X[:, 0] == 2,  1.5, 0.0)    # rare 10%: high risk

    # Strong main effect: x2 continuous
    logit += 0.8 * X[:, 2]

    # Interaction: x1==3 (rare, 10%) amplifies x3
    mask_x1_rare = (X[:, 1] == 3)
    logit += np.where(mask_x1_rare, 1.2 * X[:, 3], 0.0)

    # Triple interaction: x0==2 AND x1==3 (1% of samples) amplifies x4
    mask_triple = (X[:, 0] == 2) & (X[:, 1] == 3)
    logit += np.where(mask_triple, 2.0 * X[:, 4], 0.0)

    # Interaction: x0==1 AND x2 > 0 amplifies x5
    mask_mid_pos = (X[:, 0] == 1) & (X[:, 2] > 0)
    logit += np.where(mask_mid_pos, 1.0 * X[:, 5], 0.0)

    prob = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, prob).astype(float)

    print(f"  Dataset: n={n}, d={d_total}, y_mean={y.mean():.3f}")
    print(f"  x0 distribution: {np.bincount(X[:,0].astype(int)) / n}")
    print(f"  x1 distribution: {np.bincount(X[:,1].astype(int)) / n}")
    print(f"  Triple interaction stratum (x0=2,x1=3): "
          f"{mask_triple.sum()} samples ({mask_triple.sum()/n*100:.1f}%), "
          f"y_mean={y[mask_triple].mean():.3f}")
    print(f"  x1_rare stratum (x1=3): "
          f"{mask_x1_rare.sum()} samples ({mask_x1_rare.sum()/n*100:.1f}%), "
          f"y_mean={y[mask_x1_rare].mean():.3f}")

    return X, y


def cv_geoxgb(X, y, params, n_splits=3, seed=42):
    from geoxgb import GeoXGBClassifier
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yi = y.astype(int)
    scores = []
    t0 = time.perf_counter()
    for ti, vi in kf.split(X, yi):
        m = GeoXGBClassifier(**params).fit(X[ti], y[ti])
        prob = m.predict_proba(X[vi])
        scores.append(roc_auc_score(yi[vi], prob[:, 1]))
    elapsed = time.perf_counter() - t0
    return float(np.mean(scores)), float(np.std(scores)), elapsed


def cv_xgboost(X, y, n_splits=3, seed=42):
    import xgboost as xgb
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yi = y.astype(int)
    scores = []
    t0 = time.perf_counter()
    for ti, vi in kf.split(X, yi):
        m = xgb.XGBClassifier(n_estimators=300, random_state=seed,
                               objective="binary:logistic", verbosity=0)
        m.fit(X[ti], yi[ti])
        prob = m.predict_proba(X[vi])
        scores.append(roc_auc_score(yi[vi], prob[:, 1]))
    elapsed = time.perf_counter() - t0
    return float(np.mean(scores)), float(np.std(scores)), elapsed


if __name__ == "__main__":
    print("Partition Feature: Imbalanced X+Y Synthetic Benchmark")
    print("=" * 80)

    for n in [10000, 50000]:
        print(f"\n--- N = {n:,} ---")
        X, y = make_imbalanced_dataset(n=n)

        more_rounds = {"n_rounds": 3000, "learning_rate": 0.015, "max_depth": 5}
        configs = [
            ("GeoXGB default", {}),
            ("GeoXGB + pf", {"partition_feature": True}),
            ("GeoXGB tuned", {**more_rounds}),
            ("GeoXGB tuned + pf", {**more_rounds, "partition_feature": True}),
            ("GeoXGB tuned + pf + gg",
             {**more_rounds, "partition_feature": True,
              "global_geometry_n": min(n, 20000)}),
        ]

        results = []
        for label, params in configs:
            auc, std, t = cv_geoxgb(X, y, params)
            print(f"  {label:<28} AUC={auc:.4f} +/- {std:.4f}  ({t:.1f}s)")
            results.append((label, auc, std, t))

        # XGBoost baseline
        auc, std, t = cv_xgboost(X, y)
        print(f"  {'XGBoost default':<28} AUC={auc:.4f} +/- {std:.4f}  ({t:.1f}s)")
        results.append(("XGBoost default", auc, std, t))

        print()
        baseline = results[0][1]
        for label, auc, std, t in results:
            delta = auc - baseline
            marker = " <-- baseline" if delta == 0 else ""
            print(f"  {label:<28} delta={delta:+.4f}{marker}")

    print("\n" + "=" * 80)
