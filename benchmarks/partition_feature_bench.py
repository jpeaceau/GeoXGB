"""
Benchmark: does partition_feature improve default performance across datasets?

Compares three configs on each dataset (3-fold or 5-fold CV):
  1. GeoXGB default (partitioner='hvrt')
  2. GeoXGB default + partition_feature=True
  3. XGBoost default (300 estimators)

All use the same CV splits and random_state for fair comparison.
"""
import sys, os, numpy as np
sys.stdout.reconfigure(encoding="utf-8")

from sklearn.datasets import (load_diabetes, load_breast_cancer,
                               load_wine, load_digits, make_friedman1)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def get_dataset(name):
    if name == "diabetes":
        d = load_diabetes(); return d.data, d.target, "regression", 2
    if name == "friedman1":
        X, y = make_friedman1(n_samples=1000, noise=1.0, random_state=42)
        return X, y, "regression", 2
    if name == "breast_cancer":
        d = load_breast_cancer(); return d.data, d.target.astype(float), "binary", 2
    if name == "wine":
        d = load_wine(); return d.data, d.target.astype(float), "multiclass", 3
    if name == "digits":
        d = load_digits(); return d.data, d.target.astype(float), "multiclass", 10


def cv_score(X, y, task, n_classes, params, n_splits, seed=42):
    from geoxgb import GeoXGBRegressor, GeoXGBClassifier
    kf = (KFold if task == "regression" else StratifiedKFold)(
        n_splits=n_splits, shuffle=True, random_state=seed)
    yi = y.astype(int)
    splits = list(kf.split(X) if task == "regression" else kf.split(X, yi))
    Model = GeoXGBRegressor if task == "regression" else GeoXGBClassifier
    scores = []
    for ti, vi in splits:
        m = Model(**params).fit(X[ti], y[ti])
        if task == "regression":
            scores.append(r2_score(y[vi], m.predict(X[vi])))
        else:
            prob = m.predict_proba(X[vi])
            if n_classes == 2:
                scores.append(roc_auc_score(yi[vi], prob[:, 1]))
            else:
                yb = label_binarize(yi[vi], classes=list(range(n_classes)))
                scores.append(roc_auc_score(yb, prob, multi_class="ovr", average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def cv_xgboost(X, y, task, n_classes, n_splits, seed=42):
    import xgboost as xgb
    kf = (KFold if task == "regression" else StratifiedKFold)(
        n_splits=n_splits, shuffle=True, random_state=seed)
    yi = y.astype(int)
    splits = list(kf.split(X) if task == "regression" else kf.split(X, yi))
    scores = []
    for ti, vi in splits:
        if task == "regression":
            m = xgb.XGBRegressor(n_estimators=300, random_state=seed, verbosity=0)
            m.fit(X[ti], y[ti])
            scores.append(r2_score(y[vi], m.predict(X[vi])))
        else:
            obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
            kw = dict(num_class=n_classes) if n_classes > 2 else {}
            m = xgb.XGBClassifier(n_estimators=300, random_state=seed,
                                   objective=obj, verbosity=0, **kw)
            m.fit(X[ti], yi[ti])
            prob = m.predict_proba(X[vi])
            if n_classes == 2:
                scores.append(roc_auc_score(yi[vi], prob[:, 1]))
            else:
                yb = label_binarize(yi[vi], classes=list(range(n_classes)))
                scores.append(roc_auc_score(yb, prob, multi_class="ovr", average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


if __name__ == "__main__":
    datasets = ["diabetes", "friedman1", "breast_cancer", "wine", "digits"]

    print("Partition Feature Benchmark: default vs default+partition_feature vs XGBoost")
    print("=" * 95)

    rows = []
    for name in datasets:
        X, y, task, n_classes = get_dataset(name)
        n_splits = 5 if name in ("diabetes", "wine") else 3
        metric = "R2" if task == "regression" else "AUC"
        print(f"\n[{name}] n={len(X)}, d={X.shape[1]}, {task}, {n_splits}-fold CV")

        # 1. GeoXGB default (hvrt)
        gd, gd_s = cv_score(X, y, task, n_classes, {}, n_splits)
        print(f"  GeoXGB default          {metric}={gd:.4f} +/- {gd_s:.4f}")

        # 2. GeoXGB default + partition_feature
        pf_params = {"partition_feature": True}
        gpf, gpf_s = cv_score(X, y, task, n_classes, pf_params, n_splits)
        delta_pf = gpf - gd
        print(f"  GeoXGB + part_feat      {metric}={gpf:.4f} +/- {gpf_s:.4f}  (delta={delta_pf:+.4f})")

        # 3. XGBoost default
        xd, xd_s = cv_xgboost(X, y, task, n_classes, n_splits)
        print(f"  XGBoost default         {metric}={xd:.4f} +/- {xd_s:.4f}")

        rows.append((name, metric, gd, gd_s, gpf, gpf_s, xd, xd_s, delta_pf))

    print("\n" + "=" * 95)
    print(f"  {'Dataset':<16} {'M':<4}  {'GeoXGB default':>16}  {'+ part_feat':>16}  {'XGBoost':>16}  {'pf delta':>9}")
    print("=" * 95)
    for name, metric, gd, gd_s, gpf, gpf_s, xd, xd_s, delta_pf in rows:
        print(f"  {name:<16} {metric:<4}  {gd:.4f}+/-{gd_s:.4f}  "
              f"{gpf:.4f}+/-{gpf_s:.4f}  {xd:.4f}+/-{xd_s:.4f}  "
              f"{delta_pf:+.4f}")
    print("=" * 95)

    mean_delta = np.mean([r[-1] for r in rows])
    print(f"\nMean partition_feature delta: {mean_delta:+.4f}")
