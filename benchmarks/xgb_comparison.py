"""
Head-to-head: GeoXGB (new defaults) vs GeoXGB (HPO-best) vs XGBoost (default).
Same CV protocol for every model on every dataset. 5 datasets in parallel.
"""
import multiprocessing, os, numpy as np
from sklearn.datasets import (load_diabetes, load_breast_cancer,
                               load_wine, load_digits, make_friedman1)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import label_binarize

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

HPO_BEST = {
    "diabetes": dict(
        partitioner="pyramid_hart", method="variance_ordered",
        generation_strategy="simplex_mixup",
        n_rounds=1000, learning_rate=0.0125, max_depth=2,
        refit_interval=200, expand_ratio=0.017, reduce_ratio=0.44,
        y_weight=0.21, min_samples_leaf=14, auto_expand=True,
        hvrt_min_samples_leaf=10, auto_noise=True, noise_guard=False,
    ),
    "friedman1": dict(
        partitioner="pyramid_hart", method="variance_ordered",
        generation_strategy="simplex_mixup",
        n_rounds=5000, learning_rate=0.0147, max_depth=3,
        refit_interval=10, expand_ratio=0.356, reduce_ratio=0.83,
        y_weight=0.28, min_samples_leaf=8, auto_expand=False,
        hvrt_min_samples_leaf=33, auto_noise=False, noise_guard=True,
    ),
    "breast_cancer": dict(
        partitioner="pyramid_hart", method="variance_ordered",
        generation_strategy="simplex_mixup",
        n_rounds=3000, learning_rate=0.0858, max_depth=5,
        refit_interval=200, expand_ratio=0.395, reduce_ratio=0.355,
        y_weight=0.496, min_samples_leaf=2, auto_expand=True,
        hvrt_min_samples_leaf=36, auto_noise=True, noise_guard=False,
    ),
    "wine": dict(
        partitioner="pyramid_hart", method="variance_ordered",
        generation_strategy="simplex_mixup",
        n_rounds=1000, learning_rate=0.0897, max_depth=3,
        refit_interval=50, expand_ratio=0.057, reduce_ratio=0.855,
        y_weight=0.802, min_samples_leaf=8, auto_expand=False,
        hvrt_min_samples_leaf=24, auto_noise=True, noise_guard=False,
    ),
    "digits": dict(
        partitioner="pyramid_hart", method="variance_ordered",
        generation_strategy="simplex_mixup",
        n_rounds=3000, learning_rate=0.05, max_depth=4,
        refit_interval=50, expand_ratio=0.1, reduce_ratio=0.8,
        y_weight=0.5, min_samples_leaf=5, auto_expand=True,
        hvrt_min_samples_leaf=15, auto_noise=True, noise_guard=False,
    ),
}

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

def cv_geoxgb(X, y, task, n_classes, params, n_splits, seed=42):
    from geoxgb import GeoXGBRegressor, GeoXGBClassifier
    kf = (KFold if task == "regression" else StratifiedKFold)(
        n_splits=n_splits, shuffle=True, random_state=seed)
    yi = y.astype(int)
    splits = kf.split(X) if task == "regression" else kf.split(X, yi)
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
    splits = kf.split(X) if task == "regression" else kf.split(X, yi)
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

def run_dataset(name):
    X, y, task, n_classes = get_dataset(name)
    n_splits = 5 if name in ("diabetes", "wine") else 3
    metric = "R2" if task == "regression" else "AUC"
    print(f"[{name}] n={len(X)} {task} {n_splits}-fold", flush=True)
    gd, gd_s = cv_geoxgb(X, y, task, n_classes, {}, n_splits)
    print(f"  GeoXGB default  {metric}={gd:.4f}+/-{gd_s:.4f}", flush=True)
    gh, gh_s = cv_geoxgb(X, y, task, n_classes, HPO_BEST[name], n_splits)
    print(f"  GeoXGB HPO-best {metric}={gh:.4f}+/-{gh_s:.4f}", flush=True)
    xd, xd_s = cv_xgboost(X, y, task, n_classes, n_splits)
    print(f"  XGBoost default {metric}={xd:.4f}+/-{xd_s:.4f}", flush=True)
    return name, metric, gd, gd_s, gh, gh_s, xd, xd_s

if __name__ == "__main__":
    datasets = ["diabetes", "friedman1", "breast_cancer", "wine", "digits"]
    print("GeoXGB default / GeoXGB HPO-best / XGBoost default\n")
    with multiprocessing.Pool(processes=len(datasets)) as pool:
        results = pool.map(run_dataset, datasets)
    print("\n" + "=" * 80)
    print(f"  {'Dataset':<14} {'M':<4}  {'GeoXGB default':>16}  {'GeoXGB HPO':>16}  {'XGBoost default':>16}  {'HPO-XGB':>8}")
    print("=" * 80)
    for name, metric, gd, gd_s, gh, gh_s, xd, xd_s in results:
        delta = gh - xd
        print(f"  {name:<14} {metric:<4}  {gd:.4f}+/-{gd_s:.4f}  "
              f"{gh:.4f}+/-{gh_s:.4f}  {xd:.4f}+/-{xd_s:.4f}  "
              f"{delta:+.4f}")
    print("=" * 80)
