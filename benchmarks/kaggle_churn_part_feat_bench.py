"""Test partition_feature and global_geometry_n on churn dataset.
Run each config in a subprocess to avoid memory accumulation."""
import sys, subprocess, json
sys.stdout.reconfigure(encoding="utf-8")

CONFIGS = {
    "baseline":           '{"n_rounds":1000,"learning_rate":0.02,"max_depth":3,"refit_interval":50,"random_state":42}',
    "partition_feature":  '{"n_rounds":1000,"learning_rate":0.02,"max_depth":3,"refit_interval":50,"random_state":42,"partition_feature":true}',
    "global_geom_20k":   '{"n_rounds":1000,"learning_rate":0.02,"max_depth":3,"refit_interval":50,"random_state":42,"global_geometry_n":20000}',
    "both_20k":          '{"n_rounds":1000,"learning_rate":0.02,"max_depth":3,"refit_interval":50,"random_state":42,"partition_feature":true,"global_geometry_n":20000}',
    "global_geom_50k":   '{"n_rounds":1000,"learning_rate":0.02,"max_depth":3,"refit_interval":50,"random_state":42,"global_geometry_n":50000}',
    "both_50k":          '{"n_rounds":1000,"learning_rate":0.02,"max_depth":3,"refit_interval":50,"random_state":42,"partition_feature":true,"global_geometry_n":50000}',
    "pf_depth4":         '{"n_rounds":1000,"learning_rate":0.02,"max_depth":4,"refit_interval":50,"random_state":42,"partition_feature":true}',
    "both_50k_depth4":   '{"n_rounds":1000,"learning_rate":0.02,"max_depth":4,"refit_interval":50,"random_state":42,"partition_feature":true,"global_geometry_n":50000}',
}

WORKER = '''
import sys, json, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from geoxgb import GeoXGBClassifier

params = json.loads(sys.argv[1])
df = pd.read_csv("data/train.csv")
y = (df["Churn"] == "Yes").astype(int).values.astype(np.float64)
feat_cols = [c for c in df.columns if c not in ["id", "Churn"]]
for c in feat_cols:
    if df[c].dtype == object:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
X = df[feat_cols].values.astype(np.float64)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
tr, va = next(iter(skf.split(X, y)))

t0 = time.time()
clf = GeoXGBClassifier(**params)
clf.fit(X[tr], y[tr])
t1 = time.time()

proba = clf.predict_proba(X[va])[:, 1]
auc = roc_auc_score(y[va], proba)
part_ids = np.asarray(clf._cpp_model.apply(X[tr]))
n_parts = len(np.unique(part_ids))
fimp = np.array(clf._cpp_model.feature_importances())
n_used = int((fimp > 0).sum())
print(json.dumps({"auc": auc, "n_parts": n_parts, "n_used": n_used, "n_feats": len(fimp), "time": t1-t0}))
'''

print(f"XGBoost HPO ceiling: 0.9163\n")

for name, params_json in CONFIGS.items():
    result = subprocess.run(
        [sys.executable, "-c", WORKER, params_json],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f"{name:25s}: CRASHED (code={result.returncode})")
        if result.stderr:
            print(f"  stderr: {result.stderr[:200]}")
    else:
        r = json.loads(result.stdout.strip())
        print(f"{name:25s}: AUC={r['auc']:.4f} | {r['n_parts']:3d} parts | {r['n_used']:2d}/{r['n_feats']} feats | {r['time']:.1f}s")
