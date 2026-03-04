# HPO Guide

Hyperparameter optimisation is strongly recommended for production use.
The default parameters are well-calibrated for a broad range of datasets, but
`learning_rate` and `max_depth` in particular shift substantially across
different problems — what works for one dataset can differ 5× in `learning_rate`
from what works for another.

---

## What to tune vs what to leave alone

### Prioritise — these drive most of the accuracy

1. **`learning_rate`** — the most impactful single parameter. Search 0.005–0.05.
2. **`max_depth`** — interacts strongly with `learning_rate`. Search 2–6.
3. **`n_rounds`** — scale up freely once `learning_rate` is chosen (no overfitting
   risk at low `learning_rate`). Set high (1 000–5 000) and let it run.
4. **`reduce_ratio`** — significant secondary effect. Search 0.3–0.95.
5. **`refit_interval`** — OAT rank-2. Search 10–200.

### Secondary — tune if primary is already good

6. **`expand_ratio`** — most impactful on small datasets. Search 0.0–0.4.
7. **`y_weight`** — Optuna finds 0.21–0.28 optimal for regression; 0.1–0.5 for
   classification.

### Leave alone — changing hurts unless you know why

- `auto_noise`, `noise_guard`, `refit_noise_floor` — noise modulation defaults
  are well-tuned; disabling them removes robustness guarantees.
- `partitioner`, `method`, `generation_strategy` — geometric pipeline defaults
  are the Optuna-optimised combination. Mismatching them (e.g. `method='orthant_stratified'`
  without `partitioner='pyramid_hart'`) silently degrades performance.
- `tree_splitter`, `tree_criterion`, `variance_weighted` — minor effects;
  only tune after everything else is locked in.

---

## Regression HPO

### Recommended search space

```python
import optuna
from geoxgb import GeoXGBRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

def objective(trial):
    params = dict(
        learning_rate  = trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        max_depth      = trial.suggest_int("max_depth", 2, 5),
        n_rounds       = trial.suggest_int("n_rounds", 500, 3000, step=500),
        reduce_ratio   = trial.suggest_float("reduce_ratio", 0.3, 0.95),
        refit_interval = trial.suggest_int("refit_interval", 10, 200, step=10),
        expand_ratio   = trial.suggest_float("expand_ratio", 0.0, 0.3),
        y_weight       = trial.suggest_float("y_weight", 0.1, 0.5),
        random_state   = 42,
    )
    model = GeoXGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
print(study.best_params)
```

### Known optima (empirical, from 2 000+ trials)

| Dataset | `learning_rate` | `max_depth` | `n_rounds` | `reduce_ratio` | `refit_interval` | R² |
|---|---|---|---|---|---|---|
| diabetes | 0.0125 | 2 | 2 000 | 0.44 | 200 | 0.498 |
| Friedman-1 | 0.0147 | 3 | 1 000 | 0.83 | 10 | 0.932 |

The wide variation between datasets (reduce_ratio 0.44 vs 0.83; refit_interval
200 vs 10) is why HPO is recommended rather than relying on defaults.

---

## Classification HPO

### Recommended search space

```python
from geoxgb import GeoXGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np

def objective(trial):
    params = dict(
        learning_rate  = trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        max_depth      = trial.suggest_int("max_depth", 2, 6),
        n_rounds       = trial.suggest_int("n_rounds", 500, 3000, step=500),
        reduce_ratio   = trial.suggest_float("reduce_ratio", 0.4, 0.95),
        refit_interval = trial.suggest_int("refit_interval", 20, 200, step=20),
        expand_ratio   = trial.suggest_float("expand_ratio", 0.0, 0.3),
        y_weight       = trial.suggest_float("y_weight", 0.1, 0.6),
        random_state   = 42,
    )
    clf = GeoXGBClassifier(**params)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
print(study.best_params)
```

### Default vs XGBoost (no HPO)

| Dataset | GeoXGB default | GeoXGB HPO-best | XGBoost (300 est.) |
|---|---|---|---|
| breast_cancer AUC | 0.9943 | 0.9926 | 0.9886 |
| wine AUC | 0.9951 | 0.9993 | 0.9975 |
| digits AUC | 0.9988 | — | — |

GeoXGB defaults beat XGBoost defaults on all evaluated datasets.

---

## Using GeoXGBOptimizer

The built-in optimizer wraps Optuna with sane defaults for GeoXGB:

```python
from geoxgb.optimizer import GeoXGBOptimizer

opt = GeoXGBOptimizer(task="regression", n_trials=200, cv=5, random_state=42)
opt.fit(X_train, y_train)

print(opt.best_params_)
model = opt.best_estimator_
```

Requires `pip install geoxgb[optimizer]` (installs Optuna).

---

## Tips

- **Start with `n_rounds=1000`** and a log-uniform search over `learning_rate`
  in `[0.005, 0.05]`. The optimal `learning_rate` is the strongest signal in
  any HPO run.
- **Increase `n_rounds` after HPO** — once you have the right `learning_rate`,
  more rounds are free accuracy. With `learning_rate ≤ 0.02` there is no
  overfitting risk.
- **Fix `random_state=42`** during HPO so that trial-to-trial variance comes
  from hyperparameters, not from random seeds.
- **Use 5-fold CV** rather than a single train/test split; GeoXGB's geometry
  varies with data size and single splits can be misleading.
- **On small datasets (n < 500)**, include `expand_ratio` in the search — it
  often provides a large lift by filling in under-sampled geometric regions.
