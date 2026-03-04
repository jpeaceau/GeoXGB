# Quick Start

## Installation

```bash
pip install geoxgb
```

The C++ backend is included in the wheel. No extra dependencies beyond NumPy,
scikit-learn, and `hvrt>=2.6.1`.

## Regression

```python
from geoxgb import GeoXGBRegressor

model = GeoXGBRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

## Classification

```python
from geoxgb import GeoXGBClassifier

clf = GeoXGBClassifier()
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
labels = clf.predict(X_test)
```

## MAE Regression

For tasks where mean absolute error matters more than RMSE:

```python
from geoxgb import GeoXGBMAERegressor

model = GeoXGBMAERegressor()
model.fit(X_train, y_train)
```

## Interpretability

Pass `feature_types` to opt into the full Python interpretability API:

```python
model.fit(X_train, y_train, feature_types=["continuous"] * X_train.shape[1])

# Boosting feature importances
print(model.feature_importances(feature_names))

# Partition (geometry) importances
print(model.partition_feature_importances(feature_names))

# Noise estimate: 1.0 = clean, 0.0 = pure noise
print(model.noise_estimate())

# Sample provenance
print(model.sample_provenance())
```

## Save / Load

```python
model.save("my_model.pkl")

from geoxgb import load_model
model = load_model("my_model.pkl")
```

## Choosing a partitioner

The default `partitioner='pyramid_hart'` is fast and works well in most cases.
However, consider switching to `'hvrt'` in any of these situations:

- **Comparable accuracy:** if both partitioners score similarly on your dataset,
  prefer `'hvrt'` — its noise-invariance guarantee (Theorem 3) means it
  generalises more robustly when production data is noisier than training data.
- **Imbalanced datasets:** HVRT's variance-weighted quadric geometry gives
  minority regions better partition coverage than PyramidHART's polyhedral
  level sets.
- **Causal inference:** always use `'hvrt'`. The T⊥Q orthogonality guarantee
  is critical for unbiased treatment effect estimation.

```python
# Prefer HVRT when accuracy is close, for noise-invariance
model = GeoXGBRegressor(partitioner='hvrt')

# Always use HVRT for causal inference
causal_model = GeoXGBRegressor(partitioner='hvrt')
```
