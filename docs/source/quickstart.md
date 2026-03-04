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

## Causal Inference

For causal inference and treatment effect estimation, use the HVRT partitioner
explicitly. HVRT satisfies Theorem 3 (T⊥Q orthogonality) which makes partitions
invariant to isotropic Gaussian covariate noise — a critical property when
treatment assignment is correlated with covariates.

```python
model = GeoXGBRegressor(partitioner='hvrt')
```

PyramidHART (the regression default) sacrifices this noise-invariance for faster
geometry computation. For causal applications, always use `partitioner='hvrt'`.
