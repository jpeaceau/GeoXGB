"""
Basic regression example with GeoXGBRegressor.
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from geoxgb import GeoXGBRegressor

rng = np.random.default_rng(42)

X, y = make_regression(n_samples=500, n_features=8, n_informative=5,
                        noise=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)

reg = GeoXGBRegressor(
    n_rounds=100,
    learning_rate=0.1,
    max_depth=4,
    reduce_ratio=0.7,
    refit_interval=10,
    auto_noise=True,
    random_state=42,
)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"Trees: {reg.n_trees}")
print(f"Resamples: {reg.n_resamples}")
print(f"Noise estimate: {reg.noise_estimate():.3f}")
print(f"Sample provenance: {reg.sample_provenance()}")

feature_names = [f"feature_{i}" for i in range(X.shape[1])]
print("\nTop feature importances (boosting):")
for name, imp in list(reg.feature_importances(feature_names).items())[:5]:
    print(f"  {name}: {imp:.4f}")
