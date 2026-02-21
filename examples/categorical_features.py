"""
Mixed continuous + categorical features example.

Categorical features must be pre-encoded to numeric before passing to
GeoXGBRegressor/Classifier.  Use feature_types to tell HVRT which columns
to treat as categorical (separate z-scoring + frequency sampling in KDE).
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from geoxgb import GeoXGBRegressor

rng = np.random.default_rng(42)
N = 400

# Simulate mixed dataset
region = rng.choice(["north", "south", "east", "west"], size=N)
product = rng.choice(["A", "B", "C"], size=N)
X_cont = rng.standard_normal((N, 3))

le_region = LabelEncoder()
le_product = LabelEncoder()
region_enc = le_region.fit_transform(region).astype(np.float64)
product_enc = le_product.fit_transform(product).astype(np.float64)

X = np.column_stack([X_cont, region_enc, product_enc])
y = (
    2 * X[:, 0]
    - X[:, 1]
    + 0.5 * region_enc          # region has real effect
    + 1.0 * (product_enc == 0)  # product A has bonus
    + rng.standard_normal(N) * 0.2
)

feature_types = [
    "continuous", "continuous", "continuous",
    "categorical",  # region
    "categorical",  # product
]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)

reg = GeoXGBRegressor(n_rounds=80, learning_rate=0.1, random_state=42)
reg.fit(X_train, y_train, feature_types=feature_types)
y_pred = reg.predict(X_test)

print(f"R²: {r2_score(y_test, y_pred):.4f}")

feature_names = ["x0", "x1", "x2", "region", "product"]
print("\nFeature importances (boosting):")
for name, imp in reg.feature_importances(feature_names).items():
    print(f"  {name}: {imp:.4f}")

print("\nPartition feature importances (geometry):")
for entry in reg.partition_feature_importances(feature_names):
    print(f"  round={entry['round']}: {entry['importances']}")
