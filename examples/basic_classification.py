"""
Basic classification example with GeoXGBClassifier.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from geoxgb import GeoXGBClassifier

X, y = make_classification(
    n_samples=500, n_features=8, n_informative=5,
    n_classes=3, n_clusters_per_class=1, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42)

clf = GeoXGBClassifier(
    n_rounds=100,
    learning_rate=0.1,
    max_depth=4,
    reduce_ratio=0.7,
    refit_interval=10,
    auto_noise=True,
    random_state=42,
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClassification report:\n{classification_report(y_test, y_pred)}")
print(f"predict_proba shape: {y_proba.shape}")
print(f"Row sums (first 5): {y_proba[:5].sum(axis=1)}")
