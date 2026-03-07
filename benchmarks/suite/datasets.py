"""
Dataset manifest for the benchmarking suite.

Each dataset is a dict with:
    name, category, task ('regression'|'binary'|'multiclass'),
    loader (callable returning X, y)

All datasets are deterministic (sklearn built-ins + synthetic generators).
No network dependencies — fully offline.
"""
from __future__ import annotations
import numpy as np
from sklearn.datasets import (
    load_diabetes, load_breast_cancer, load_wine, load_digits, load_iris,
    fetch_california_housing,
    make_regression, make_classification, make_friedman1, make_friedman2,
    make_friedman3, make_circles, make_moons,
)

# ---------------------------------------------------------------------------
# Helpers for synthetic dataset generation
# ---------------------------------------------------------------------------

def _make_xor(n_samples=2000, noise=0.15, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += rng.randn(n_samples, 2) * noise
    return X, y

def _make_checkerboard(n_samples=3000, grid=3, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.uniform(0, grid, size=(n_samples, 2))
    y = ((np.floor(X[:, 0]).astype(int) + np.floor(X[:, 1]).astype(int)) % 2)
    return X, y

def _make_spatial_regression(n_samples=3000, n_clusters=8, random_state=42):
    """Regression with spatial cluster structure (simulates lat/lon effects)."""
    rng = np.random.RandomState(random_state)
    centers = rng.uniform(-5, 5, size=(n_clusters, 2))
    labels = rng.randint(0, n_clusters, n_samples)
    X_spatial = centers[labels] + rng.randn(n_samples, 2) * 0.5
    X_extra = rng.randn(n_samples, 4)
    X = np.hstack([X_spatial, X_extra])
    cluster_effects = rng.randn(n_clusters) * 3
    y = cluster_effects[labels] + X[:, 2] * 0.5 + rng.randn(n_samples) * 0.3
    return X, y

def _make_heterogeneous_regression(n_samples=2000, random_state=42):
    """Regression with mixed continuous + integer-valued (simulated categorical) features."""
    rng = np.random.RandomState(random_state)
    X_cont = rng.randn(n_samples, 5)
    X_cat = rng.randint(0, 5, size=(n_samples, 3)).astype(float)
    X = np.hstack([X_cont, X_cat])
    cat_effects = np.array([[0.5, -0.3, 1.0, -0.8, 0.2],
                            [0.2, 0.7, -0.4, 0.1, 0.9],
                            [-0.6, 0.3, 0.8, -0.2, 0.4]])
    y = X_cont @ np.array([1.0, -0.5, 0.3, 0.8, -0.2])
    for j in range(3):
        y += cat_effects[j][X_cat[:, j].astype(int)]
    y += rng.randn(n_samples) * 0.3
    return X, y

def _make_heterogeneous_classification(n_samples=2000, random_state=42):
    """Binary classification with mixed continuous + integer-valued features."""
    rng = np.random.RandomState(random_state)
    X_cont = rng.randn(n_samples, 5)
    X_cat = rng.randint(0, 4, size=(n_samples, 3)).astype(float)
    X = np.hstack([X_cont, X_cat])
    logit = X_cont[:, 0] - 0.5 * X_cont[:, 1] + 0.3 * X_cont[:, 2]
    cat_effects = np.array([[0.3, -0.5, 0.8, -0.2],
                            [0.1, 0.4, -0.6, 0.3],
                            [-0.4, 0.2, 0.5, -0.1]])
    for j in range(3):
        logit += cat_effects[j][X_cat[:, j].astype(int)]
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.rand(n_samples) < prob).astype(int)
    return X, y

def _make_collinear_regression(n_samples=2000, n_features=10,
                                effective_rank=3, random_state=42):
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           n_informative=n_features, effective_rank=effective_rank,
                           noise=5.0, random_state=random_state)
    return X, y

def _make_interaction_regression(n_samples=3000, random_state=42):
    """Regression with explicit pairwise and triple interactions."""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, 6)
    y = (2.0 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2]
         + 3.0 * X[:, 0] * X[:, 1]
         + 1.5 * X[:, 2] * X[:, 3]
         + 2.0 * X[:, 0] * X[:, 1] * X[:, 4]
         + rng.randn(n_samples) * 0.5)
    return X, y


# ---------------------------------------------------------------------------
# Master dataset registry
# ---------------------------------------------------------------------------

DATASETS: list[dict] = [
    # =====================================================================
    # REGRESSION
    # =====================================================================

    # --- 1. reg_low_d_smooth ---
    {"name": "diabetes", "category": "reg_low_d_smooth", "task": "regression",
     "loader": lambda: load_diabetes(return_X_y=True)},
    {"name": "friedman1_500", "category": "reg_low_d_smooth", "task": "regression",
     "loader": lambda: make_friedman1(n_samples=500, n_features=5, random_state=42)},
    {"name": "friedman2_2k", "category": "reg_low_d_smooth", "task": "regression",
     "loader": lambda: make_friedman2(n_samples=2000, random_state=42)},
    {"name": "friedman3_2k", "category": "reg_low_d_smooth", "task": "regression",
     "loader": lambda: make_friedman3(n_samples=2000, random_state=42)},

    # --- 2. reg_high_d_sparse ---
    {"name": "sparse_50d", "category": "reg_high_d_sparse", "task": "regression",
     "loader": lambda: make_regression(n_samples=2000, n_features=50,
                                        n_informative=5, noise=10.0, random_state=42)},
    {"name": "sparse_100d", "category": "reg_high_d_sparse", "task": "regression",
     "loader": lambda: make_regression(n_samples=3000, n_features=100,
                                        n_informative=10, noise=10.0, random_state=42)},
    {"name": "sparse_200d", "category": "reg_high_d_sparse", "task": "regression",
     "loader": lambda: make_regression(n_samples=5000, n_features=200,
                                        n_informative=15, noise=10.0, random_state=42)},

    # --- 3. reg_noisy ---
    {"name": "noisy_10d", "category": "reg_noisy", "task": "regression",
     "loader": lambda: make_regression(n_samples=2000, n_features=10,
                                        n_informative=8, noise=50.0, random_state=42)},
    {"name": "friedman1_noisy", "category": "reg_noisy", "task": "regression",
     "loader": lambda: (lambda Xy: (Xy[0], Xy[1] + np.random.RandomState(42).randn(len(Xy[1])) * 5.0))(
         make_friedman1(n_samples=2000, n_features=10, random_state=42))},
    {"name": "noisy_sparse_30d", "category": "reg_noisy", "task": "regression",
     "loader": lambda: make_regression(n_samples=3000, n_features=30,
                                        n_informative=5, noise=100.0, random_state=42)},

    # --- 4. reg_nonlinear_interaction ---
    {"name": "friedman1_2k", "category": "reg_nonlinear_interaction", "task": "regression",
     "loader": lambda: make_friedman1(n_samples=2000, n_features=10, random_state=42)},
    {"name": "interaction_6d", "category": "reg_nonlinear_interaction", "task": "regression",
     "loader": lambda: _make_interaction_regression(n_samples=3000, random_state=42)},
    {"name": "friedman2_5k", "category": "reg_nonlinear_interaction", "task": "regression",
     "loader": lambda: make_friedman2(n_samples=5000, random_state=42)},

    # --- 5. reg_large_n ---
    {"name": "large_10d_50k", "category": "reg_large_n", "task": "regression",
     "loader": lambda: make_regression(n_samples=50000, n_features=10,
                                        n_informative=8, noise=5.0, random_state=42)},
    {"name": "large_20d_100k", "category": "reg_large_n", "task": "regression",
     "loader": lambda: make_regression(n_samples=100000, n_features=20,
                                        n_informative=15, noise=5.0, random_state=42)},
    {"name": "california_housing", "category": "reg_large_n", "task": "regression",
     "loader": lambda: fetch_california_housing(return_X_y=True)},

    # --- 6. reg_small_n ---
    {"name": "tiny_5d_100", "category": "reg_small_n", "task": "regression",
     "loader": lambda: make_regression(n_samples=100, n_features=5,
                                        n_informative=4, noise=5.0, random_state=42)},
    {"name": "tiny_10d_200", "category": "reg_small_n", "task": "regression",
     "loader": lambda: make_regression(n_samples=200, n_features=10,
                                        n_informative=7, noise=5.0, random_state=42)},
    {"name": "tiny_8d_300", "category": "reg_small_n", "task": "regression",
     "loader": lambda: make_regression(n_samples=300, n_features=8,
                                        n_informative=6, noise=5.0, random_state=42)},

    # --- 7. reg_collinear ---
    {"name": "collinear_10d_r3", "category": "reg_collinear", "task": "regression",
     "loader": lambda: _make_collinear_regression(2000, 10, 3, 42)},
    {"name": "collinear_30d_r5", "category": "reg_collinear", "task": "regression",
     "loader": lambda: _make_collinear_regression(3000, 30, 5, 42)},
    {"name": "collinear_50d_r8", "category": "reg_collinear", "task": "regression",
     "loader": lambda: _make_collinear_regression(5000, 50, 8, 42)},

    # --- 8. reg_spatial ---
    {"name": "spatial_6d_3k", "category": "reg_spatial", "task": "regression",
     "loader": lambda: _make_spatial_regression(3000, 8, 42)},
    {"name": "spatial_6d_8k", "category": "reg_spatial", "task": "regression",
     "loader": lambda: _make_spatial_regression(8000, 12, 42)},
    {"name": "spatial_6d_15k", "category": "reg_spatial", "task": "regression",
     "loader": lambda: _make_spatial_regression(15000, 20, 42)},

    # --- 9. reg_heterogeneous ---
    {"name": "hetero_reg_2k", "category": "reg_heterogeneous", "task": "regression",
     "loader": lambda: _make_heterogeneous_regression(2000, 42)},
    {"name": "hetero_reg_5k", "category": "reg_heterogeneous", "task": "regression",
     "loader": lambda: _make_heterogeneous_regression(5000, 42)},
    {"name": "hetero_reg_10k", "category": "reg_heterogeneous", "task": "regression",
     "loader": lambda: _make_heterogeneous_regression(10000, 42)},

    # =====================================================================
    # BINARY CLASSIFICATION
    # =====================================================================

    # --- 10. clf_binary_balanced ---
    {"name": "breast_cancer", "category": "clf_binary_balanced", "task": "binary",
     "loader": lambda: load_breast_cancer(return_X_y=True)},
    {"name": "balanced_10d_2k", "category": "clf_binary_balanced", "task": "binary",
     "loader": lambda: make_classification(n_samples=2000, n_features=10,
                                            n_informative=7, n_redundant=2,
                                            random_state=42)},
    {"name": "balanced_20d_5k", "category": "clf_binary_balanced", "task": "binary",
     "loader": lambda: make_classification(n_samples=5000, n_features=20,
                                            n_informative=12, n_redundant=4,
                                            random_state=42)},

    # --- 11. clf_binary_imbalanced ---
    {"name": "imbal_10d_90_10", "category": "clf_binary_imbalanced", "task": "binary",
     "loader": lambda: make_classification(n_samples=3000, n_features=10,
                                            n_informative=7, n_redundant=2,
                                            weights=[0.9, 0.1], random_state=42)},
    {"name": "imbal_20d_95_5", "category": "clf_binary_imbalanced", "task": "binary",
     "loader": lambda: make_classification(n_samples=5000, n_features=20,
                                            n_informative=12, n_redundant=4,
                                            weights=[0.95, 0.05], random_state=42)},
    {"name": "imbal_15d_99_1", "category": "clf_binary_imbalanced", "task": "binary",
     "loader": lambda: make_classification(n_samples=10000, n_features=15,
                                            n_informative=10, n_redundant=3,
                                            weights=[0.99, 0.01], random_state=42)},

    # --- 12. clf_binary_high_d ---
    {"name": "highd_50d_3k", "category": "clf_binary_high_d", "task": "binary",
     "loader": lambda: make_classification(n_samples=3000, n_features=50,
                                            n_informative=10, n_redundant=10,
                                            random_state=42)},
    {"name": "highd_100d_5k", "category": "clf_binary_high_d", "task": "binary",
     "loader": lambda: make_classification(n_samples=5000, n_features=100,
                                            n_informative=15, n_redundant=15,
                                            random_state=42)},
    {"name": "highd_200d_5k", "category": "clf_binary_high_d", "task": "binary",
     "loader": lambda: make_classification(n_samples=5000, n_features=200,
                                            n_informative=20, n_redundant=20,
                                            random_state=42)},

    # --- 13. clf_binary_noisy ---
    {"name": "noisy_flip5", "category": "clf_binary_noisy", "task": "binary",
     "loader": lambda: make_classification(n_samples=3000, n_features=10,
                                            n_informative=7, flip_y=0.05,
                                            random_state=42)},
    {"name": "noisy_flip10", "category": "clf_binary_noisy", "task": "binary",
     "loader": lambda: make_classification(n_samples=3000, n_features=10,
                                            n_informative=7, flip_y=0.10,
                                            random_state=42)},
    {"name": "noisy_flip20", "category": "clf_binary_noisy", "task": "binary",
     "loader": lambda: make_classification(n_samples=3000, n_features=10,
                                            n_informative=7, flip_y=0.20,
                                            random_state=42)},

    # --- 14. clf_binary_heterogeneous ---
    {"name": "hetero_clf_2k", "category": "clf_binary_heterogeneous", "task": "binary",
     "loader": lambda: _make_heterogeneous_classification(2000, 42)},
    {"name": "hetero_clf_5k", "category": "clf_binary_heterogeneous", "task": "binary",
     "loader": lambda: _make_heterogeneous_classification(5000, 42)},
    {"name": "hetero_clf_10k", "category": "clf_binary_heterogeneous", "task": "binary",
     "loader": lambda: _make_heterogeneous_classification(10000, 42)},

    # =====================================================================
    # MULTICLASS CLASSIFICATION
    # =====================================================================

    # --- 15. clf_multi_few (K=3-5) ---
    {"name": "iris", "category": "clf_multi_few", "task": "multiclass",
     "loader": lambda: load_iris(return_X_y=True)},
    {"name": "wine", "category": "clf_multi_few", "task": "multiclass",
     "loader": lambda: load_wine(return_X_y=True)},
    {"name": "multi_5class_3k", "category": "clf_multi_few", "task": "multiclass",
     "loader": lambda: make_classification(n_samples=3000, n_features=15,
                                            n_informative=10, n_classes=5,
                                            n_clusters_per_class=2,
                                            random_state=42)},
    {"name": "multi_4class_5k", "category": "clf_multi_few", "task": "multiclass",
     "loader": lambda: make_classification(n_samples=5000, n_features=12,
                                            n_informative=8, n_classes=4,
                                            n_clusters_per_class=2,
                                            random_state=42)},

    # --- 16. clf_multi_many (K>=10) ---
    {"name": "digits", "category": "clf_multi_many", "task": "multiclass",
     "loader": lambda: load_digits(return_X_y=True)},
    {"name": "multi_10class_5k", "category": "clf_multi_many", "task": "multiclass",
     "loader": lambda: make_classification(n_samples=5000, n_features=20,
                                            n_informative=15, n_classes=10,
                                            n_clusters_per_class=1,
                                            random_state=42)},
    {"name": "multi_15class_8k", "category": "clf_multi_many", "task": "multiclass",
     "loader": lambda: make_classification(n_samples=8000, n_features=25,
                                            n_informative=18, n_classes=15,
                                            n_clusters_per_class=1,
                                            random_state=42)},

    # --- 17. clf_multi_imbalanced ---
    {"name": "multi_imb_5class", "category": "clf_multi_imbalanced", "task": "multiclass",
     "loader": lambda: make_classification(n_samples=5000, n_features=15,
                                            n_informative=10, n_classes=5,
                                            n_clusters_per_class=1,
                                            weights=[0.4, 0.25, 0.15, 0.12, 0.08],
                                            random_state=42)},
    {"name": "multi_imb_8class", "category": "clf_multi_imbalanced", "task": "multiclass",
     "loader": lambda: make_classification(n_samples=8000, n_features=20,
                                            n_informative=14, n_classes=8,
                                            n_clusters_per_class=1,
                                            weights=[0.30, 0.20, 0.15, 0.10,
                                                     0.08, 0.07, 0.05, 0.05],
                                            random_state=42)},
    {"name": "multi_imb_10class", "category": "clf_multi_imbalanced", "task": "multiclass",
     "loader": lambda: make_classification(n_samples=10000, n_features=20,
                                            n_informative=15, n_classes=10,
                                            n_clusters_per_class=1,
                                            weights=[0.25, 0.15, 0.12, 0.10, 0.08,
                                                     0.07, 0.06, 0.06, 0.06, 0.05],
                                            random_state=42)},

    # --- 18. clf_large_n ---
    {"name": "large_binary_50k", "category": "clf_large_n", "task": "binary",
     "loader": lambda: make_classification(n_samples=50000, n_features=15,
                                            n_informative=10, n_redundant=3,
                                            random_state=42)},
    {"name": "large_multi_50k", "category": "clf_large_n", "task": "multiclass",
     "loader": lambda: make_classification(n_samples=50000, n_features=20,
                                            n_informative=15, n_classes=5,
                                            n_clusters_per_class=2,
                                            random_state=42)},
    {"name": "large_binary_100k", "category": "clf_large_n", "task": "binary",
     "loader": lambda: make_classification(n_samples=100000, n_features=20,
                                            n_informative=14, n_redundant=4,
                                            random_state=42)},

    # --- 19. clf_small_n ---
    {"name": "tiny_binary_100", "category": "clf_small_n", "task": "binary",
     "loader": lambda: make_classification(n_samples=100, n_features=5,
                                            n_informative=4, n_redundant=1,
                                            random_state=42)},
    {"name": "tiny_multi_150", "category": "clf_small_n", "task": "multiclass",
     "loader": lambda: make_classification(n_samples=150, n_features=8,
                                            n_informative=6, n_classes=3,
                                            n_clusters_per_class=1,
                                            random_state=42)},
    {"name": "tiny_binary_200", "category": "clf_small_n", "task": "binary",
     "loader": lambda: make_classification(n_samples=200, n_features=8,
                                            n_informative=6, n_redundant=2,
                                            random_state=42)},

    # --- 20. clf_adversarial ---
    {"name": "xor_2d", "category": "clf_adversarial", "task": "binary",
     "loader": lambda: _make_xor(2000, 0.15, 42)},
    {"name": "circles", "category": "clf_adversarial", "task": "binary",
     "loader": lambda: make_circles(n_samples=2000, noise=0.1, factor=0.4,
                                     random_state=42)},
    {"name": "moons", "category": "clf_adversarial", "task": "binary",
     "loader": lambda: make_moons(n_samples=2000, noise=0.15, random_state=42)},
    {"name": "checkerboard", "category": "clf_adversarial", "task": "binary",
     "loader": lambda: _make_checkerboard(3000, 3, 42)},
]


def get_datasets(category: str | None = None,
                 dataset_name: str | None = None) -> list[dict]:
    """Filter datasets by category and/or name."""
    result = DATASETS
    if category:
        result = [d for d in result if d["category"] == category]
    if dataset_name:
        result = [d for d in result if d["name"] == dataset_name]
    return result


def list_categories() -> list[str]:
    """Return sorted unique category names."""
    return sorted(set(d["category"] for d in DATASETS))
