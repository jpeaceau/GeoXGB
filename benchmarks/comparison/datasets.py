"""
Real-world dataset loaders for GeoXGB comparison benchmark.
===========================================================
All datasets are returned with y normalised to (mean=0, std=1).
Features are NOT normalised — models handle their own scaling.

Datasets
--------
  california_housing  sklearn  n≈20k→cap  d=8
  diabetes            sklearn  n=442      d=10
  concrete            OpenML 4353  n=1030  d=8
  kin8nm              OpenML 189   n=8192  d=8
  abalone             OpenML 183   n=4177  d=8
  airfoil             OpenML 4549  n=1503  d=5
  wine_quality        OpenML 40691 n=1599  d=11
  cpusmall            OpenML 227   n=8192  d=12

OpenML datasets are fetched via sklearn.datasets.fetch_openml (cached to
~/scikit_learn_data/); first run requires internet access.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

# ── Cap ─────────────────────────────────────────────────────────────────────

MAX_N: int = 5_000   # subsample ceiling (balances coverage vs. GeoXGB runtime)


def _cap(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(y) > MAX_N:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), MAX_N, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def _normalise_y(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, std = float(y.mean()), float(y.std())
    y_norm = (y - mu) / (std + 1e-10)
    return X.astype(np.float64), y_norm.astype(np.float64)


# ── Loaders ─────────────────────────────────────────────────────────────────

def _load_california() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_california_housing
    ds = fetch_california_housing()
    return ds.data, ds.target


def _load_diabetes() -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import load_diabetes
    ds = load_diabetes()
    return ds.data, ds.target


def _load_openml(data_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an OpenML dataset by numeric ID.

    Robustness handling:
    - Categorical features are one-hot encoded via pd.get_dummies.
    - Categorical / string targets are cast to float.
    - Multi-target datasets use the first (default) target only.
    - Datasets with no default target use their last column as target.
    """
    import pandas as pd
    from sklearn.datasets import fetch_openml
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = fetch_openml(data_id=data_id, as_frame=True, parser="auto")

    # ── Target ────────────────────────────────────────────────────────────────
    if ds.target is not None and len(ds.target_names) > 0:
        y_raw = ds.target
        df    = ds.data
    else:
        # No default target: last column is the target
        df    = ds.data.iloc[:, :-1]
        y_raw = ds.data.iloc[:, -1]

    # Convert category / object / string target to float
    if hasattr(y_raw, "cat"):              # pandas Categorical
        y_raw = y_raw.astype(str)
    y = np.asarray(y_raw, dtype=float)

    # Handle 2-D y (multi-target) — take first column
    if y.ndim > 1:
        y = y[:, 0]

    # ── Features: handle categorical columns ─────────────────────────────────
    # Low-cardinality (<=10 unique) → one-hot encode; otherwise → cast to float
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    ohe_cols  = [c for c in cat_cols if df[c].nunique() <= 10]
    cont_cats = [c for c in cat_cols if df[c].nunique() > 10]

    if cont_cats:
        df[cont_cats] = df[cont_cats].astype(str).astype(float)
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

    X = df.to_numpy(dtype=float)

    return X, y


# ── Registry ─────────────────────────────────────────────────────────────────
#
#   Each entry: (loader_callable, short_description)
#
REGISTRY: dict[str, tuple[Callable, str]] = {
    "california_housing": (
        _load_california,
        f"sklearn     n~20k->{MAX_N}  d=8   house prices",
    ),
    "diabetes": (
        _load_diabetes,
        "sklearn     n=442   d=10  disease progression",
    ),
    "concrete": (
        lambda: _load_openml(4353),
        "OpenML 4353  n=1030  d=8   compressive strength",
    ),
    "kin8nm": (
        lambda: _load_openml(189),
        f"OpenML 189   n=8192->{MAX_N}  d=8   robot arm kinematics",
    ),
    "abalone": (
        lambda: _load_openml(183),
        "OpenML 183   n=4177  d=10  shell rings (Sex one-hot)",
    ),
    "energy_efficiency": (
        lambda: _load_openml(1472),
        "OpenML 1472  n=768   d=9   building heating load",
    ),
    "wine_quality": (
        lambda: _load_openml(40691),
        "OpenML 40691 n=1599  d=11  red wine physicochemistry",
    ),
    "cpusmall": (
        lambda: _load_openml(227),
        f"OpenML 227   n=8192->{MAX_N}  d=12  CPU activity",
    ),
}


def load_datasets(
    verbose: bool = True,
    cap_seed: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """
    Load all registered datasets.

    Returns
    -------
    dict mapping name → (X, y, description)
        X  : float64 array, shape (n, d)
        y  : float64 array, shape (n,) — z-score normalised
        desc : short human-readable description

    Datasets that fail to load are silently skipped with a warning.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}

    for name, (loader, desc) in REGISTRY.items():
        try:
            X_raw, y_raw = loader()
            X, y = _cap(X_raw, y_raw, seed=cap_seed)
            X, y = _normalise_y(X, y)
            out[name] = (X, y, desc)
            if verbose:
                print(f"  loaded  {name:<22}  n={len(y):5d}  d={X.shape[1]:2d}  {desc}")
        except Exception as exc:          # noqa: BLE001
            if verbose:
                print(f"  SKIP    {name:<22}  ({type(exc).__name__}: {exc})")

    return out


if __name__ == "__main__":
    print("Loading datasets …\n")
    ds = load_datasets(verbose=True)
    print(f"\n{len(ds)} datasets loaded successfully.")
