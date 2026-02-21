"""
Interpretability methods demo — GeoXGBRegressor.

Demonstrates:
  - feature_importances()
  - partition_feature_importances()
  - partition_trace()
  - partition_tree_rules()
  - sample_provenance()
  - noise_estimate()
"""
import numpy as np
from geoxgb import GeoXGBRegressor

rng = np.random.default_rng(42)
N, P = 400, 8
SIGNAL = [0, 1, 2]  # informative features
NOISE  = [3, 4, 5, 6, 7]

X = rng.standard_normal((N, P))
y = 3 * X[:, 0] - 2 * X[:, 1] + np.sin(X[:, 2]) + rng.standard_normal(N) * 0.1

feature_names = [f"signal_{i}" if i in SIGNAL else f"noise_{i}" for i in range(P)]

reg = GeoXGBRegressor(
    n_rounds=50,
    learning_rate=0.1,
    refit_interval=10,
    expand_ratio=0.1,
    auto_noise=True,
    random_state=42,
)
reg.fit(X, y)

# ── Boosting feature importances ────────────────────────────────────────────
print("=== Feature importances (boosting trees) ===")
for name, imp in reg.feature_importances(feature_names).items():
    bar = "█" * int(imp * 40)
    print(f"  {name:15s} {bar} {imp:.4f}")

# ── Partition feature importances ───────────────────────────────────────────
print("\n=== Partition feature importances (geometry) ===")
for entry in reg.partition_feature_importances(feature_names):
    print(f"  round={entry['round']:3d}: {entry['importances']}")

# ── Partition trace ──────────────────────────────────────────────────────────
print("\n=== Partition trace ===")
for entry in reg.partition_trace():
    print(
        f"  round={entry['round']:3d}  "
        f"noise_mod={entry['noise_modulation']:.3f}  "
        f"n_samples={entry['n_samples']}  "
        f"n_reduced={entry['n_reduced']}  "
        f"n_expanded={entry['n_expanded']}  "
        f"n_partitions={len(entry['partitions'])}"
    )

# ── Partition tree rules ─────────────────────────────────────────────────────
print("\n=== Partition tree rules (round 0) ===")
print(reg.partition_tree_rules(round_idx=0))

# ── Sample provenance ────────────────────────────────────────────────────────
print("=== Sample provenance ===")
prov = reg.sample_provenance()
for k, v in prov.items():
    print(f"  {k}: {v}")

# ── Noise estimate ───────────────────────────────────────────────────────────
print(f"\n=== Noise estimate ===\n  {reg.noise_estimate():.4f}")
print(f"\nn_trees={reg.n_trees}  n_resamples={reg.n_resamples}")
