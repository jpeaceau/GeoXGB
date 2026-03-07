"""Global configuration for the benchmarking suite."""
from pathlib import Path

# 5 seeds for reproducible CV / train-test splits
SEEDS = [42, 123, 456, 789, 1337]

# Cross-validation folds (default mode)
N_FOLDS = 5

# HPO settings
HPO_TRIALS_GEOXGB = 50
HPO_TRIALS_XGBOOST = 50
HPO_CV = 3

# Tiered HPO budget by dataset size
def hpo_trials_for_n(n: int, base: int = 50) -> int:
    if n >= 10_000:
        return max(base // 3, 15)
    if n >= 2_000:
        return max(base * 2 // 3, 30)
    return base

# Output directory
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model save directory (serialized models for inspection)
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
