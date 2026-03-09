"""Global configuration for the benchmarking suite."""
from pathlib import Path

# Seeds for reproducible train-test splits (HPO already does 3-fold CV internally)
SEEDS = [42]

# Cross-validation folds (default mode)
N_FOLDS = 3

# HPO settings
HPO_TRIALS_GEOXGB = 30
HPO_TRIALS_XGBOOST = 30
HPO_CV = 3

# Tiered HPO budget by dataset size
def hpo_trials_for_n(n: int, base: int = 30) -> int:
    if n >= 50_000:
        return 8
    if n >= 10_000:
        return max(base // 3, 10)
    if n >= 2_000:
        return max(base * 2 // 3, 20)
    return base

# Output directory
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model save directory (serialized models for inspection)
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
