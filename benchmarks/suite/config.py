"""Global configuration for the benchmarking suite."""
from pathlib import Path

# Seeds for reproducible train-test splits (HPO already does 3-fold CV internally)
SEEDS = [42]

# Cross-validation folds (default mode)
N_FOLDS = 3

# HPO settings
HPO_TRIALS = 50
HPO_CV = 3

# Tiered HPO budget by dataset size
def hpo_trials_for_n(n: int, base: int = HPO_TRIALS) -> int:
    if n >= 50_000:
        return max(base // 5, 10)
    if n >= 10_000:
        return max(base // 3, 15)
    if n >= 2_000:
        return max(base * 2 // 3, 30)
    return base

# Output directory — top-level project results folder
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model save directory (serialized models for inspection)
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
