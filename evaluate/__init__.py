# Evaluation module for RL-enhanced speculative decoding

__version__ = "0.1.0"

from pathlib import Path

EVALUATE_DIR = Path(__file__).parent
RESULTS_DIR = EVALUATE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

__all__ = ["benchmark_rl_inference", "analyze_results"]
