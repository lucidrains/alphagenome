"""
Evaluation utilities for bactagenome
"""

from .metrics import BacterialMetrics, calculate_r2, calculate_auroc
from .visualization import plot_predictions, plot_training_curves

__all__ = [
    "BacterialMetrics",
    "calculate_r2",
    "calculate_auroc",
    "plot_predictions",
    "plot_training_curves",
]