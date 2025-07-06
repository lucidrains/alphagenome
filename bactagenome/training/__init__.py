"""
Training utilities for BactaGenome
"""

from .losses import BacterialLossFunction
from .trainer import BactaGenomeTrainer

__all__ = [
    "BacterialLossFunction",
    "BactaGenomeTrainer"
]