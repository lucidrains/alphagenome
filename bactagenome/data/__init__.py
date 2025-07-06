"""
Data loading and preprocessing for BactaGenome
"""

from .dummy import DummyBacterialDataset, DummyBacterialTargetsDataset

__all__ = [
    "DummyBacterialDataset",
    "DummyBacterialTargetsDataset"
]