"""
bactagenome: Bacterial Genome Modeling with AlphaGenome Architecture

A bacterial-specific adaptation of AlphaGenome for synthetic biology applications.
"""

__version__ = "0.1.0"
__author__ = "HKUST-GZ 2025 iGEM Team"

from .model.core import BactaGenome
from .model.config import BactaGenomeConfig

__all__ = ["BactaGenome", "BactaGenomeConfig"]