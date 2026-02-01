# Ported from alphagenome_research reference implementation
# (reference/src/alphagenome_research/model/variant_scoring/)
"""Variant and interval scoring for PyTorch AlphaGenome.

Ported from alphagenome_research reference implementation.
"""

from .variant_scoring import (
    VariantScorer,
    ScoreVariantOutput,
    ScoreVariantResult,
    align_alternate,
    get_resolution,
    create_anndata,
)
from .gene_mask import GeneVariantScorer
from .center_mask import CenterMaskVariantScorer, create_center_mask
from .splice_junction import SpliceJunctionVariantScorer
from .contact_map import ContactMapScorer
from .polyadenylation import PolyadenylationVariantScorer, PolyadenylationVariantMasks
from .interval_scoring import IntervalScorer, GeneIntervalScorer

__all__ = [
    'VariantScorer',
    'ScoreVariantOutput',
    'ScoreVariantResult',
    'align_alternate',
    'get_resolution',
    'create_anndata',
    'GeneVariantScorer',
    'CenterMaskVariantScorer',
    'create_center_mask',
    'SpliceJunctionVariantScorer',
    'ContactMapScorer',
    'PolyadenylationVariantScorer',
    'PolyadenylationVariantMasks',
    'IntervalScorer',
    'GeneIntervalScorer',
]
