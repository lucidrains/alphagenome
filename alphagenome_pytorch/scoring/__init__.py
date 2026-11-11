# Ported from alphagenome_research reference implementation
# (reference/src/alphagenome_research/model/variant_scoring/)
"""Variant and interval scoring for PyTorch AlphaGenome."""

from __future__ import annotations

from importlib import import_module

from .variant_scoring import (
    DirectGeneExpressionScorer,
    ScoreVariantOutput,
    ScoreVariantResult,
    VariantScorer,
    align_alternate,
    create_anndata,
    get_resolution,
)

__all__ = [
    'VariantScorer',
    'DirectGeneExpressionScorer',
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

# optional scorers, pulled in on first access

_optional = {
    'GeneVariantScorer': '.gene_mask',
    'CenterMaskVariantScorer': '.center_mask',
    'create_center_mask': '.center_mask',
    'SpliceJunctionVariantScorer': '.splice_junction',
    'ContactMapScorer': '.contact_map',
    'PolyadenylationVariantScorer': '.polyadenylation',
    'PolyadenylationVariantMasks': '.polyadenylation',
    'IntervalScorer': '.interval_scoring',
    'GeneIntervalScorer': '.interval_scoring',
}


def __getattr__(name: str):
    if name not in _optional:
        raise AttributeError(f"module '{__name__}' has no attribute {name!r}")

    return getattr(import_module(_optional[name], __name__), name)
