# Ported from alphagenome_research reference implementation
# (reference/src/alphagenome_research/model/variant_scoring/)
"""Variant and interval scoring for PyTorch AlphaGenome.

This package exposes lightweight scoring utilities by default and lazily
imports optional scorers that require extra dependencies.
"""

from __future__ import annotations

from importlib import import_module

from .variant_scoring import (
    VariantScorer,
    ScoreVariantOutput,
    ScoreVariantResult,
    align_alternate,
    get_resolution,
    create_anndata,
)

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

_LAZY_ATTRS = {
    'GeneVariantScorer': ('.gene_mask', 'GeneVariantScorer'),
    'CenterMaskVariantScorer': ('.center_mask', 'CenterMaskVariantScorer'),
    'create_center_mask': ('.center_mask', 'create_center_mask'),
    'SpliceJunctionVariantScorer': ('.splice_junction', 'SpliceJunctionVariantScorer'),
    'ContactMapScorer': ('.contact_map', 'ContactMapScorer'),
    'PolyadenylationVariantScorer': ('.polyadenylation', 'PolyadenylationVariantScorer'),
    'PolyadenylationVariantMasks': ('.polyadenylation', 'PolyadenylationVariantMasks'),
    'IntervalScorer': ('.interval_scoring', 'IntervalScorer'),
    'GeneIntervalScorer': ('.interval_scoring', 'GeneIntervalScorer'),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        try:
            module = import_module(module_name, __name__)
        except Exception as exc:
            raise ImportError(
                f"Optional scoring dependency required for '{name}'. "
                "Install with: pip install 'alphagenome-pytorch[scoring]'"
            ) from exc
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))
