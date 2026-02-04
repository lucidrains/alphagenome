# Ported from alphagenome_research reference implementation
# (reference/src/alphagenome_research/model/variant_scoring/)
"""Implementation of gene mask variant scoring.

This module provides variant scoring that aggregates predictions across gene
regions, supporting different scoring strategies like log fold change (LFC),
activity scoring, and splicing effect scoring.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol

import anndata
import numpy as np
import pandas as pd
import torch

from .variant_scoring import (
    Interval,
    OutputMetadata,
    OutputType,
    ScoreVariantOutput,
    ScoreVariantResult,
    Variant,
    VariantScorer,
    align_alternate,
    create_anndata,
    get_resolution,
)

if TYPE_CHECKING:
    from torch import Tensor


class BaseVariantScorer(Enum):
    """Base variant scoring strategies for gene mask scoring."""

    GENE_MASK_LFC = auto()
    GENE_MASK_ACTIVE = auto()
    GENE_MASK_SPLICING = auto()


@dataclass
class GeneMaskSettings:
    """Settings for gene mask variant scoring.

    Attributes:
        base_variant_scorer: The scoring strategy to use.
        requested_output: The output type to score.
    """

    base_variant_scorer: BaseVariantScorer
    requested_output: OutputType


class GeneMaskExtractor(Protocol):
    """Protocol for gene mask extractors.

    Implementations should provide a method to extract boolean gene masks
    and associated metadata for a given genomic interval.
    """

    def extract(
        self, interval: Interval, variant: Variant | None = None
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Extracts gene masks and metadata for an interval.

        Args:
            interval: The genomic interval.
            variant: Optional variant for variant-aware extraction.

        Returns:
            Tuple of (gene_mask, metadata) where gene_mask is a boolean array
            of shape [sequence_length, num_genes] and metadata is a DataFrame
            with gene information including 'Strand' column.
        """
        ...


def _score_gene_variant(
    ref: 'Tensor',
    alt: 'Tensor',
    gene_mask: 'Tensor',
    *,
    settings: GeneMaskSettings,
) -> 'Tensor':
    """Scores a variant using gene masks.

    Args:
        ref: Reference predictions, shape [S, T].
        alt: Alternate predictions, shape [S, T].
        gene_mask: Boolean gene mask, shape [S, G].
        settings: The scoring settings.

    Returns:
        Scores of shape [G, T].
    """
    match settings.base_variant_scorer:
        case BaseVariantScorer.GENE_MASK_LFC:
            # Scores are the log fold change between the mean prediction of REF
            # and ALT within each gene mask.
            gene_mask_float = gene_mask.float()
            gene_mask_sum = gene_mask_float.sum(dim=0).unsqueeze(-1)  # [G, 1]
            ref_mean = torch.einsum('lt,lg->gt', ref, gene_mask_float) / gene_mask_sum
            alt_mean = torch.einsum('lt,lg->gt', alt, gene_mask_float) / gene_mask_sum
            return torch.log(alt_mean + 1e-3) - torch.log(ref_mean + 1e-3)

        case BaseVariantScorer.GENE_MASK_ACTIVE:
            # Scores are the maximum of the mean prediction for REF and ALT
            # within each gene mask.
            gene_mask_float = gene_mask.float()
            gene_mask_sum = gene_mask_float.sum(dim=0).unsqueeze(-1)  # [G, 1]
            ref_score = torch.einsum('lt,lg->gt', ref, gene_mask_float) / gene_mask_sum
            alt_score = torch.einsum('lt,lg->gt', alt, gene_mask_float) / gene_mask_sum
            return torch.maximum(alt_score, ref_score)

        case BaseVariantScorer.GENE_MASK_SPLICING:
            # Scores are the maximum of the absolute difference between REF and
            # ALT within each gene mask.
            # Process each gene separately to reduce memory footprint.
            diff = torch.abs(alt - ref)  # [S, T]
            num_genes = gene_mask.shape[1]
            results = []
            for g in range(num_genes):
                mask = gene_mask[:, g].unsqueeze(-1)  # [S, 1]
                masked_diff = diff * mask.float()  # [S, T]
                max_score = masked_diff.max(dim=0).values  # [T]
                results.append(max_score)
            return torch.stack(results, dim=0)  # [G, T]

        case _:
            raise ValueError(
                f'Unsupported base variant scorer: {settings.base_variant_scorer}.'
            )


class GeneVariantScorer(VariantScorer[np.ndarray, pd.DataFrame, GeneMaskSettings]):
    """Variant scorer that computes scores for different genes.

    This scorer aggregates variant effects within gene regions using different
    strategies (LFC, activity, or splicing scoring). It requires a gene mask
    extractor that provides boolean masks indicating which positions belong to
    which genes.
    """

    def __init__(self, gene_mask_extractor: GeneMaskExtractor):
        """Initializes the GeneVariantScorer.

        Args:
            gene_mask_extractor: Gene mask extractor to use for extracting
                gene regions from genomic annotations.
        """
        self._gene_mask_extractor = gene_mask_extractor

    def get_masks_and_metadata(
        self,
        interval: Interval,
        variant: Variant,
        *,
        settings: GeneMaskSettings,
        track_metadata: OutputMetadata,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Get gene masks and metadata for the given interval and variant.

        Note that the gene mask returned for the REF allele is just the normal
        gene mask extracted from the GTF file from the interval. However, the
        gene mask can be different for the ALT allele in the case of indels. We
        handle this by using the align_alternate function to align the ALT
        predictions to the REF coordinate space, so that the same gene mask can
        be applied to both.

        Args:
            interval: Genomic interval to extract gene masks for.
            variant: Variant that may alter the gene mask in the case of the
                ALT allele.
            settings: The variant scorer settings.
            track_metadata: Track metadata for the variant (unused).

        Returns:
            Tuple of (gene_mask, mask_metadata) where:
                - gene_mask: Boolean array of shape [S, G] indicating which
                  positions belong to which genes
                - mask_metadata: DataFrame with gene information including
                  'Strand' column

        Raises:
            ValueError: If the resolution is not 1 (required for gene scoring).
        """
        del track_metadata  # Unused
        if get_resolution(settings.requested_output) != 1:
            raise ValueError(
                'Only resolution = 1 is supported for gene variant scoring.'
            )
        gene_mask, metadata = self._gene_mask_extractor.extract(interval, variant)
        return gene_mask, metadata

    def score_variant(
        self,
        ref: Mapping[OutputType, 'Tensor'],
        alt: Mapping[OutputType, 'Tensor'],
        *,
        masks: np.ndarray,
        settings: GeneMaskSettings,
        variant: Variant | None = None,
        interval: Interval | None = None,
    ) -> ScoreVariantOutput:
        """Scores the variant using gene masks.

        Args:
            ref: Reference predictions by output type.
            alt: Alternate predictions by output type.
            masks: Gene mask array, shape [S, G].
            settings: Scoring settings.
            variant: The variant being scored (used for indel alignment).
            interval: The genomic interval (used for indel alignment).

        Returns:
            Dictionary with 'score' key containing scores of shape [G, T].
        """
        alt_tensor = alt[settings.requested_output]
        ref_tensor = ref[settings.requested_output]
        gene_mask = torch.from_numpy(masks).to(alt_tensor.device)

        # Align alternate predictions for indels
        if variant is not None and interval is not None:
            alt_tensor = align_alternate(alt_tensor, variant, interval)

        output = _score_gene_variant(ref_tensor, alt_tensor, gene_mask, settings=settings)
        return {'score': output}

    def finalize_variant(
        self,
        scores: ScoreVariantResult,
        *,
        track_metadata: OutputMetadata,
        mask_metadata: pd.DataFrame,
        settings: GeneMaskSettings,
    ) -> anndata.AnnData:
        """Returns summarized scores for the given scores and metadata.

        Applies strand matching to ensure gene-track strand compatibility.
        Scores for mismatched strands are set to NaN.

        Args:
            scores: Dictionary of scores from score_variant.
            track_metadata: Track metadata by output type.
            mask_metadata: Gene metadata DataFrame with 'Strand' column.
            settings: Scoring settings.

        Returns:
            AnnData object with scores [G, T] and gene/track metadata.
        """
        output_metadata = track_metadata.get(settings.requested_output)

        # Create strand mask to filter scores by matching strand
        # Keep scores where gene strand matches track strand, or where
        # track strand is unstranded ('.')
        strand_mask = (
            np.asarray(mask_metadata['Strand'].values)[:, None]
            == output_metadata['strand'].values[None]
        ) | (output_metadata['strand'].values[None] == '.')

        # Convert scores to numpy if needed
        score_array = scores['score']
        if isinstance(score_array, torch.Tensor):
            score_array = score_array.cpu().numpy()

        # Apply strand mask, setting mismatched entries to NaN
        scores_masked = np.where(strand_mask, score_array, np.nan)

        return create_anndata(
            scores_masked,
            obs=mask_metadata,
            var=output_metadata,
        )
