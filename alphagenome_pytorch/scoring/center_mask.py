# Ported from alphagenome_research reference implementation
# (reference/src/alphagenome_research/model/variant_scoring/center_mask.py)
"""Center mask variant scoring implementation for PyTorch.

This module provides center mask-based variant scoring, which creates a boolean
mask centered on a variant position and aggregates predictions within that mask.

The aggregation supports multiple modes for computing variant effect scores:
- DIFF_MEAN: Difference of means between ALT and REF
- ACTIVE_MEAN: Maximum of means (for activity-based scoring)
- DIFF_SUM: Difference of sums
- ACTIVE_SUM: Maximum of sums
- L2_DIFF: L2 norm of difference
- L2_DIFF_LOG1P: L2 norm of log1p-transformed difference
- DIFF_SUM_LOG2: Difference of log2-transformed sums
- DIFF_LOG2_SUM: Difference of log2(1 + sum) values
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TypeVar, Generic

import numpy as np
import torch


class AggregationType(enum.Enum):
    """Aggregation types for center mask variant scoring."""

    DIFF_MEAN = 'diff_mean'
    ACTIVE_MEAN = 'active_mean'
    DIFF_SUM = 'diff_sum'
    ACTIVE_SUM = 'active_sum'
    L2_DIFF = 'l2_diff'
    L2_DIFF_LOG1P = 'l2_diff_log1p'
    DIFF_SUM_LOG2 = 'diff_sum_log2'
    DIFF_LOG2_SUM = 'diff_log2_sum'


class OutputType(enum.Enum):
    """Output types for genomic predictions."""

    ATAC = 'atac'
    CAGE = 'cage'
    DNASE = 'dnase'
    RNA_SEQ = 'rna_seq'
    CHIP_HISTONE = 'chip_histone'
    CHIP_TF = 'chip_tf'
    SPLICE_SITES = 'splice_sites'
    SPLICE_SITE_USAGE = 'splice_site_usage'
    SPLICE_JUNCTIONS = 'splice_junctions'
    CONTACT_MAPS = 'contact_maps'
    PROCAP = 'procap'


@dataclass(frozen=True)
class Interval:
    """Genomic interval representation.

    Attributes:
        chrom: Chromosome name.
        start: Start position (0-based, inclusive).
        end: End position (0-based, exclusive).
        strand: Strand ('+', '-', or '.').
    """

    chrom: str
    start: int
    end: int
    strand: str = '.'

    @property
    def width(self) -> int:
        """Returns the width of the interval."""
        return self.end - self.start

    @property
    def negative_strand(self) -> bool:
        """Returns True if the interval is on the negative strand."""
        return self.strand == '-'

    def resize(self, width: int) -> 'Interval':
        """Returns a new interval centered on this interval with the given width."""
        center = (self.start + self.end) // 2
        new_start = center - width // 2
        new_end = new_start + width
        return Interval(self.chrom, new_start, new_end, self.strand)

    @classmethod
    def from_str(cls, s: str) -> 'Interval':
        """Parse an interval from string format 'chr:start-end:strand'."""
        parts = s.split(':')
        chrom = parts[0]
        pos_parts = parts[1].split('-')
        start = int(pos_parts[0])
        end = int(pos_parts[1])
        strand = parts[2] if len(parts) > 2 else '.'
        return cls(chrom, start, end, strand)


@dataclass(frozen=True)
class Variant:
    """Variant representation.

    Attributes:
        chrom: Chromosome name.
        position: 0-based position.
        reference_bases: Reference allele.
        alternate_bases: Alternate allele.
    """

    chrom: str
    position: int
    reference_bases: str
    alternate_bases: str

    @property
    def start(self) -> int:
        """Returns the start position of the variant."""
        return self.position

    @classmethod
    def from_str(cls, s: str) -> 'Variant':
        """Parse a variant from string format 'chr:pos:ref>alt'."""
        parts = s.split(':')
        chrom = parts[0]
        position = int(parts[1])
        alleles = parts[2].split('>')
        ref = alleles[0]
        alt = alleles[1]
        return cls(chrom, position, ref, alt)


@dataclass
class CenterMaskScorer:
    """Configuration for center mask variant scoring.

    Attributes:
        requested_output: The output type to score.
        aggregation_type: How to aggregate predictions within the mask.
        width: Width of the center mask in base pairs. If None, use full interval.
    """

    requested_output: OutputType
    aggregation_type: AggregationType
    width: Optional[int] = None


# Type aliases
ScoreVariantOutput = Dict[str, torch.Tensor | np.ndarray]
ScoreVariantResult = Dict[str, np.ndarray]


def get_resolution(output_type: OutputType) -> int:
    """Returns the resolution in base pairs for the given output type.

    Args:
        output_type: The output type.

    Returns:
        Resolution in base pairs (1, 128, or 2048).

    Raises:
        ValueError: If the output type is unknown.
    """
    if not isinstance(output_type, OutputType):
        output_name = getattr(output_type, 'name', None)
        if output_name is not None:
            try:
                output_type = OutputType[output_name]
            except KeyError:
                pass

    match output_type:
        case (
            OutputType.ATAC
            | OutputType.CAGE
            | OutputType.DNASE
            | OutputType.RNA_SEQ
            | OutputType.SPLICE_SITES
            | OutputType.SPLICE_SITE_USAGE
            | OutputType.SPLICE_JUNCTIONS
            | OutputType.PROCAP
        ):
            return 1
        case OutputType.CHIP_HISTONE | OutputType.CHIP_TF:
            return 128
        case OutputType.CONTACT_MAPS:
            return 2048
        case _:
            raise ValueError(f'Unknown output type: {output_type}.')


def create_center_mask(
    interval: Interval,
    variant: Variant,
    *,
    width: Optional[int],
    resolution: int,
) -> np.ndarray:
    """Creates a boolean mask centered on a variant for a given interval.

    The mask has shape [S, 1] where S is the sequence length at the target
    resolution. The mask is True within a window of `width` base pairs
    centered on the variant position.

    Args:
        interval: The genomic interval.
        variant: The variant to center the mask on.
        width: Width of the center mask in base pairs. If None, the entire
            interval is masked (if the variant is within the interval).
        resolution: Resolution in base pairs (1, 128, or 2048).

    Returns:
        Boolean mask of shape [S, 1] where S = interval.width // resolution.
    """
    seq_length = interval.width // resolution

    if width is None:
        # Full interval mask if variant is within interval
        if interval.start <= variant.start < interval.end:
            mask = np.ones([seq_length, 1], dtype=bool)
        else:
            mask = np.zeros([seq_length, 1], dtype=bool)
    else:
        target_resolution_width = math.ceil(width / resolution)

        # Determine the position of the variant in the specified resolution
        variant_start = getattr(variant, 'start', variant.position)
        base_resolution_center = variant_start - interval.start
        target_resolution_center = base_resolution_center // resolution

        # Compute start and end indices of the variant-centered mask
        target_resolution_start = max(
            target_resolution_center - target_resolution_width // 2, 0
        )
        target_resolution_end = min(
            (target_resolution_center - target_resolution_width // 2)
            + target_resolution_width,
            seq_length,
        )

        # If the variant is not within the interval, return an empty mask.
        # Otherwise, build the mask using target_resolution_start/end.
        mask = np.zeros([seq_length, 1], dtype=bool)
        if interval.start <= variant.start < interval.end:
            mask[target_resolution_start:target_resolution_end] = True

    return mask


def _apply_aggregation(
    ref: torch.Tensor,
    alt: torch.Tensor,
    masks: torch.Tensor | np.ndarray,
    aggregation_type: AggregationType,
) -> torch.Tensor:
    """Apply aggregation to ref and alt predictions using the given mask.

    Args:
        ref: Reference predictions of shape [S, T] where S is sequence length
            and T is number of tracks.
        alt: Alternate predictions of shape [S, T].
        masks: Boolean mask of shape [S, 1] or [S].
        aggregation_type: The aggregation type to apply.

    Returns:
        Aggregated scores of shape [T].

    Raises:
        ValueError: If the aggregation type is unknown.
    """
    # Convert mask to torch tensor if needed
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks).to(ref.device)

    # Squeeze the mask to 1D if it has shape [S, 1]
    mask_1d = masks.squeeze(-1) if masks.ndim == 2 else masks

    # Get masked values
    ref_masked = ref[mask_1d]  # Shape: [N, T] where N is number of True values
    alt_masked = alt[mask_1d]  # Shape: [N, T]

    match aggregation_type:
        case AggregationType.DIFF_MEAN:
            # alt.mean(where=mask) - ref.mean(where=mask)
            return alt_masked.mean(dim=0) - ref_masked.mean(dim=0)

        case AggregationType.ACTIVE_MEAN:
            # max(alt.mean(where=mask), ref.mean(where=mask))
            return torch.maximum(
                alt_masked.mean(dim=0),
                ref_masked.mean(dim=0),
            )

        case AggregationType.DIFF_SUM:
            # alt.sum(where=mask) - ref.sum(where=mask)
            return alt_masked.sum(dim=0) - ref_masked.sum(dim=0)

        case AggregationType.ACTIVE_SUM:
            # max(alt.sum(where=mask), ref.sum(where=mask))
            return torch.maximum(
                alt_masked.sum(dim=0),
                ref_masked.sum(dim=0),
            )

        case AggregationType.L2_DIFF:
            # sqrt(sum((alt - ref)^2, where=mask))
            diff_squared = (alt_masked - ref_masked) ** 2
            return torch.sqrt(diff_squared.sum(dim=0))

        case AggregationType.L2_DIFF_LOG1P:
            # sqrt(sum((log1p(alt) - log1p(ref))^2, where=mask))
            log_diff_squared = (torch.log1p(alt_masked) - torch.log1p(ref_masked)) ** 2
            return torch.sqrt(log_diff_squared.sum(dim=0))

        case AggregationType.DIFF_SUM_LOG2:
            # sum(log2(alt + 1), where=mask) - sum(log2(ref + 1), where=mask)
            return torch.log2(alt_masked + 1).sum(dim=0) - torch.log2(
                ref_masked + 1
            ).sum(dim=0)

        case AggregationType.DIFF_LOG2_SUM:
            # log2(1 + sum(alt, where=mask)) - log2(1 + sum(ref, where=mask))
            return torch.log2(1 + alt_masked.sum(dim=0)) - torch.log2(
                1 + ref_masked.sum(dim=0)
            )

        case _:
            raise ValueError(f'Unknown aggregation type: {aggregation_type}.')


class CenterMaskVariantScorer:
    """Variant scorer that aggregates ALT - REF in a window around the variant.

    This scorer creates a boolean mask centered on the variant position and
    computes aggregated scores within that mask. It supports multiple
    aggregation modes for different use cases (e.g., log fold change,
    maximum activity, L2 norm).

    Example:
        >>> scorer = CenterMaskVariantScorer()
        >>> settings = CenterMaskScorer(
        ...     requested_output=OutputType.ATAC,
        ...     aggregation_type=AggregationType.DIFF_MEAN,
        ...     width=501,
        ... )
        >>> mask, metadata = scorer.get_masks_and_metadata(
        ...     interval, variant, settings=settings, track_metadata={}
        ... )
        >>> scores = scorer.score_variant(ref_preds, alt_preds, masks=mask, settings=settings)
    """

    def get_masks_and_metadata(
        self,
        interval: Interval,
        variant: Variant,
        *,
        settings: CenterMaskScorer,
        track_metadata: Any,
    ) -> Tuple[np.ndarray, None]:
        """Create center mask and metadata for the given interval and variant.

        Args:
            interval: The genomic interval.
            variant: The variant to score.
            settings: Configuration for center mask scoring.
            track_metadata: Track metadata (unused for center mask scoring).

        Returns:
            Tuple of (mask, metadata) where mask has shape [S, 1] and metadata
            is None (center mask scoring does not require additional metadata).
        """
        del track_metadata  # Unused

        resolution = get_resolution(settings.requested_output)
        mask = create_center_mask(
            interval, variant, width=settings.width, resolution=resolution
        )

        return mask, None

    def score_variant(
        self,
        ref: Mapping[OutputType, torch.Tensor],
        alt: Mapping[OutputType, torch.Tensor],
        *,
        masks: torch.Tensor | np.ndarray,
        settings: CenterMaskScorer,
        variant: Optional[Variant] = None,
        interval: Optional[Interval] = None,
    ) -> ScoreVariantOutput:
        """Score a variant by aggregating predictions within the center mask.

        Args:
            ref: Reference predictions mapping output type to tensors of shape [S, T].
            alt: Alternate predictions mapping output type to tensors of shape [S, T].
            masks: Boolean mask of shape [S, 1].
            settings: Configuration for center mask scoring.
            variant: The variant being scored (unused).
            interval: The genomic interval (unused).

        Returns:
            Dictionary with 'score' key containing aggregated scores of shape [T].
        """
        del variant, interval  # Unused

        ref_tensor = ref[settings.requested_output]
        alt_tensor = alt[settings.requested_output]

        output = _apply_aggregation(
            ref_tensor, alt_tensor, masks, settings.aggregation_type
        )
        return {'score': output}

    def finalize_variant(
        self,
        scores: ScoreVariantResult,
        *,
        track_metadata: Any,
        mask_metadata: None,
        settings: CenterMaskScorer,
    ) -> Any:
        """Finalize variant scores into an AnnData object.

        This method creates an AnnData object from the aggregated scores,
        using the track metadata to annotate the variables (tracks).

        Args:
            scores: Dictionary with 'score' key containing shape [T] array.
            track_metadata: Metadata for the output tracks. Should be a dict-like
                object with a `get` method that returns track metadata for the
                requested output type.
            mask_metadata: Mask metadata (None for center mask scoring).
            settings: Configuration for center mask scoring.

        Returns:
            AnnData object with shape [1, num_tracks] containing the scores,
            with track metadata as variable annotations.

        Note:
            This method requires the `anndata` package to be installed.
            If not available, consider using the raw scores directly.
        """
        del mask_metadata  # Unused

        try:
            import anndata
            import pandas as pd
        except ImportError:
            raise ImportError(
                'anndata and pandas are required for finalize_variant. '
                'Install them with: pip install anndata pandas'
            )

        output_metadata = track_metadata.get(settings.requested_output)

        if output_metadata is None:
            # Create minimal metadata if not provided
            num_tracks = scores['score'].shape[-1]
            output_metadata = pd.DataFrame({
                'name': [f'track_{i}' for i in range(num_tracks)],
                'strand': ['.'] * num_tracks,
            })

        num_tracks = len(output_metadata)
        score_array = np.asarray(scores['score'])

        # Ensure score array has correct shape [1, num_tracks]
        score_matrix = score_array[np.newaxis, :num_tracks]

        # Create AnnData with proper index casting
        var = output_metadata.copy()
        var.index = var.index.map(str)

        return anndata.AnnData(
            np.ascontiguousarray(score_matrix),
            obs=None,
            var=var,
        )
