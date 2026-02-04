# Ported from alphagenome_research reference implementation
# (reference/src/alphagenome_research/model/variant_scoring/)
"""Base class for variant scorers.

This module provides the abstract base class for variant scoring strategies
and utility functions for variant effect prediction.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

# Type variables for generic variant scorer
VariantMaskT = TypeVar('VariantMaskT')
VariantMetadataT = TypeVar('VariantMetadataT')
VariantSettingsT = TypeVar('VariantSettingsT')

# Type aliases for score outputs
ScoreVariantOutput = Mapping[str, Union[Tensor, np.ndarray]]
ScoreVariantResult = Mapping[str, np.ndarray]


class OutputType(Enum):
    """Output types for genomic predictions.

    Each output type has an associated resolution (number of base pairs per bin).
    """

    ATAC = auto()
    CAGE = auto()
    DNASE = auto()
    RNA_SEQ = auto()
    CHIP_HISTONE = auto()
    CHIP_TF = auto()
    SPLICE_SITES = auto()
    SPLICE_SITE_USAGE = auto()
    SPLICE_JUNCTIONS = auto()
    CONTACT_MAPS = auto()
    PROCAP = auto()


if TYPE_CHECKING:
    import anndata
    import pandas as pd

# Type alias for output metadata (mapping from OutputType to track metadata)
OutputMetadata = Mapping[OutputType, 'pd.DataFrame']


@dataclass
class Interval:
    """Represents a genomic interval.

    Attributes:
        chromosome: Chromosome name (e.g., 'chr1').
        start: 0-based start position (inclusive).
        end: 0-based end position (exclusive).
        strand: Strand ('+', '-', or '.').
    """

    chromosome: str
    start: int
    end: int
    strand: str = '+'

    @property
    def width(self) -> int:
        """Returns the width of the interval in base pairs."""
        return self.end - self.start

    @property
    def negative_strand(self) -> bool:
        """Returns True if the interval is on the negative strand."""
        return self.strand == '-'

    def resize(self, width: int) -> 'Interval':
        """Returns a new interval centered on this one with the given width."""
        center = (self.start + self.end) // 2
        new_start = center - width // 2
        new_end = new_start + width
        return Interval(self.chromosome, new_start, new_end, self.strand)

    def boundary_shift(
        self, start_offset: int = 0, end_offset: int = 0, use_strand: bool = True
    ) -> 'Interval':
        """Returns a new interval with shifted boundaries."""
        if use_strand and self.negative_strand:
            return Interval(
                self.chromosome,
                self.start - end_offset,
                self.end - start_offset,
                self.strand,
            )
        return Interval(
            self.chromosome,
            self.start + start_offset,
            self.end + end_offset,
            self.strand,
        )


@dataclass
class Variant:
    """Represents a genomic variant.

    Attributes:
        chromosome: Chromosome name.
        position: 1-based position of the variant.
        reference_bases: Reference allele sequence.
        alternate_bases: Alternate allele sequence.
    """

    chromosome: str
    position: int
    reference_bases: str
    alternate_bases: str

    @property
    def start(self) -> int:
        """Returns 0-based start position."""
        return self.position - 1

    @property
    def end(self) -> int:
        """Returns 0-based end position (exclusive)."""
        return self.start + len(self.reference_bases)

    def reference_overlaps(self, interval: Interval) -> bool:
        """Returns True if reference allele overlaps the interval."""
        return self.start < interval.end and self.end > interval.start

    def alternate_overlaps(self, interval: Interval) -> bool:
        """Returns True if alternate allele overlaps the interval."""
        alt_end = self.start + len(self.alternate_bases)
        return self.start < interval.end and alt_end > interval.start

    def split(self, position: int) -> tuple['Variant | None', 'Variant | None']:
        """Splits the variant at the given position.

        Returns:
            Tuple of (left_variant, right_variant).
        """
        if position <= self.start:
            return None, self
        if position >= self.end:
            return self, None

        # Split the reference and alternate at the relative position
        rel_pos = position - self.start
        left_ref = self.reference_bases[:rel_pos]
        right_ref = self.reference_bases[rel_pos:]

        # For alternate, we need to handle insertions/deletions carefully
        # This is a simplified split - in practice, the reference implementation
        # handles this more carefully
        left_alt = self.alternate_bases[:rel_pos] if rel_pos <= len(self.alternate_bases) else self.alternate_bases
        right_alt = self.alternate_bases[rel_pos:] if rel_pos < len(self.alternate_bases) else ''

        left = Variant(self.chromosome, self.position, left_ref, left_alt)
        right = Variant(self.chromosome, position + 1, right_ref, right_alt)

        return left, right


def create_anndata(
    scores: np.ndarray,
    *,
    obs: 'pd.DataFrame | None',
    var: 'pd.DataFrame',
) -> 'anndata.AnnData':
    """Helper function for creating AnnData objects.

    Args:
        scores: Score matrix of shape [num_genes, num_tracks].
        obs: Observation metadata (row metadata, e.g., gene information).
        var: Variable metadata (column metadata, e.g., track information).

    Returns:
        AnnData object with the scores and metadata.
    """
    try:
        import anndata
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            'anndata and pandas are required to create AnnData outputs. '
            "Install them with: pip install 'alphagenome-pytorch[scoring]'"
        ) from exc

    var = var.copy()
    # Explicitly cast dataframe indices to str to avoid
    # ImplicitModificationWarning being logged over and over again.
    var.index = var.index.map(str)

    if obs is not None:
        obs = obs.copy()
        obs.index = obs.index.map(str).astype(str)

    return anndata.AnnData(np.ascontiguousarray(scores), obs=obs, var=var)


def get_resolution(output_type: OutputType) -> int:
    """Returns the resolution (bp per bin) for the given output type.

    Args:
        output_type: The output type to get resolution for.

    Returns:
        Resolution in base pairs.

    Raises:
        ValueError: If the output type is unknown.
    """
    match output_type:
        case OutputType.ATAC:
            return 1
        case OutputType.CAGE:
            return 1
        case OutputType.DNASE:
            return 1
        case OutputType.RNA_SEQ:
            return 1
        case OutputType.CHIP_HISTONE:
            return 128
        case OutputType.CHIP_TF:
            return 128
        case OutputType.SPLICE_SITES:
            return 1
        case OutputType.SPLICE_SITE_USAGE:
            return 1
        case OutputType.SPLICE_JUNCTIONS:
            return 1
        case OutputType.CONTACT_MAPS:
            return 2048
        case OutputType.PROCAP:
            return 1
        case _:
            raise ValueError(f'Unknown output type: {output_type}.')


class VariantScorer(
    Generic[VariantMaskT, VariantMetadataT, VariantSettingsT],
    metaclass=abc.ABCMeta,
):
    """Abstract class for variant scorers.

    A variant scorer computes effect scores for genomic variants by comparing
    predictions from reference and alternate alleles.

    Type Parameters:
        VariantMaskT: Type of masks used for scoring.
        VariantMetadataT: Type of metadata returned alongside masks.
        VariantSettingsT: Type of settings/configuration for the scorer.
    """

    @abc.abstractmethod
    def get_masks_and_metadata(
        self,
        interval: Interval,
        variant: Variant,
        *,
        settings: VariantSettingsT,
        track_metadata: OutputMetadata,
    ) -> tuple[VariantMaskT, VariantMetadataT]:
        """Returns masks and metadata for the given interval, variant and metadata.

        The generated masks and metadata will be passed to `score_variant` and
        `finalize_variant` respectively.

        Args:
            interval: The interval to score.
            variant: The variant to extract the masks/metadata for.
            settings: The variant scorer settings.
            track_metadata: The track metadata required to finalize the variant.

        Returns:
            A tuple of (masks, metadata), where:
                masks: The masks required to score the variant, such as gene or TSS
                    or strand masks. These will be passed into the `score_variants`
                    function.
                metadata: The metadata required to finalize the variant. These will
                    be passed into the `finalize_variants` function.

            The formats/shapes of masks and metadata will vary across variant
            scorers depending on their individual needs.
        """

    @abc.abstractmethod
    def score_variant(
        self,
        ref: Mapping[OutputType, 'Tensor'],
        alt: Mapping[OutputType, 'Tensor'],
        *,
        masks: VariantMaskT,
        settings: VariantSettingsT,
        variant: Variant | None = None,
        interval: Interval | None = None,
    ) -> ScoreVariantOutput:
        """Generates a score per track for the provided ref/alt predictions.

        Args:
            ref: Reference predictions, mapping output types to tensors.
            alt: Alternative predictions, mapping output types to tensors.
            masks: The masks for scoring the variant.
            settings: The variant scorer settings.
            variant: The variant to score.
            interval: The interval to score.

        Returns:
            Dictionary of scores to be passed to `finalize_variant`.
        """

    @abc.abstractmethod
    def finalize_variant(
        self,
        scores: ScoreVariantResult,
        *,
        track_metadata: OutputMetadata,
        mask_metadata: VariantMetadataT,
        settings: VariantSettingsT,
    ) -> anndata.AnnData:
        """Returns finalized scores for the given scores and metadata.

        Args:
            scores: Dictionary of scores generated from `score_variant` function.
            track_metadata: Metadata describing the tracks for each output_type.
            mask_metadata: Metadata describing the masks.
            settings: The variant scorer settings.

        Returns:
            An AnnData object containing the final variant outputs. The entries
            will vary across scorers depending on their individual needs.
        """


def align_alternate(
    alt: 'Tensor | np.ndarray',
    variant: Variant,
    interval: Interval,
) -> 'Tensor':
    """Aligns ALT predictions to match the REF allele's sequence length.

    This function adjusts the `alt` prediction array to account for indels
    (insertions or deletions) present in the `variant`.

    For insertions, the function summarizes the inserted region by taking the
    maximum value across the alternate bases and pads the end with zeros to
    maintain the original sequence length.

    For deletions, zero signal is inserted at the locations corresponding to
    the deleted bases in the reference.

    Args:
        alt: The ALT allele predictions, shape [sequence_length, num_tracks].
        variant: The variant containing the indel information.
        interval: The genomic interval.

    Returns:
        The aligned ALT predictions, shape [sequence_length, num_tracks].
    """
    # Convert to torch tensor if needed
    if isinstance(alt, np.ndarray):
        alt = torch.from_numpy(alt)

    insertion_length = len(variant.alternate_bases) - len(variant.reference_bases)
    deletion_length = -insertion_length
    variant_start_in_vector = variant.start - interval.start

    # We assume that variants are left-aligned, and that insertions/deletions
    # for multi-change variants occur at the end of the variant.
    # We only need to align that insertion/deletion portion.
    variant_start_in_vector += (
        min(len(variant.reference_bases), len(variant.alternate_bases)) - 1
    )
    original_length = alt.shape[0]

    if insertion_length > 0:
        # Summarize potential insertions by computing the maximum score across
        # alternate bases.
        insert_start = variant_start_in_vector
        insert_end = variant_start_in_vector + insertion_length + 1

        pool_alt_past_ref = alt[insert_start:insert_end].max(dim=0, keepdim=True).values

        alt = torch.cat(
            [
                alt[:variant_start_in_vector],
                pool_alt_past_ref,
                alt[(variant_start_in_vector + insertion_length + 1):],
                torch.zeros((insertion_length, alt.shape[1]), dtype=alt.dtype, device=alt.device),
            ],
            dim=0,
        )
        # Truncate to the original sequence length in case the alt insertion
        # spills over the original sequence length. This happens only for
        # insertions longer than half the interval.
        alt = alt[:original_length]

    elif deletion_length > 0:
        # Handle potential deletions by inserting zero signal at deletion
        # locations.
        alt = torch.cat(
            [
                alt[:(variant_start_in_vector + 1)],
                torch.zeros((deletion_length, alt.shape[1]), dtype=alt.dtype, device=alt.device),
                alt[(variant_start_in_vector + 1):],
            ],
            dim=0,
        )
        alt = alt[:original_length]

    return alt
