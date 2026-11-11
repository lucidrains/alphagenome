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
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import torch
from torch import Tensor

VariantMaskT = TypeVar('VariantMaskT')
VariantMetadataT = TypeVar('VariantMetadataT')
VariantSettingsT = TypeVar('VariantSettingsT')

ScoreVariantOutput = Mapping[str, Tensor | np.ndarray]
ScoreVariantResult = Mapping[str, np.ndarray]


def exists(v):
    return v is not None


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
    SC_ATAC = auto()
    METHYLATION = auto()
    GENE_EXPRESSION = auto()
    SINGLE_CELL_GENE_EXPRESSION = auto()
    DENOISED_SINGLE_CELL_GENE_EXPRESSION = auto()


if TYPE_CHECKING:
    import anndata
    import pandas as pd

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
    except ImportError as exc:
        raise ImportError(
            'anndata is required to create AnnData outputs. '
            "Install it with: pip install 'alphagenome-pytorch[scoring]'"
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
        case (
            OutputType.ATAC
            | OutputType.SC_ATAC
            | OutputType.CAGE
            | OutputType.DNASE
            | OutputType.RNA_SEQ
            | OutputType.METHYLATION
            | OutputType.SPLICE_SITES
            | OutputType.SPLICE_SITE_USAGE
            | OutputType.SPLICE_JUNCTIONS
            | OutputType.PROCAP
            | OutputType.GENE_EXPRESSION
            | OutputType.SINGLE_CELL_GENE_EXPRESSION
            | OutputType.DENOISED_SINGLE_CELL_GENE_EXPRESSION
        ):
            return 1
        case OutputType.CHIP_HISTONE:
            return 128
        case OutputType.CHIP_TF:
            return 128
        case OutputType.CONTACT_MAPS:
            return 2048
        case _:
            raise ValueError(f'Unknown output type: {output_type}.')


class VariantScorer(
    Generic[VariantMaskT, VariantMetadataT, VariantSettingsT],
    abc.ABC,
):
    """Abstract class for variant scorers.

    A variant scorer computes effect scores for genomic variants by comparing
    predictions from reference and alternate alleles.
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


HEAD_TO_OUTPUT_SPEC = dict(
    atac = ('ATAC', 1),
    sc_atac = ('SC_ATAC', 1),
    cage = ('CAGE', 1),
    dnase = ('DNASE', 1),
    rna_seq = ('RNA_SEQ', 1),
    procap = ('PROCAP', 1),
    chip_histone = ('CHIP_HISTONE', 128),
    chip_tf = ('CHIP_TF', 128),
    contact_maps = ('CONTACT_MAPS', 2048),
    methylation = ('METHYLATION', 1),
    splice_sites_classification = ('SPLICE_SITES', 1),
    splice_sites_usage = ('SPLICE_SITE_USAGE', 1),
    splice_sites_junction = ('SPLICE_JUNCTIONS', 1),
    gene_expression = ('GENE_EXPRESSION', 1),
    single_cell_gene_expression = ('SINGLE_CELL_GENE_EXPRESSION', 1),
    denoised_single_cell_gene_expression = ('DENOISED_SINGLE_CELL_GENE_EXPRESSION', 1),
)


def _select_primary_prediction(head_out, resolution: int):
    if isinstance(head_out, dict):
        for prefix in ('scaled_predictions', 'predictions'):
            key = f'{prefix}_{resolution}bp'
            if key in head_out:
                return head_out[key]
    return head_out


def _resolve_mapping_key(mapping, key):
    if key in mapping:
        return key
    name = key.name if isinstance(key, Enum) else (key if isinstance(key, str) else None)
    if not exists(name):
        return None
    for candidate in (name, name.lower(), name.upper()):
        if candidate in mapping:
            return candidate
    for candidate in mapping:
        if isinstance(candidate, Enum) and candidate.name == name.upper():
            return candidate
    return None


def _mask_scored_prediction(pred, mask):
    if not exists(pred) or not exists(mask):
        return pred
    if isinstance(pred, dict) and 'predictions' in pred:
        inner = pred['predictions']
        if not exists(inner):
            return pred
        tiled = torch.as_tensor(mask, device = inner.device, dtype = torch.bool).repeat(2) if torch.is_tensor(inner) else np.tile(mask, 2)
        return dict(pred, predictions = inner[..., tiled])
    if torch.is_tensor(pred):
        return pred[..., torch.as_tensor(mask, device = pred.device, dtype = torch.bool)]
    return pred[..., mask]


def _rc_scored_prediction(pred, reindex):
    from torch_einops_utils.shape import is_shape

    if not exists(pred) or not exists(reindex) or (isinstance(pred, dict) and 'predictions' in pred):
        return pred
    if torch.is_tensor(pred):
        flip_dims = (1,) if is_shape(pred, 'b n c') else ((1, 2) if is_shape(pred, 'b h w c') else ())
        pred_rc = pred.flip(flip_dims) if flip_dims else pred
        return pred_rc.index_select(-1, torch.as_tensor(reindex, device = pred_rc.device, dtype = torch.long))
    pred_rc = pred[:, ::-1, :] if pred.ndim == 3 else (pred[:, ::-1, ::-1, :] if pred.ndim == 4 else pred)
    return pred_rc[..., reindex]


def score_variant_with_model(
    model,
    seq_ref,
    seq_alt,
    organism_index,
    scorer,
    settings,
    variant,
    interval,
    track_metadata = None,
    *,
    organism = None,
    **infer_kwargs,
):
    from torch_einops_utils import tree_map_tensor
    from torch_einops_utils.shape import is_shape, size

    ref_out = model.inference(seq_ref, organism_index, **infer_kwargs)
    alt_out = model.inference(seq_alt, organism_index, **infer_kwargs)

    if model.heads and isinstance(ref_out, dict) and ref_out:
        if not exists(organism):
            if len(ref_out) == 1:
                organism = next(iter(ref_out))
            elif 'shared' in ref_out:
                organism = 'shared'
            else:
                raise ValueError('Multiple organisms present in model output; pass `organism=...`.')
        if organism not in ref_out:
            raise KeyError(f'Organism {organism!r} not found in model output {list(ref_out)}.')
        ref_out, alt_out = ref_out[organism], alt_out[organism]

    requested_output = settings if isinstance(settings, (Enum, str)) else settings.requested_output
    output_enum = type(requested_output) if isinstance(requested_output, Enum) else None

    if exists(output_enum) and isinstance(ref_out, dict):
        def map_outputs(outputs):
            mapped = {
                output_enum[spec[0]]: _select_primary_prediction(val, spec[1])
                for head_name, val in outputs.items()
                if exists(spec := HEAD_TO_OUTPUT_SPEC.get(head_name)) and spec[0] in output_enum.__members__
            }
            return mapped or outputs

        ref_out, alt_out = map_outputs(ref_out), map_outputs(alt_out)
        if requested_output not in ref_out:
            raise KeyError(f'Requested output {requested_output!r} not found in model predictions.')

    metadata = track_metadata

    try:
        padding = metadata.padding
    except AttributeError:
        padding = None

    if exists(requested_output) and exists(padding) and isinstance(ref_out, dict):
        padding_key = _resolve_mapping_key(padding, requested_output)
        if exists(padding_key):
            mask = ~np.asarray(padding[padding_key])
            out_key = _resolve_mapping_key(ref_out, requested_output) or requested_output
            if out_key in ref_out:
                ref_out[out_key] = _mask_scored_prediction(ref_out[out_key], mask)
            if out_key in alt_out:
                alt_out[out_key] = _mask_scored_prediction(alt_out[out_key], mask)
            if isinstance(track_metadata, Mapping) and exists(meta := track_metadata.get(padding_key)):
                track_metadata = {out_key: meta[mask]}

    try:
        reindexing = metadata.strand_reindexing
    except AttributeError:
        reindexing = None

    if exists(requested_output) and exists(interval) and interval.negative_strand and exists(reindexing) and isinstance(ref_out, dict):
        reindex_key = _resolve_mapping_key(reindexing, requested_output)
        if exists(reindex_key):
            reindex = reindexing[reindex_key]
            out_key = _resolve_mapping_key(ref_out, requested_output) or requested_output
            if out_key in ref_out:
                ref_out[out_key] = _rc_scored_prediction(ref_out[out_key], reindex)
            if out_key in alt_out:
                alt_out[out_key] = _rc_scored_prediction(alt_out[out_key], reindex)

    is_single = isinstance(seq_ref, str) or (torch.is_tensor(seq_ref) and (is_shape(seq_ref, 'n') or size(seq_ref, '[b] ...') == 1))
    if is_single and isinstance(ref_out, dict):
        squeeze_fn = lambda t: t.squeeze(0) if is_shape(t, '1 _ ...') else t
        ref_out, alt_out = tree_map_tensor(squeeze_fn, ref_out), tree_map_tensor(squeeze_fn, alt_out)

    masks, mask_metadata = scorer.get_masks_and_metadata(interval, variant, settings = settings, track_metadata = track_metadata)
    scores = scorer.score_variant(ref_out, alt_out, masks = masks, settings = settings, variant = variant, interval = interval)
    scores_np = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in scores.items()}

    return scorer.finalize_variant(scores_np, track_metadata = track_metadata, mask_metadata = mask_metadata, settings = settings)


class DirectGeneExpressionScorer(VariantScorer[None, 'pd.DataFrame | None', OutputType]):
    """Scores Alt/Ref gene expression as log(y_alt + eps) - log(y_ref + eps)."""

    def __init__(self, pseudocount = 1e-3, gene_metadata = None):
        self.pseudocount = pseudocount
        self.gene_metadata = gene_metadata

    def get_masks_and_metadata(
        self,
        interval: Interval,
        variant: Variant,
        *,
        settings: OutputType,
        track_metadata: OutputMetadata,
    ) -> tuple[None, 'pd.DataFrame | None']:
        return None, self.gene_metadata

    def score_variant(
        self,
        ref: Mapping[OutputType, 'Tensor'],
        alt: Mapping[OutputType, 'Tensor'],
        *,
        masks = None,
        settings: OutputType,
        variant: Variant | None = None,
        interval: Interval | None = None,
    ) -> ScoreVariantOutput:
        ref_expr, alt_expr = ref[settings], alt[settings]

        if isinstance(ref_expr, dict):
            ref_expr = ref_expr['predictions_1bp']

        if isinstance(alt_expr, dict):
            alt_expr = alt_expr['predictions_1bp']

        return dict(score = torch.log(alt_expr + self.pseudocount) - torch.log(ref_expr + self.pseudocount))

    def finalize_variant(
        self,
        scores: ScoreVariantResult,
        *,
        track_metadata: OutputMetadata,
        mask_metadata: 'pd.DataFrame | None',
        settings: OutputType,
    ) -> anndata.AnnData:
        import pandas as pd

        score = np.asarray(scores['score'])
        var = track_metadata.get(settings) if isinstance(track_metadata, Mapping) else track_metadata

        if not isinstance(var, pd.DataFrame):
            var = pd.DataFrame(dict(name = [f'channel_{i}' for i in range(score.shape[-1])]))

        return create_anndata(score, obs = mask_metadata, var = var)
