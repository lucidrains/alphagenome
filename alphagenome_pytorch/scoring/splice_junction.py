# Ported from alphagenome_research reference implementation
# (reference/src/alphagenome_research/model/variant_scoring/)
#
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of splice junction variant scoring (PyTorch port)."""

from collections.abc import Mapping
from typing import Any

import anndata
import numpy as np
import pandas as pd
import pyranges
import torch

from alphagenome_pytorch.scoring.variant_scoring import (
    Interval,
    Variant,
    create_anndata,
)

_MAX_SPLICE_SITES = 256
PAD_VALUE = -1


def _create_empty(mask_metadata: pd.DataFrame, track_metadata: pd.DataFrame):
    """Create empty AnnData object for splice junction scoring."""
    junction_columns = [
        'junction_Start',
        'junction_End',
    ]
    return create_anndata(
        np.zeros((0, len(track_metadata['name'])), dtype=np.float32),
        obs=pd.DataFrame(columns=list(mask_metadata.columns) + junction_columns),
        var=pd.DataFrame({
            'strand': '.',
            'name': track_metadata['name'],
            'gtex_tissue': track_metadata['gtex_tissue'],
            'ontology_curie': track_metadata.get('ontology_curie'),
            'biosample_type': track_metadata.get('biosample_type'),
            'biosample_name': track_metadata.get('biosample_name'),
            'biosample_life_stage': track_metadata.get('biosample_life_stage'),
            'data_source': track_metadata.get('data_source'),
            'Assay title': track_metadata.get('Assay title'),
        }),
    )


def _create(
    junction_scores: pd.DataFrame,
    mask_metadata: pd.DataFrame,
    track_metadata: pd.DataFrame,
) -> anndata.AnnData:
    """Converts a dataframe of junction scores to an AnnData object."""
    if mask_metadata.empty or junction_scores.empty:
        raise ValueError('Both junction_scores and mask_metadata must be non-empty')

    junction_scores = junction_scores[
        junction_scores['gene_id'].isin(mask_metadata['gene_id'])
    ]

    gene_max_scores = []
    track_names = track_metadata['name']
    for gene_id in junction_scores['gene_id'].unique():
        gene_junction_scores = junction_scores[
            junction_scores.gene_id == gene_id
        ].reset_index(drop=True)
        gene_junction_scores = gene_junction_scores.iloc[
            gene_junction_scores[track_names].values.argmax(0)
        ]
        gene_max_scores.append(gene_junction_scores)
    junction_scores = pd.concat(gene_max_scores)
    score_values = junction_scores[track_names].values

    # Merge junction information with mask metadata.
    junctions_all_genes = junction_scores[['gene_id', 'Start', 'End']]
    junctions_all_genes.columns = ['gene_id', 'junction_Start', 'junction_End']
    mask_metadata = junctions_all_genes.merge(
        mask_metadata, on='gene_id', sort=False
    )
    # Create the final track metadata.
    track_metadata = pd.DataFrame({
        'strand': '.',  # We already matched prediction by strand.
        'name': track_names,
        'gtex_tissue': track_metadata['gtex_tissue'],
        'ontology_curie': track_metadata.get('ontology_curie'),
        'biosample_type': track_metadata.get('biosample_type'),
        'biosample_name': track_metadata.get('biosample_name'),
        'biosample_life_stage': track_metadata.get('biosample_life_stage'),
        'data_source': track_metadata.get('data_source'),
        'Assay title': track_metadata.get('Assay title'),
    })
    ann_data = create_anndata(
        score_values,
        obs=mask_metadata,
        var=track_metadata,
    )
    # Remove duplicated junctions. Per gene, we report junctions that has maximum
    # score in any tissue. For the reported junctions, we return their predictions
    # in all tissues.
    return ann_data[~ann_data.obs.duplicated()].copy()


def unstack_junction_predictions(
    splice_junction_prediction: np.ndarray,  # [D, D, _]
    splice_site_positions: np.ndarray,  # [4, D]
    interval: Interval | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unstack splice junction predictions to long format.

    Args:
        splice_junction_prediction: Junction predictions of shape [D, D, T*2]
            where T is number of tracks and 2 is for +/- strands.
        splice_site_positions: Splice site positions of shape [4, D] where
            the 4 rows are: pos_donors, pos_acceptors, neg_donors, neg_acceptors.
        interval: Optional interval to offset positions.

    Returns:
        Tuple of (junction_predictions, strands, starts, ends).
    """
    # Unpack splice junction predictions: reshape from [D, D, T*2] to [D, D, 2, T]
    d1, d2, t_times_2 = splice_junction_prediction.shape
    t = t_times_2 // 2
    splice_junction_prediction = splice_junction_prediction.reshape(d1, d2, 2, t)
    splice_junction_prediction = splice_junction_prediction.transpose(0, 1, 3, 2)
    # Now shape is [D, D, T, 2]

    # Convert splice site positions.
    remove_padding_fn = lambda x: x[x != PAD_VALUE]
    pos_donors = remove_padding_fn(splice_site_positions[0])
    pos_acceptors = remove_padding_fn(splice_site_positions[1])
    neg_donors = remove_padding_fn(splice_site_positions[2])
    neg_acceptors = remove_padding_fn(splice_site_positions[3])

    # Positive strand junctions
    junction_pred_pos = splice_junction_prediction[
        : len(pos_donors), : len(pos_acceptors), :, 0
    ]
    # Reshape from [D, A, T] to [D*A, T]
    junction_pred_pos = junction_pred_pos.reshape(-1, t)
    num_pos_donors = len(pos_donors)
    pos_donors_expanded = np.repeat(pos_donors, len(pos_acceptors))
    pos_acceptors_expanded = np.tile(pos_acceptors, num_pos_donors)

    # Negative strand junctions
    junction_pred_neg = splice_junction_prediction[
        : len(neg_donors), : len(neg_acceptors), :, 1
    ]
    # Reshape from [D, A, T] to [D*A, T]
    junction_pred_neg = junction_pred_neg.reshape(-1, t)
    num_neg_donors = len(neg_donors)
    neg_donors_expanded = np.repeat(neg_donors, len(neg_acceptors))
    neg_acceptors_expanded = np.tile(neg_acceptors, num_neg_donors)

    # Combine into a single output.
    junction_predictions = np.concatenate(
        [junction_pred_pos, junction_pred_neg], axis=0
    )

    # Junction start and end positions.
    starts = np.concatenate([pos_donors_expanded, neg_acceptors_expanded]) + 1
    starts = starts + (interval.start if interval is not None else 0)
    ends = np.concatenate([pos_acceptors_expanded, neg_donors_expanded])
    ends = ends + (interval.start if interval is not None else 0)
    strands = np.array(['+'] * len(pos_donors_expanded) + ['-'] * len(neg_donors_expanded))
    filter_mask = (starts < ends) & (starts > 0)

    return (
        junction_predictions[filter_mask],
        strands[filter_mask],
        starts[filter_mask],
        ends[filter_mask],
    )


def junction_predictions_to_dataframe(
    splice_junction_prediction: np.ndarray,  # [D, D, _]
    splice_site_positions: np.ndarray,  # [T_mul_4, D]
    metadata: pd.DataFrame | dict[str, Any],
    interval: Interval,
) -> pd.DataFrame:
    """Convert splice junction predictions to a dataframe."""
    junction_predictions, strands, starts, ends = unstack_junction_predictions(
        splice_junction_prediction, splice_site_positions, interval
    )

    if isinstance(metadata, dict):
        track_names = metadata['name']
    else:
        track_names = metadata['name'].tolist()

    junctions = pd.DataFrame({
        'Chromosome': interval.chromosome,
        'Start': starts,
        'End': ends,
        'Strand': strands,
    })
    predictions = pd.DataFrame(junction_predictions, columns=track_names)
    return pd.concat([junctions, predictions], axis=1)


class SpliceJunctionVariantScorer:
    """Implements the SpliceJunction variant scoring strategy.

    Scores variants by the maximum of absolute delta pair counts of junctions
    within the input interval. Junctions are annotated by overlapping with the
    gtf gene intervals.
    """

    def __init__(self, gtf: pd.DataFrame, gene_mask_extractor: Any = None):
        """Initialize the scorer.

        Args:
            gtf: GTF DataFrame with gene annotations.
            gene_mask_extractor: Optional gene mask extractor. If not provided,
                a simple implementation will be used.
        """
        self._gtf = gtf
        self._gene_mask_extractor = gene_mask_extractor

    def get_masks_and_metadata(
        self,
        interval: Interval,
        variant: Variant,
        *,
        settings=None,
        track_metadata: pd.DataFrame | None = None,
    ) -> tuple[None, pd.DataFrame]:
        """Get masks and metadata for variant scoring.

        Args:
            interval: The genomic interval.
            variant: The variant to score.
            track_metadata: Optional track metadata (unused).

        Returns:
            Tuple of (None, metadata DataFrame).
        """
        del track_metadata, settings
        if self._gene_mask_extractor is not None:
            _, metadata = self._gene_mask_extractor.extract(interval, variant)
        else:
            # Simple fallback: extract genes overlapping the variant
            metadata = self._extract_genes(interval, variant)
        metadata = metadata.copy()
        metadata['interval'] = interval
        return None, metadata

    def _extract_genes(self, interval: Interval, variant: Variant) -> pd.DataFrame:
        """Extract genes overlapping the variant."""
        gene_df = self._gtf[self._gtf['Feature'] == 'gene'].copy()
        if gene_df.empty:
            return pd.DataFrame(columns=[
                'gene_id', 'Strand', 'gene_name', 'gene_type',
                'interval_start', 'Chromosome', 'Start', 'End'
            ])

        # Filter to chromosome
        gene_df = gene_df[gene_df['Chromosome'] == variant.chromosome]

        # Filter to overlapping genes
        variant_end = max(variant.end, variant.start + len(variant.alternate_bases))
        gene_df = gene_df[
            (gene_df['End'] > variant.start) & (gene_df['Start'] < variant_end)
        ]

        return pd.DataFrame({
            'gene_id': gene_df['gene_id'],
            'Strand': gene_df['Strand'],
            'gene_name': gene_df.get('gene_name', ''),
            'gene_type': gene_df.get('gene_type', ''),
            'interval_start': interval.start,
            'Chromosome': gene_df['Chromosome'],
            'Start': gene_df['Start'],
            'End': gene_df['End'],
        })

    def score_variant(
        self,
        ref: Mapping[str, dict[str, torch.Tensor]],
        alt: Mapping[str, dict[str, torch.Tensor]],
        *,
        masks: None,
        settings,
        variant: Variant | None = None,
        interval: Interval | None = None,
    ) -> dict[str, np.ndarray]:
        """Score the variant by computing delta junction predictions.

        Args:
            ref: Reference predictions mapping output type to predictions dict.
            alt: Alternative predictions mapping output type to predictions dict.
            masks: Unused masks parameter.
            requested_output: The output type to use for scoring.
            variant: Optional variant (unused).
            interval: Optional interval (unused).

        Returns:
            Dictionary with 'delta_counts' and 'splice_site_positions'.
        """
        del variant, interval, masks

        requested_output = getattr(settings, 'requested_output', settings)
        output_key = _resolve_output_key(ref, requested_output)

        ref_junctions = ref[output_key]['predictions']
        alt_junctions = alt[output_key]['predictions']
        splice_site_positions = ref[output_key]['splice_site_positions']

        # Ignore splice sites beyond the max_splice_sites specified.
        ref_junctions = ref_junctions[:_MAX_SPLICE_SITES, :_MAX_SPLICE_SITES]
        alt_junctions = alt_junctions[:_MAX_SPLICE_SITES, :_MAX_SPLICE_SITES]
        splice_site_positions = splice_site_positions[:, :_MAX_SPLICE_SITES]

        # Apply log offset
        def apply_log_offset(x: torch.Tensor) -> torch.Tensor:
            return torch.log(x + 1e-7)

        ref_junctions = apply_log_offset(ref_junctions)
        alt_junctions = apply_log_offset(alt_junctions)

        delta_counts = (alt_junctions - ref_junctions).to(torch.float16)

        return {
            'delta_counts': delta_counts.cpu().numpy(),
            'splice_site_positions': splice_site_positions.cpu().numpy()
            if isinstance(splice_site_positions, torch.Tensor)
            else splice_site_positions,
        }

    def finalize_variant(
        self,
        scores: dict[str, np.ndarray],
        *,
        track_metadata: pd.DataFrame,
        mask_metadata: pd.DataFrame,
        settings=None,
    ) -> anndata.AnnData:
        """Finalize variant scoring by joining with gene metadata.

        Args:
            scores: Dictionary containing 'delta_counts' and 'splice_site_positions'.
            track_metadata: Track metadata DataFrame.
            mask_metadata: Mask metadata DataFrame with gene info and interval.
            requested_output: Optional output type (unused).

        Returns:
            AnnData object with variant scores.
        """
        requested_output = getattr(settings, 'requested_output', settings)

        if isinstance(track_metadata, dict) and requested_output is not None:
            meta_key = _resolve_output_key(track_metadata, requested_output)
            if meta_key is not None:
                track_metadata = track_metadata[meta_key]

        if mask_metadata.empty:
            return _create_empty(mask_metadata, track_metadata)

        delta_counts = scores['delta_counts']

        interval = mask_metadata['interval'].values[0]
        mask_metadata = mask_metadata.drop(columns=['interval'])

        delta_counts = junction_predictions_to_dataframe(
            np.abs(delta_counts).astype(np.float32),
            scores['splice_site_positions'],
            metadata=track_metadata,
            interval=interval,
        )
        if delta_counts.empty:
            return _create_empty(mask_metadata, track_metadata)

        junction_scores = (
            pyranges.PyRanges(delta_counts)
            .join(pyranges.PyRanges(mask_metadata), strandedness='same')
            .df
        )

        if not junction_scores.empty:
            junction_scores = junction_scores[
                (junction_scores['Start'] > junction_scores['Start_b'])
                & (junction_scores['End'] < junction_scores['End_b'])
            ]
            return _create(junction_scores, mask_metadata, track_metadata)
        else:
            return _create_empty(mask_metadata, track_metadata)


def _resolve_output_key(outputs: Mapping, requested_output):
    if requested_output in outputs:
        return requested_output
    if hasattr(requested_output, 'name'):
        name = requested_output.name
        if name in outputs:
            return name
        if name.lower() in outputs:
            return name.lower()
        for key in outputs:
            if hasattr(key, 'name') and key.name == name:
                return key
    if isinstance(requested_output, str):
        if requested_output in outputs:
            return requested_output
        if requested_output.lower() in outputs:
            return requested_output.lower()
        if requested_output.upper() in outputs:
            return requested_output.upper()
        for key in outputs:
            if hasattr(key, 'name') and key.name == requested_output.upper():
                return key
    raise KeyError(f'Requested output {requested_output!r} not found in predictions.')
