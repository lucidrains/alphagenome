# Ported from alphagenome_research reference implementation
# (reference/src/alphagenome_research/model/interval_scoring/)
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

"""PyTorch implementation of interval-level scoring.

This module provides interval-level scoring strategies (as opposed to
variant-level scoring). Interval scoring aggregates predictions over
genomic regions such as genes.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Generic, TypeVar, Union

import anndata
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from alphagenome.data import genome
from alphagenome.data import track_data
from alphagenome.models import dna_output
from alphagenome.models import interval_scorers
from alphagenome_research.model.variant_scoring import (
    gene_mask_extractor as gene_mask_extractor_lib,
)

from .variant_scoring import create_anndata, get_resolution

# Type variables for generic interval scorer
IntervalMaskT = TypeVar('IntervalMaskT')
IntervalMetadataT = TypeVar('IntervalMetadataT')
IntervalSettingsT = TypeVar('IntervalSettingsT')

# Type aliases for score outputs
ScoreIntervalOutput = Mapping[str, Union[Tensor, np.ndarray]]
ScoreIntervalResult = Mapping[str, np.ndarray]


class IntervalScorer(
    Generic[IntervalMaskT, IntervalMetadataT, IntervalSettingsT],
    metaclass=abc.ABCMeta,
):
    """Abstract base class for interval scorers.

    An interval scorer computes scores aggregated over genomic intervals,
    such as genes. Unlike variant scorers which compare reference and
    alternate predictions, interval scorers work with a single set of
    predictions.

    Type Parameters:
        IntervalMaskT: Type of masks used for scoring (e.g., gene masks).
        IntervalMetadataT: Type of metadata returned alongside masks.
        IntervalSettingsT: Type of settings/configuration for the scorer.
    """

    @abc.abstractmethod
    def get_masks_and_metadata(
        self,
        interval: genome.Interval,
        *,
        settings: IntervalSettingsT,
        track_metadata: dna_output.OutputMetadata,
    ) -> tuple[IntervalMaskT, IntervalMetadataT]:
        """Returns masks and metadata for the given interval and metadata.

        The generated masks and metadata will be passed to `score_interval`
        and `finalize_interval` respectively.

        Args:
            interval: The interval to score.
            settings: The interval scorer settings.
            track_metadata: The model's track metadata.

        Returns:
            A tuple of (masks, metadata), where:
                masks: The masks required to score the interval, such as gene
                    or TSS or strand masks. These will be passed into the
                    `score_interval` function.
                metadata: The metadata required to finalize the interval. These
                    will be passed into the `finalize_interval` function.

            The formats/shapes of masks and metadata will vary across interval
            scorers depending on their individual needs.
        """

    @abc.abstractmethod
    def score_interval(
        self,
        predictions: Mapping[dna_output.OutputType, Tensor],
        *,
        masks: IntervalMaskT,
        settings: IntervalSettingsT,
        interval: genome.Interval | None = None,
    ) -> ScoreIntervalOutput:
        """Generates a score per track for the provided predictions.

        Args:
            predictions: Model predictions for the interval, mapping output
                types to tensors of shape [S, T] (sequence_length x tracks).
            masks: The masks for scoring the interval.
            settings: The interval scorer settings.
            interval: The interval to score.

        Returns:
            Dictionary of scores to be passed to `finalize_interval`.
        """

    @abc.abstractmethod
    def finalize_interval(
        self,
        scores: ScoreIntervalResult,
        *,
        track_metadata: dna_output.OutputMetadata,
        mask_metadata: IntervalMetadataT,
        settings: IntervalSettingsT,
    ) -> anndata.AnnData:
        """Returns finalized scores for the given scores and metadata.

        Args:
            scores: Dictionary of scores generated from `score_interval`.
            track_metadata: Metadata describing the tracks for each output_type.
            mask_metadata: Metadata describing the masks.
            settings: The interval scorer settings.

        Returns:
            An AnnData object containing the final interval outputs. The entries
            will vary across scorers depending on their individual needs.
        """


class GeneIntervalScorer(
    IntervalScorer[
        Union[np.ndarray, Tensor],  # IntervalMaskT: boolean mask [S, G]
        pd.DataFrame,  # IntervalMetadataT: gene metadata
        interval_scorers.GeneMaskScorer,  # IntervalSettingsT
    ]
):
    """Interval scorer that aggregates predictions over gene regions.

    This scorer uses gene masks to aggregate predictions within gene regions,
    supporting both mean and sum aggregation types.
    """

    def __init__(
        self,
        gene_mask_extractor: gene_mask_extractor_lib.GeneMaskExtractor,
    ) -> None:
        """Initialize the GeneIntervalScorer.

        Args:
            gene_mask_extractor: Gene mask extractor to use for extracting
                gene regions from genomic annotations.
        """
        self._gene_mask_extractor = gene_mask_extractor

    def get_masks_and_metadata(
        self,
        interval: genome.Interval,
        *,
        settings: interval_scorers.GeneMaskScorer,
        track_metadata: dna_output.OutputMetadata,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Get gene masks and metadata for the given interval.

        Args:
            interval: Genomic interval to extract gene masks for.
            settings: The interval scorer settings.
            track_metadata: Track metadata (unused but required by interface).

        Returns:
            Tuple of (gene_mask, metadata) where:
                gene_mask: Boolean array of shape [S, G] indicating which
                    positions belong to which genes.
                metadata: DataFrame with gene annotation metadata.

        Raises:
            ValueError: If the interval is on the negative strand or if the
                specified width is larger than the interval width.
        """
        del track_metadata  # Unused

        if interval.negative_strand:
            raise ValueError(
                'IntervalScorers do not support negative strands (negative '
                'strand predictions should already be reverse-complemented '
                'prior to scoring and thus masks should be generated on the '
                'positive strand).'
            )
        if settings.width is not None and settings.width > interval.width:
            raise ValueError('Interval width must be >= the center mask width.')

        resolution = get_resolution(settings.requested_output)
        target_interval = (
            interval.resize(width=settings.width)
            if settings.width is not None
            else interval
        )

        gene_mask, metadata = self._gene_mask_extractor.extract(target_interval)

        # Pad the gene mask if the target interval is smaller than the input
        interval_padding = interval.width - target_interval.width
        gene_mask = np.pad(
            gene_mask,
            ((interval_padding // 2, (interval_padding + 1) // 2), (0, 0)),
        )

        # Downsample mask to match output resolution if needed
        if resolution > 1:
            gene_mask = gene_mask.reshape(
                (gene_mask.shape[0] // resolution, resolution, -1)
            ).max(axis=1)

        return gene_mask, metadata

    def score_interval(
        self,
        predictions: Mapping[dna_output.OutputType, Tensor],
        *,
        masks: Union[np.ndarray, Tensor],
        settings: interval_scorers.GeneMaskScorer,
        interval: genome.Interval | None = None,
    ) -> ScoreIntervalOutput:
        """Score predictions aggregated over gene regions.

        Args:
            predictions: Model predictions mapping output type to tensor [S, T].
            masks: Boolean gene mask of shape [S, G] (sequence x genes).
            settings: The interval scorer settings.
            interval: The genomic interval (unused).

        Returns:
            Dictionary containing the 'score' tensor of shape [G, T].
        """
        del interval  # Unused

        tracks = predictions[settings.requested_output]

        # Convert masks to tensor if needed
        if isinstance(masks, np.ndarray):
            gene_mask = torch.from_numpy(masks).to(
                dtype=torch.float32, device=tracks.device
            )
        else:
            gene_mask = masks.float()

        match settings.aggregation_type:
            case interval_scorers.IntervalAggregationType.MEAN:
                # Mean aggregation: sum(tracks * mask) / sum(mask)
                mask_sum = gene_mask.sum(dim=0, keepdim=True).T  # [G, 1]
                output = torch.einsum('lt,lg->gt', tracks, gene_mask) / mask_sum

            case interval_scorers.IntervalAggregationType.SUM:
                # Sum aggregation: sum(tracks * mask)
                output = torch.einsum('lt,lg->gt', tracks, gene_mask)

            case _:
                raise ValueError(
                    f'Unsupported aggregation type: {settings.aggregation_type}.'
                )

        return {'score': output}

    def finalize_interval(
        self,
        scores: ScoreIntervalResult,
        *,
        track_metadata: dna_output.OutputMetadata,
        mask_metadata: pd.DataFrame,
        settings: interval_scorers.GeneMaskScorer,
    ) -> anndata.AnnData:
        """Finalize and format interval scores as AnnData.

        Args:
            scores: Dictionary containing 'score' array of shape [G, T].
            track_metadata: Metadata for the output tracks.
            mask_metadata: Metadata for the gene masks.
            settings: The interval scorer settings.

        Returns:
            AnnData object with scores and metadata.
        """
        output_metadata = track_metadata.get(settings.requested_output)
        assert isinstance(output_metadata, track_data.TrackMetadata)

        # Convert scores to numpy if needed
        score_array = scores['score']
        if isinstance(score_array, torch.Tensor):
            score_array = score_array.cpu().numpy()

        return create_anndata(
            score_array,
            obs=mask_metadata,
            var=output_metadata,
        )
