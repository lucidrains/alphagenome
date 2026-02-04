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

"""Implementation of contact map variant scorer (PyTorch port).

This implements the Orca scoring strategy from Zhou 2022.
"""

import math
from collections.abc import Mapping
from typing import Any

import anndata
import numpy as np
import pandas as pd
import torch

from alphagenome_pytorch.scoring.variant_scoring import (
    Interval,
    Variant,
    create_anndata,
    get_resolution,
)

# Default resolution for contact maps
CONTACT_MAP_RESOLUTION = 2048


def create_center_mask(
    interval: Interval,
    variant: Variant,
    *,
    width: int | None,
    resolution: int,
) -> np.ndarray:
    """Creates a mask centered on a variant for a given interval.

    Args:
        interval: The genomic interval.
        variant: The variant to center the mask on.
        width: The width of the mask in base pairs. If None, mask covers full interval.
        resolution: The resolution of the mask (bin size).

    Returns:
        Boolean mask array of shape [S, 1].
    """
    if width is None:
        if interval.start <= variant.start < interval.end:
            mask = np.ones([interval.width // resolution, 1], dtype=bool)
        else:
            mask = np.zeros([interval.width // resolution, 1], dtype=bool)
    else:
        target_resolution_width = math.ceil(width / resolution)

        # Determine the position of the variant in the specified resolution.
        variant_start = getattr(variant, 'start', variant.position)
        base_resolution_center = variant_start - interval.start
        target_resolution_center = base_resolution_center // resolution

        # Compute start and end indices of the variant-centered mask.
        target_resolution_start = max(
            target_resolution_center - target_resolution_width // 2, 0
        )
        target_resolution_end = min(
            (target_resolution_center - target_resolution_width // 2)
            + target_resolution_width,
            interval.width // resolution,
        )

        # If the variant is not within the interval, we return an empty mask.
        # Otherwise, we build the mask using our target_resolution_start/end,
        # taking into account the resolution.
        mask = np.zeros([interval.width // resolution, 1], dtype=bool)
        if interval.start <= variant.start < interval.end:
            mask[target_resolution_start:target_resolution_end] = 1

    return mask


class ContactMapScorer:
    """Implements the contact map variant scoring strategy from Zhou 2022 (Orca).

    Designed for single nucleotide variant (SNV) scoring, where the expected
    effect is local to the variant position.

    Not compatible with indels or structural variants that involve changing more
    than the number of nucleotides in a 2Kb window.

    Citation: As described in the Zhou manuscript:
      https://doi.org/10.1038/s41588-022-01065-4:

    "The disruption impact on local genome interactions is measured by 1-Mb
    structural impact score, which is the average absolute log fold change of
    interactions between the disruption position and all other positions in the
    1-Mb window"

    The Orca scoring strategy is open sourced here (line 120):
    https://github.com/jzhoulab/orca_manuscript/blob/main/virtual_screen/local_interaction_screen.py
    """

    def __init__(self, resolution: int = CONTACT_MAP_RESOLUTION):
        """Initialize the ContactMapScorer.

        Args:
            resolution: Resolution of the contact map in base pairs.
        """
        self._resolution = resolution

    def get_masks_and_metadata(
        self,
        interval: Interval,
        variant: Variant,
        *,
        settings,
        track_metadata: pd.DataFrame | dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, None]:
        """Get masks and metadata for variant scoring.

        Args:
            interval: The genomic interval.
            variant: The variant to score.
            track_metadata: Optional track metadata (unused).

        Returns:
            Tuple of (center mask, None).

        Raises:
            ValueError: If the variant affects more than one bin or no bins.
        """
        del track_metadata  # Unused.

        requested_output = getattr(settings, 'requested_output', settings)
        try:
            resolution = get_resolution(requested_output)
        except Exception:
            resolution = self._resolution
        mask = create_center_mask(
            interval, variant, width=resolution, resolution=resolution
        )

        if mask.sum() > 1:
            raise ValueError(
                'The ContactMapScorer only accepts input variants that affect one bin'
                ' position. However, there is more than one position affected by the'
                ' variant at this bin resolution. This could indicate a malformed'
                ' center mask. Please check `create_center_mask` logic. Debugging'
                f' details: {variant=}, {interval=}, bin width={resolution},'
                f' {mask.sum()=}.'
            )
        elif mask.sum() == 0:
            raise ValueError(
                'The variant does not affect any positions at this bin resolution.'
                f' Debugging details: {variant=}, {interval=}, bin'
                f' width={resolution}, {mask.sum()=}.'
            )
        return mask, None

    def score_variant(
        self,
        ref: Mapping[str, torch.Tensor],
        alt: Mapping[str, torch.Tensor],
        *,
        masks: np.ndarray,
        settings,
        variant: Variant | None = None,
        interval: Interval | None = None,
    ) -> dict[str, np.ndarray]:
        """Score the variant using the Orca strategy.

        Computes mean absolute difference and selects the variant row.

        Args:
            ref: Reference predictions mapping output type to tensor.
            alt: Alternative predictions mapping output type to tensor.
            masks: Center mask for the variant.
            requested_output: The output type to use for scoring.
            variant: Optional variant (unused).
            interval: Optional interval (unused).

        Returns:
            Dictionary with 'score' key containing the variant score.
        """
        del variant, interval  # Unused.

        requested_output = getattr(settings, 'requested_output', settings)
        output_key = _resolve_output_key(ref, requested_output)

        ref_tensor = ref[output_key]
        alt_tensor = alt[output_key]

        # Mean absolute difference, reduced over contact map rows.
        # Ref, alt shape: [H, W, C]
        # Temps shape: [W, C]
        abs_diff = torch.abs(alt_tensor - ref_tensor).mean(dim=0)

        # Convert masks to tensor
        masks_tensor = torch.from_numpy(masks).to(abs_diff.device)

        # Use center mask to select the variant row.
        # Right now, assumes there is a single value.
        # Output shape: [C]
        output = abs_diff[masks_tensor.squeeze().argmax(), :]

        return {'score': output.cpu().numpy()}

    def finalize_variant(
        self,
        scores: dict[str, np.ndarray],
        *,
        track_metadata: pd.DataFrame,
        mask_metadata: None = None,
        settings=None,
    ) -> anndata.AnnData:
        """Finalize variant scoring by creating AnnData object.

        Args:
            scores: Dictionary containing 'score'.
            track_metadata: Track metadata DataFrame.
            mask_metadata: Unused (None).
            requested_output: Optional output type (unused).

        Returns:
            AnnData object with variant scores.
        """
        del mask_metadata  # Unused.

        requested_output = getattr(settings, 'requested_output', settings)
        if isinstance(track_metadata, dict) and requested_output is not None:
            meta_key = _resolve_output_key(track_metadata, requested_output)
            if meta_key is not None:
                track_metadata = track_metadata[meta_key]

        num_tracks = len(track_metadata)
        return create_anndata(
            scores['score'][np.newaxis, :num_tracks],
            obs=None,
            var=track_metadata,
        )


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
