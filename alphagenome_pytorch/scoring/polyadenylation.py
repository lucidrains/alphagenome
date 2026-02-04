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

"""Implements a variant scorer for polyadenylation (PyTorch port)."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import anndata
import numpy as np
import pandas as pd
import torch

from alphagenome_pytorch.scoring.variant_scoring import (
    Interval,
    Variant,
    align_alternate,
    create_anndata,
)

MAX_GENES = 22
MAX_PAS = 136
_PAS_MASK_WIDTH = 400


@dataclass
class PolyadenylationVariantMasks:
    """Masks for polyadenylation variant scoring."""

    pas_mask: np.ndarray | torch.Tensor  # [S, G, P]
    gene_mask: np.ndarray | torch.Tensor  # [G]


def _aggregate_maximum_ratio_coverage_fc(
    ref: torch.Tensor,  # [S, T]
    alt: torch.Tensor,  # [S, T]
    gene_pas_mask: torch.Tensor,  # [S, G, P]
) -> torch.Tensor:
    """Implements the Borzoi statistic for paQTL variant scoring.

    Args:
        ref: Reference predictions of shape [S, T].
        alt: Alternative predictions of shape [S, T].
        gene_pas_mask: Boolean mask of shape [S, G, P] indicating PAS sites.

    Returns:
        Scores of shape [G, T].
    """
    # Convert mask to float for einsum
    gene_pas_mask_float = gene_pas_mask.float()

    # ref: [S, T], gene_pas_mask: [S, G, P]
    # ref_aggregation: [G, P, T]
    ref_aggregation = torch.einsum('st,sgp->gpt', ref, gene_pas_mask_float)
    alt_aggregation = torch.einsum('st,sgp->gpt', alt, gene_pas_mask_float)

    covr_ratio = alt_aggregation / ref_aggregation
    covr_ratio = torch.nan_to_num(covr_ratio, posinf=0, neginf=0, nan=0)
    # Shape: [G, P, T]

    # Get proximal vs distal counts for all possible polyadenylation site
    # split versions.
    k_interval = torch.arange(MAX_PAS, device=ref.device)  # All PAS for the interval.

    # Create mask for potential proximal pas site splits across the interval.
    # Each PAS is added to the mask in sequential order for each gene, which
    # ensures that aggregating the proximal counts take the first k PAS sites
    # for each gene.
    proximal_sites = k_interval[None] <= k_interval[:, None]  # [K, P]

    # Get total number of pas sites per gene
    gene_pas_max = gene_pas_mask_float.max(dim=0).values  # [G, P]
    k_total = gene_pas_max.sum(dim=-1, keepdim=True)  # [G, 1]

    # Get number of pas sites included in the proximal split per gene.
    k_gene = gene_pas_max.cumsum(dim=-1)  # [G, P]
    k_scaling = ((k_total - k_gene) / k_gene).T[:, :, None]  # [P, G, 1]

    # Convert to [K, P] -> einsum with [G, P, T]
    proximal_sites_float = proximal_sites.float()
    proximal_counts = torch.einsum('gpt,kp->kgt', covr_ratio, proximal_sites_float)
    distal_counts = torch.einsum('gpt,kp->kgt', covr_ratio, (~proximal_sites).float())

    scores = torch.abs(torch.log2(k_scaling * proximal_counts / distal_counts))
    # We are converting nan to num to keep all the padding cases at 0.
    scores = torch.nan_to_num(scores, posinf=0, neginf=0, nan=0)
    # Shape: [K, G, T]

    return scores.max(dim=0).values  # [G, T]


class PolyadenylationVariantScorer:
    """Variant scorer for polyadenylation."""

    def __init__(
        self,
        gtf: pd.DataFrame,
        pas_gtf: pd.DataFrame,
        gene_mask_extractor: Any | None = None,
    ):
        """Initialize the scorer.

        Args:
            gtf: GTF DataFrame with gene annotations.
            pas_gtf: GTF DataFrame with polyadenylation site annotations.
            gene_mask_extractor: Optional gene mask extractor.
        """
        self._gtf = gtf
        self._gene_mask_extractor = gene_mask_extractor

        if 'gene_id_nopatch' not in pas_gtf.columns:
            pas_gtf = pas_gtf.copy()
            pas_gtf['gene_id_nopatch'] = pas_gtf['gene_id'].str.split(
                '.', expand=True
            )[0]

        self._pas_per_gene = {
            gene_id_gtf: df
            for gene_id_gtf, df in pas_gtf.groupby('gene_id_nopatch')
        }

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
            'gene_id': gene_df['gene_id'].values,
            'Strand': gene_df['Strand'].values,
            'gene_name': gene_df.get('gene_name', pd.Series([''] * len(gene_df))).values,
            'gene_type': gene_df.get('gene_type', pd.Series([''] * len(gene_df))).values,
            'interval_start': interval.start,
            'Chromosome': gene_df['Chromosome'].values,
            'Start': gene_df['Start'].values,
            'End': gene_df['End'].values,
        })

    def get_masks_and_metadata(
        self,
        interval: Interval,
        variant: Variant,
        *,
        settings=None,
        track_metadata: pd.DataFrame | dict[str, Any] | None = None,
    ) -> tuple[PolyadenylationVariantMasks, pd.DataFrame]:
        """Get masks and metadata for variant scoring.

        Args:
            interval: The genomic interval.
            variant: The variant to score.
            track_metadata: Optional track metadata (unused).

        Returns:
            Tuple of (PolyadenylationVariantMasks, metadata DataFrame).

        Raises:
            ValueError: If too many genes are found for the interval.
        """
        del track_metadata, settings

        if self._gene_mask_extractor is not None:
            _, gene_metadata = self._gene_mask_extractor.extract(interval, variant)
        else:
            gene_metadata = self._extract_genes(interval, variant)

        if len(gene_metadata) > MAX_GENES:
            raise ValueError(
                f'Too many genes found for interval {interval}: {len(gene_metadata)}'
            )

        gene_metadata_rows = []
        gene_padding_mask = np.zeros(MAX_GENES, dtype=bool)
        pas_mask = np.zeros(
            (interval.width, MAX_GENES, MAX_PAS),
            dtype=bool,
        )

        has_gene_id_nopatch = 'gene_id_nopatch' in gene_metadata.columns

        for gene_index, gene_row in gene_metadata.iterrows():
            if has_gene_id_nopatch:
                gene_id = gene_row['gene_id_nopatch']
            else:
                gene_id = gene_row['gene_id'].split('.')[0]

            if gene_id not in self._pas_per_gene:
                continue

            gene_pas = self._pas_per_gene[gene_id]
            gene_pas = gene_pas[gene_pas['pas_strand'] == gene_row['Strand']]

            if (
                gene_pas.shape[0] == 0
                # Check at least 80% of a gene's PAS sites fall within the interval.
                or np.mean((gene_pas['Start'] >= interval.start).values) < 0.8
                or np.mean((gene_pas['End'] < interval.end).values) < 0.8
            ):
                # No PAS sites in interval for the gene.
                continue

            # Only look at PAS sites that fall in the interval.
            gene_pas = gene_pas[
                (gene_pas['Start'] >= interval.start)
                & (gene_pas['End'] < interval.end)
            ]

            if gene_pas.shape[0] == 1:
                # Only one PAS site in interval for the gene.
                continue
            else:
                pas_interval_start = gene_pas['Start'] - interval.start
                gene_pas = gene_pas.sort_values(by='Start')
                # Get PAS metadata for gene.
                gene_row_metadata = gene_row.to_dict()
                dist = np.abs(gene_pas['Start'] - variant.position)
                gene_row_metadata['num_pas'] = len(gene_pas)
                gene_row_metadata['min_pas_var_distance'] = dist.min()
                gene_padding_mask[gene_index] = True
                gene_metadata_rows.append(gene_row_metadata)

            for (pas_index, pas_row), p_interval_start in zip(
                gene_pas.reset_index(drop=True).iterrows(),
                pas_interval_start,
                strict=True,
            ):
                # Defaults to only doing upstream coverage of PAS site.
                if pas_row.pas_strand == '+':
                    bin_end = p_interval_start + 1
                    bin_start = bin_end - _PAS_MASK_WIDTH
                else:
                    bin_start = p_interval_start
                    bin_end = bin_start + _PAS_MASK_WIDTH
                bin_start = max(min(bin_start, interval.width), 0)
                bin_end = max(min(bin_end, interval.width), 0)
                pas_mask[bin_start:bin_end, gene_index, pas_index] = True

        return (
            PolyadenylationVariantMasks(
                pas_mask=pas_mask, gene_mask=gene_padding_mask
            ),
            pd.DataFrame(gene_metadata_rows),
        )

    def score_variant(
        self,
        ref: Mapping[str, torch.Tensor],
        alt: Mapping[str, torch.Tensor],
        *,
        masks: PolyadenylationVariantMasks,
        settings,
        variant: Variant | None = None,
        interval: Interval | None = None,
    ) -> dict[str, np.ndarray]:
        """Score the variant by computing coverage ratios.

        Args:
            ref: Reference predictions mapping output type to tensor.
            alt: Alternative predictions mapping output type to tensor.
            masks: PolyadenylationVariantMasks with PAS and gene masks.
            requested_output: The output type to use for scoring.
            variant: The variant being scored (used for alignment).
            interval: The genomic interval (used for alignment).

        Returns:
            Dictionary with 'scores' and 'gene_mask'.
        """
        requested_output = getattr(settings, 'requested_output', settings)
        output_key = _resolve_output_key(ref, requested_output)

        ref_tensor = ref[output_key]
        alt_tensor = alt[output_key]

        # Align alternate if variant causes indel
        if variant is not None and interval is not None:
            alt_tensor = align_alternate(alt_tensor, variant, interval)

        # Convert masks to tensors
        pas_mask_tensor = torch.from_numpy(masks.pas_mask).to(ref_tensor.device)

        scores = _aggregate_maximum_ratio_coverage_fc(
            ref_tensor, alt_tensor, pas_mask_tensor
        )

        return {
            'scores': scores.cpu().numpy(),
            'gene_mask': masks.gene_mask,
        }

    def finalize_variant(
        self,
        scores: dict[str, np.ndarray],
        *,
        track_metadata: pd.DataFrame,
        mask_metadata: pd.DataFrame,
        settings=None,
    ) -> anndata.AnnData:
        """Finalize variant scoring by filtering with gene mask.

        Args:
            scores: Dictionary containing 'scores' and 'gene_mask'.
            track_metadata: Track metadata DataFrame.
            mask_metadata: Mask metadata DataFrame with gene info.
            requested_output: Optional output type (unused).

        Returns:
            AnnData object with variant scores.
        """
        requested_output = getattr(settings, 'requested_output', settings)
        if isinstance(track_metadata, dict) and requested_output is not None:
            meta_key = _resolve_output_key(track_metadata, requested_output)
            if meta_key is not None:
                track_metadata = track_metadata[meta_key]

        return create_anndata(
            scores['scores'][scores['gene_mask']],
            obs=mask_metadata,
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
