# Ported from alphagenome_research reference implementation.
"""Bundle names and their associated keys and resolutions.

Each bundle represents a specific type of genomic data (e.g., ATAC-seq, CAGE)
with its own resolution and data/mask structure.
"""

import enum
from typing import Dict

import torch


class BundleName(enum.Enum):
    """Bundle names for different genomic data types."""

    ATAC = 'atac'
    DNASE = 'dnase'
    PROCAP = 'procap'
    CAGE = 'cage'
    RNA_SEQ = 'rna_seq'
    CHIP_TF = 'chip_tf'
    CHIP_HISTONE = 'chip_histone'
    CONTACT_MAPS = 'contact_maps'
    SPLICE_SITES_CLASSIFICATION = 'splice_sites'
    SPLICE_SITES_USAGE = 'splice_site_usage'
    SPLICE_SITES_JUNCTION = 'splice_junctions'
    SPLICE_SITES_POSITIONS = 'splice_site_positions'

    def get_dtypes(self) -> Dict[str, torch.dtype]:
        """Returns the keys and dtypes for the given bundle."""
        match self:
            case BundleName.ATAC:
                return {'atac': torch.bfloat16, 'atac_mask': torch.bool}
            case BundleName.DNASE:
                return {'dnase': torch.bfloat16, 'dnase_mask': torch.bool}
            case BundleName.PROCAP:
                return {'procap': torch.bfloat16, 'procap_mask': torch.bool}
            case BundleName.CAGE:
                return {'cage': torch.bfloat16, 'cage_mask': torch.bool}
            case BundleName.RNA_SEQ:
                return {
                    'rna_seq': torch.bfloat16,
                    'rna_seq_mask': torch.bool,
                    'rna_seq_strand': torch.int32,
                }
            case BundleName.CHIP_TF:
                return {'chip_tf': torch.float32, 'chip_tf_mask': torch.bool}
            case BundleName.CHIP_HISTONE:
                return {'chip_histone': torch.float32, 'chip_histone_mask': torch.bool}
            case BundleName.CONTACT_MAPS:
                return {'contact_maps': torch.float32}
            case BundleName.SPLICE_SITES_CLASSIFICATION:
                return {'splice_sites': torch.bool}
            case BundleName.SPLICE_SITES_USAGE:
                return {'splice_site_usage': torch.float16}
            case BundleName.SPLICE_SITES_JUNCTION:
                return {'splice_junctions': torch.float32}
            case BundleName.SPLICE_SITES_POSITIONS:
                return {'splice_site_positions': torch.int32}
            case _:
                raise ValueError(f'Unknown bundle name: {self}')

    def get_resolution(self) -> int:
        """Returns the resolution (in base pairs) for the given bundle."""
        match self:
            case (
                BundleName.ATAC
                | BundleName.DNASE
                | BundleName.PROCAP
                | BundleName.CAGE
                | BundleName.RNA_SEQ
                | BundleName.SPLICE_SITES_CLASSIFICATION
                | BundleName.SPLICE_SITES_USAGE
                | BundleName.SPLICE_SITES_JUNCTION
                | BundleName.SPLICE_SITES_POSITIONS
            ):
                return 1
            case BundleName.CHIP_TF | BundleName.CHIP_HISTONE:
                return 128
            case BundleName.CONTACT_MAPS:
                return 2048
            case _:
                raise ValueError(f'Unknown bundle name: {self}')

    def get_data_key(self) -> str:
        """Returns the primary data key for this bundle."""
        return self.value

    def get_mask_key(self) -> str | None:
        """Returns the mask key for this bundle, or None if no mask."""
        dtypes = self.get_dtypes()
        mask_key = f'{self.value}_mask'
        return mask_key if mask_key in dtypes else None
