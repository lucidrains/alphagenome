# Ported from alphagenome_research reference implementation.
"""Batch data structures for AlphaGenome PyTorch.

This module defines the DataBatch dataclass that holds batched training data
with torch tensors, matching the schema from the reference implementation.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import torch

from alphagenome_pytorch.data.bundles import BundleName


@dataclass
class DataBatch:
    """Input batch for the AlphaGenome model.

    This dataclass holds all possible inputs for training/inference.
    All tensors are optional and may be None if not present in the batch.

    Shapes:
        B: batch size
        S: sequence length (DNA length)
        S_DNA: DNA sequence length (same as S)
        C_*: number of channels for each bundle type
        P: number of splice site positions
    """

    # Core inputs
    dna_sequence: Optional[torch.Tensor] = None  # [B, S_DNA, 4] float32
    organism_index: Optional[torch.Tensor] = None  # [B] int32

    # ATAC-seq data (1bp resolution)
    atac: Optional[torch.Tensor] = None  # [B, S, C_ATAC] bfloat16
    atac_mask: Optional[torch.Tensor] = None  # [B, S, C_ATAC] bool

    # DNase-seq data (1bp resolution)
    dnase: Optional[torch.Tensor] = None  # [B, S, C_DNASE] bfloat16
    dnase_mask: Optional[torch.Tensor] = None  # [B, S, C_DNASE] bool

    # PRO-cap data (1bp resolution)
    procap: Optional[torch.Tensor] = None  # [B, S, C_PROCAP] bfloat16
    procap_mask: Optional[torch.Tensor] = None  # [B, S, C_PROCAP] bool

    # CAGE data (1bp resolution)
    cage: Optional[torch.Tensor] = None  # [B, S, C_CAGE] bfloat16
    cage_mask: Optional[torch.Tensor] = None  # [B, S, C_CAGE] bool

    # RNA-seq data (1bp resolution)
    rna_seq: Optional[torch.Tensor] = None  # [B, S, C_RNA_SEQ] bfloat16
    rna_seq_mask: Optional[torch.Tensor] = None  # [B, S, C_RNA_SEQ] bool
    rna_seq_strand: Optional[torch.Tensor] = None  # [B, 1, C_RNA_SEQ] int32

    # ChIP-seq TF data (128bp resolution)
    chip_tf: Optional[torch.Tensor] = None  # [B, S//128, C_CHIP_TF] float32
    chip_tf_mask: Optional[torch.Tensor] = None  # [B, S//128, C_CHIP_TF] bool

    # ChIP-seq histone data (128bp resolution)
    chip_histone: Optional[torch.Tensor] = None  # [B, S//128, C_CHIP_HISTONE] float32
    chip_histone_mask: Optional[torch.Tensor] = None  # [B, S//128, C_CHIP_HISTONE] bool

    # Contact maps (2048bp resolution)
    contact_maps: Optional[torch.Tensor] = None  # [B, S//2048, S//2048, C_CONTACT_MAPS] float32

    # Splice site data
    splice_sites: Optional[torch.Tensor] = None  # [B, S, C_SPLICE_SITES] bool
    splice_site_usage: Optional[torch.Tensor] = None  # [B, S, C_SPLICE_SITE_USAGE] float16
    splice_junctions: Optional[torch.Tensor] = None  # [B, P, P, C_SPLICE_JUNCTIONS] float32
    splice_site_positions: Optional[torch.Tensor] = None  # [B, 4, P] int32

    def get_organism_index(self) -> torch.Tensor:
        """Returns the organism index data.

        Raises:
            ValueError: If organism index is not present in the batch.
        """
        if self.organism_index is None:
            raise ValueError('Organism index data is not present in the batch.')
        return self.organism_index

    def get_genome_tracks(
        self, bundle: BundleName
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the genome tracks data and mask for the given bundle.

        Args:
            bundle: The bundle type to retrieve data for.

        Returns:
            Tuple of (data, mask) tensors.

        Raises:
            ValueError: If the bundle is not a genome tracks bundle or data is missing.
        """
        match bundle:
            case BundleName.ATAC:
                data, mask = self.atac, self.atac_mask
            case BundleName.DNASE:
                data, mask = self.dnase, self.dnase_mask
            case BundleName.PROCAP:
                data, mask = self.procap, self.procap_mask
            case BundleName.CAGE:
                data, mask = self.cage, self.cage_mask
            case BundleName.RNA_SEQ:
                data, mask = self.rna_seq, self.rna_seq_mask
            case BundleName.CHIP_TF:
                data, mask = self.chip_tf, self.chip_tf_mask
            case BundleName.CHIP_HISTONE:
                data, mask = self.chip_histone, self.chip_histone_mask
            case _:
                raise ValueError(
                    f'Unknown bundle name: {bundle!r}. Is it a genome tracks bundle?'
                )

        if data is None or mask is None:
            raise ValueError(f'{bundle.name!r} data is not present in the batch.')
        return data, mask

    def to(self, device: torch.device) -> 'DataBatch':
        """Move all tensors to the specified device.

        Args:
            device: The target device.

        Returns:
            A new DataBatch with all tensors on the specified device.
        """
        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                kwargs[f.name] = value.to(device)
            else:
                kwargs[f.name] = value
        return DataBatch(**kwargs)

    def pin_memory(self) -> 'DataBatch':
        """Pin memory for all tensors (for faster CPU to GPU transfer).

        Returns:
            A new DataBatch with all tensors pinned in memory.
        """
        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                kwargs[f.name] = value.pin_memory()
            else:
                kwargs[f.name] = value
        return DataBatch(**kwargs)


def collate_batch(samples: List[Dict[str, Any]]) -> DataBatch:
    """Collate function for PyTorch DataLoader.

    Takes a list of sample dictionaries and stacks them into a single DataBatch.

    Args:
        samples: List of dictionaries, each containing sample data.
            Keys should match DataBatch field names.

    Returns:
        A DataBatch containing stacked tensors from all samples.
    """
    if not samples:
        return DataBatch()

    # Get all unique keys from samples
    all_keys = set()
    for sample in samples:
        all_keys.update(sample.keys())

    # Stack tensors for each key
    collated: Dict[str, Any] = {}
    for key in all_keys:
        values = [sample.get(key) for sample in samples]

        # Skip if all values are None
        if all(v is None for v in values):
            collated[key] = None
            continue

        # Handle tensors
        if isinstance(values[0], torch.Tensor):
            # Stack along batch dimension
            collated[key] = torch.stack(values, dim=0)
        else:
            # For non-tensor values, just take the first one
            # (assumes all samples have the same value for non-tensor fields)
            collated[key] = values[0]

    # Create DataBatch with only valid fields
    valid_field_names = {f.name for f in fields(DataBatch)}
    filtered_collated = {k: v for k, v in collated.items() if k in valid_field_names}

    return DataBatch(**filtered_collated)


def dict_to_batch(data: Dict[str, Any], organism_index: int = 0) -> DataBatch:
    """Convert a dictionary of numpy arrays or tensors to a DataBatch.

    This is useful for converting raw TFRecord data to a DataBatch.

    Args:
        data: Dictionary mapping field names to arrays/tensors.
        organism_index: The organism index to use if not present in data.

    Returns:
        A DataBatch instance.
    """
    batch_data: Dict[str, Any] = {}
    valid_field_names = {f.name for f in fields(DataBatch)}

    for key, value in data.items():
        if key not in valid_field_names:
            continue

        if value is None:
            batch_data[key] = None
        elif isinstance(value, torch.Tensor):
            batch_data[key] = value
        else:
            # Convert numpy array to tensor
            batch_data[key] = torch.from_numpy(value)

    # Set organism index if not present
    if 'organism_index' not in batch_data or batch_data['organism_index'] is None:
        batch_data['organism_index'] = torch.tensor([organism_index], dtype=torch.int32)

    return DataBatch(**batch_data)
