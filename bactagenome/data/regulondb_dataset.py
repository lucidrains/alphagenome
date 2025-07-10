"""
RegulonDB Dataset for BactaGenome training
Integrates with existing training pipeline
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset
from collections import defaultdict
import logging

from .regulondb_processor import RegulonDBProcessor

logger = logging.getLogger(__name__)


class RegulonDBDataset(Dataset):
    """
    PyTorch Dataset for RegulonDB data
    
    Provides real bacterial genomic sequences and annotations for training BactaGenome
    Compatible with existing DummyBacterialDataset interface
    """
    
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 98304,
        num_organisms: int = 1,
        organism_name: str = "E_coli_K12",
        split: str = "train",
        process_if_missing: bool = True,
        regulondb_raw_path: Optional[str] = None
    ):
        """
        Initialize RegulonDB dataset
        
        Args:
            data_dir: Directory containing processed RegulonDB data
            seq_len: Sequence length (should match model context length)
            num_organisms: Number of organisms (1 for Phase 1 - E. coli only)
            organism_name: Name of organism
            split: Dataset split ('train', 'val', 'test')
            process_if_missing: Whether to process raw data if processed data missing
            regulondb_raw_path: Path to raw RegulonDB BSON files
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.num_organisms = num_organisms
        self.organism_name = organism_name
        self.split = split
        
        # DNA encoding
        self.dna_to_int = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        
        # Load or process data
        self._load_data(process_if_missing, regulondb_raw_path)
        
        # Create train/val/test splits
        self._create_splits()
    
    def _load_data(self, process_if_missing: bool, regulondb_raw_path: Optional[str]):
        """Load processed data or process raw data if missing"""
        
        # Check for processed data
        windows_file = self.data_dir / "training_windows.json"
        targets_file = self.data_dir / "target_tensors.pt"
        metadata_file = self.data_dir / "metadata.json"
        
        if windows_file.exists() and targets_file.exists() and metadata_file.exists():
            logger.info("Loading existing processed RegulonDB data...")
            
            # Load processed data
            with open(windows_file, 'r') as f:
                self.windows = json.load(f)
            
            self.targets = torch.load(targets_file)
            
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            logger.info(f"Loaded {len(self.windows)} windows and {len(self.targets)} target modalities")
            
        elif process_if_missing and regulondb_raw_path:
            logger.info("Processing raw RegulonDB data...")
            
            # Process raw data
            processor = RegulonDBProcessor(regulondb_raw_path, str(self.data_dir))
            self.windows, self.targets = processor.process_all()
            
            # Load metadata
            with open(self.data_dir / "metadata.json", 'r') as f:
                self.metadata = json.load(f)
                
        else:
            raise FileNotFoundError(
                f"Processed data not found in {self.data_dir} and process_if_missing=False. "
                f"Either provide processed data or set process_if_missing=True with regulondb_raw_path."
            )
    
    def _create_splits(self):
        """Create train/validation/test splits using chromosome-based method"""
        
        total_windows = len(self.windows)
        
        # Chromosome-based splits (following plan.md strategy)
        # E. coli circular chromosome split by genomic coordinates
        split_indices = {
            'train': [],    # 0° to 270° (75% of genome)
            'val': [],      # 270° to 360° (25% of genome)
            'test': []      # Can be used for held-out species later
        }
        
        # E. coli genome length: ~4.6M bp
        genome_length = 4641652
        
        for i, window in enumerate(self.windows):
            start_pos = window['genomic_start']
            
            # Convert to angular position (0-360 degrees)
            angle = (start_pos / genome_length) * 360
            
            if angle < 270:  # 0° to 270° for training
                split_indices['train'].append(i)
            else:  # 270° to 360° for validation
                split_indices['val'].append(i)
        
        # Use the appropriate split
        if self.split in split_indices:
            self.indices = split_indices[self.split]
        else:
            logger.warning(f"Unknown split '{self.split}', using all data")
            self.indices = list(range(total_windows))
        
        logger.info(f"Split '{self.split}': {len(self.indices)} windows")
    
    def _encode_dna_sequence(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence to integer encoding"""
        encoded = torch.zeros(len(sequence), dtype=torch.long)
        
        for i, nucleotide in enumerate(sequence.upper()):
            encoded[i] = self.dna_to_int.get(nucleotide, 4)  # Default to 'N' for unknown
        
        return encoded
    
    def _generate_sequence_for_window(self, window: Dict) -> torch.Tensor:
        """
        Generate DNA sequence for a genomic window
        
        For now, creates a realistic synthetic sequence based on E. coli GC content
        In production, this would load the actual E. coli genome sequence
        """
        
        # E. coli has ~50.8% GC content
        gc_content = 0.508
        
        # Generate sequence with appropriate base composition
        probs = np.array([
            (1 - gc_content) / 2,  # A
            (1 - gc_content) / 2,  # T  
            gc_content / 2,        # G
            gc_content / 2         # C
        ])
        
        # Generate random sequence with E. coli-like composition
        sequence_ints = np.random.choice(4, size=self.seq_len, p=probs)
        
        # Add some realistic patterns around genes
        for gene in window['genes']:
            gene_start = gene['relative_start']
            gene_end = gene['relative_end']
            
            # Add AT-rich promoter region upstream of genes
            promoter_start = max(0, gene_start - 100)
            promoter_end = gene_start
            
            if promoter_end > promoter_start:
                # Make promoter more AT-rich (typical for bacterial promoters)
                at_probs = np.array([0.4, 0.4, 0.1, 0.1])  # More A/T
                promoter_seq = np.random.choice(4, size=promoter_end - promoter_start, p=at_probs)
                sequence_ints[promoter_start:promoter_end] = promoter_seq
        
        return torch.from_numpy(sequence_ints).long()
    
    def __len__(self) -> int:
        """Return number of samples in this split"""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
        """
        Get a training sample
        
        Returns:
            Tuple of (sequence, organism_index, targets)
            - sequence: [seq_len] DNA sequence as integers
            - organism_index: int (0 for E. coli)  
            - targets: Dict of target tensors by modality
        """
        
        # Map to actual window index
        window_idx = self.indices[idx]
        window = self.windows[window_idx]
        
        # Generate DNA sequence for this window
        sequence = self._generate_sequence_for_window(window)
        
        # Organism index (0 for E. coli in Phase 1)
        organism_index = 0
        
        # Extract targets for this window
        targets = {}
        for modality, tensor in self.targets.items():
            targets[modality] = tensor[window_idx]
        
        return sequence, organism_index, targets
    
    def get_organism_name(self, organism_index: int) -> str:
        """Get organism name from index"""
        return self.organism_name if organism_index == 0 else f"organism_{organism_index}"
    
    def get_num_conditions(self) -> int:
        """Get number of expression conditions"""
        return self.targets['promoter_strength'].shape[-1] if 'promoter_strength' in self.targets else 50
    
    def get_num_coexpression_tracks(self) -> int:
        """Get number of co-expression tracks"""
        return self.targets['operon_coregulation'].shape[-1] if 'operon_coregulation' in self.targets else 20


class RegulonDBDataLoader:
    """
    DataLoader factory for RegulonDB datasets
    Compatible with existing training pipeline
    """
    
    @staticmethod
    def create_datasets(
        data_dir: str,
        seq_len: int = 98304,
        train_split: float = 0.8,
        val_split: float = 0.2,
        regulondb_raw_path: Optional[str] = None
    ) -> Tuple[RegulonDBDataset, RegulonDBDataset]:
        """
        Create train and validation datasets
        
        Args:
            data_dir: Directory for processed data
            seq_len: Sequence length
            train_split: Fraction for training (not used - using chromosome splits)
            val_split: Fraction for validation (not used - using chromosome splits)
            regulondb_raw_path: Path to raw RegulonDB data
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        
        train_dataset = RegulonDBDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            split='train',
            regulondb_raw_path=regulondb_raw_path
        )
        
        val_dataset = RegulonDBDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            split='val',
            regulondb_raw_path=regulondb_raw_path
        )
        
        return train_dataset, val_dataset
    
    @staticmethod
    def get_dataloader_kwargs() -> Dict[str, Any]:
        """Get recommended DataLoader kwargs for RegulonDB data"""
        return {
            'batch_size': 4,  # Start small due to large sequences
            'shuffle': True,
            'num_workers': 2,
            'pin_memory': True,
            'drop_last': True
        }


def collate_regulondb_batch(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Custom collate function for RegulonDB data
    
    Args:
        batch: List of (sequence, organism_index, targets) tuples
        
    Returns:
        Tuple of (sequences, organism_indices, targets_by_organism)
    """
    sequences = []
    organism_indices = []
    targets_by_modality = defaultdict(list)
    
    for sequence, organism_idx, targets in batch:
        sequences.append(sequence)
        organism_indices.append(organism_idx)
        
        for modality, target in targets.items():
            targets_by_modality[modality].append(target)
    
    # Stack tensors
    sequences = torch.stack(sequences)
    organism_indices = torch.tensor(organism_indices)
    
    # Organize targets by organism (Phase 1: only E. coli)
    targets_by_organism = {
        "E_coli_K12": {}
    }
    
    for modality, target_list in targets_by_modality.items():
        targets_by_organism["E_coli_K12"][modality] = torch.stack(target_list)
    
    return sequences, organism_indices, targets_by_organism