"""
Data augmentation for bacterial genomic sequences
Based on AlphaGenome augmentation strategies adapted for bacterial data
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class BacterialSequenceAugmentation:
    """
    AlphaGenome-style data augmentation for bacterial sequences
    
    Implements two key augmentation strategies adapted for bacterial genomes:
    1. Shift augmentation: Random shifts of ±256bp (reduced from AlphaGenome's ±1024bp)
    2. Reverse complement: 50% probability of reverse complementing sequence
    
    Rationale for reduced shift range:
    - Bacterial regulatory elements are more compact than mammalian
    - Typical bacterial promoters: -35 to -10 regions (~25bp span)
    - Ribosome binding sites: ~8bp upstream of start codon
    - Operons: genes are tightly clustered with minimal intergenic regions
    - ±256bp covers typical bacterial regulatory contexts without losing biological relevance
    """
    
    def __init__(
        self,
        shift_range: int = 256,  # Reduced from AlphaGenome's 1024bp for bacterial regulatory scale
        reverse_complement_prob: float = 0.5,
        enable_shift: bool = True,
        enable_reverse_complement: bool = True,
        circular_genome: bool = True
    ):
        """
        Initialize augmentation
        
        Args:
            shift_range: Maximum shift distance in bp (±shift_range)
            reverse_complement_prob: Probability of reverse complement (0.0-1.0)
            enable_shift: Whether to apply shift augmentation
            enable_reverse_complement: Whether to apply reverse complement
            circular_genome: Whether to treat genome as circular (bacterial)
        """
        self.shift_range = shift_range
        self.reverse_complement_prob = reverse_complement_prob
        self.enable_shift = enable_shift
        self.enable_reverse_complement = enable_reverse_complement
        self.circular_genome = circular_genome
        
        # DNA encoding mappings
        self.dna_to_int = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.int_to_dna = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        
        # Reverse complement mapping for integer encoding
        self.complement_map = torch.tensor([1, 0, 3, 2, 4])  # A->T, T->A, G->C, C->G, N->N
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply augmentation to a training sample
        
        Args:
            sample: Dictionary containing 'dna', targets, and metadata
            
        Returns:
            Augmented sample with same structure
        """
        # Copy sample to avoid modifying original
        augmented_sample = sample.copy()
        
        # Extract DNA sequence tensor
        dna_sequence = sample['dna']  # [seq_len] tensor of integers
        seq_len = dna_sequence.shape[0]
        
        # Track augmentations applied
        augmentations_applied = []
        
        # 1. Shift augmentation
        shift_amount = 0
        if self.enable_shift:
            shift_amount = self._apply_shift_augmentation(augmented_sample)
            if shift_amount != 0:
                augmentations_applied.append(f"shift_{shift_amount}")
        
        # 2. Reverse complement augmentation  
        is_reverse_complemented = False
        if self.enable_reverse_complement:
            is_reverse_complemented = self._apply_reverse_complement_augmentation(augmented_sample)
            if is_reverse_complemented:
                augmentations_applied.append("reverse_complement")
        
        # Add augmentation metadata
        augmented_sample['augmentations'] = augmentations_applied
        augmented_sample['shift_amount'] = shift_amount
        augmented_sample['is_reverse_complemented'] = is_reverse_complemented
        
        return augmented_sample
    
    def _apply_shift_augmentation(self, sample: Dict[str, Any]) -> int:
        """
        Apply random shift augmentation to sequence and targets
        
        Args:
            sample: Sample dictionary to modify in-place
            
        Returns:
            Amount of shift applied (in bp)
        """
        if not self.enable_shift:
            return 0
            
        # Sample random shift from uniform distribution
        shift_amount = np.random.randint(-self.shift_range, self.shift_range + 1)
        
        if shift_amount == 0:
            return 0
        
        dna_sequence = sample['dna']
        seq_len = dna_sequence.shape[0]
        
        # Apply circular shift for bacterial genomes
        if self.circular_genome:
            # Circular shift - wrap around at genome boundaries
            shifted_sequence = torch.roll(dna_sequence, shift_amount)
            
            # Shift all target tensors correspondingly
            for key, target in sample.items():
                if key.startswith('target_') and isinstance(target, torch.Tensor):
                    if len(target.shape) >= 1:
                        # Shift targets at appropriate resolution
                        if 'gene_expression' in key or 'operon_membership' in key:
                            # 1bp resolution targets
                            shifted_target = torch.roll(target, shift_amount, dims=0)
                        elif 'gene_density' in key:
                            # 128bp resolution targets
                            shift_128bp = shift_amount // 128
                            if shift_128bp != 0:
                                shifted_target = torch.roll(target, shift_128bp, dims=0)
                            else:
                                shifted_target = target
                        else:
                            # Unknown resolution - try to infer from shape
                            target_len = target.shape[0]
                            if target_len == seq_len:
                                # 1bp resolution
                                shifted_target = torch.roll(target, shift_amount, dims=0)
                            elif target_len == seq_len // 128:
                                # 128bp resolution
                                shift_bins = shift_amount // 128
                                shifted_target = torch.roll(target, shift_bins, dims=0)
                            else:
                                # Unknown - don't shift
                                shifted_target = target
                        
                        sample[key] = shifted_target
        else:
            # Linear shift - pad with N's or truncate
            if shift_amount > 0:
                # Shift right - pad left with N's, truncate right
                padding = torch.full((shift_amount,), 4, dtype=dna_sequence.dtype)  # N = 4
                shifted_sequence = torch.cat([padding, dna_sequence[:-shift_amount]])
            else:
                # Shift left - pad right with N's, truncate left
                padding = torch.full((-shift_amount,), 4, dtype=dna_sequence.dtype)
                shifted_sequence = torch.cat([dna_sequence[-shift_amount:], padding])
            
            # For linear genomes, targets would need complex handling
            # For bacterial genomes, we typically use circular handling
        
        sample['dna'] = shifted_sequence
        return shift_amount
    
    def _apply_reverse_complement_augmentation(self, sample: Dict[str, Any]) -> bool:
        """
        Apply reverse complement augmentation with specified probability
        
        Args:
            sample: Sample dictionary to modify in-place
            
        Returns:
            Whether reverse complement was applied
        """
        if not self.enable_reverse_complement:
            return False
        
        # Random decision based on probability
        if np.random.random() > self.reverse_complement_prob:
            return False
        
        dna_sequence = sample['dna']
        
        # Apply reverse complement to DNA sequence
        # 1. Apply complement mapping first
        complement_sequence = self.complement_map[dna_sequence]
        # 2. Then reverse the sequence
        reverse_complement_sequence = torch.flip(complement_sequence, dims=[0])
        
        sample['dna'] = reverse_complement_sequence
        
        # Apply reverse complement to all target tensors
        for key, target in sample.items():
            if key.startswith('target_') and isinstance(target, torch.Tensor):
                if len(target.shape) >= 1:
                    # Reverse all targets along sequence dimension
                    reversed_target = torch.flip(target, dims=[0])
                    sample[key] = reversed_target
        
        return True
    
    def reverse_augmentation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reverse the augmentations applied to a sample (for evaluation)
        
        Args:
            sample: Augmented sample with augmentation metadata
            
        Returns:
            Sample with augmentations reversed
        """
        if 'augmentations' not in sample:
            return sample
        
        reversed_sample = sample.copy()
        augmentations = sample.get('augmentations', [])
        
        # Reverse in opposite order
        for aug in reversed(augmentations):
            if aug == 'reverse_complement':
                # Apply reverse complement again (it's self-inverse)
                self._apply_reverse_complement_augmentation(reversed_sample)
            elif aug.startswith('shift_'):
                # Apply opposite shift
                shift_amount = -sample.get('shift_amount', 0)
                if shift_amount != 0:
                    # Create temporary augmenter with opposite shift
                    temp_sample = {'dna': reversed_sample['dna']}
                    for key, value in reversed_sample.items():
                        if key.startswith('target_'):
                            temp_sample[key] = value
                    
                    # Apply opposite shift
                    if self.circular_genome:
                        shifted_dna = torch.roll(temp_sample['dna'], shift_amount)
                        temp_sample['dna'] = shifted_dna
                        
                        for key, target in temp_sample.items():
                            if key.startswith('target_'):
                                if 'gene_expression' in key or 'operon_membership' in key:
                                    temp_sample[key] = torch.roll(target, shift_amount, dims=0)
                                elif 'gene_density' in key:
                                    shift_bins = shift_amount // 128
                                    if shift_bins != 0:
                                        temp_sample[key] = torch.roll(target, shift_bins, dims=0)
                    
                    # Update reversed sample
                    for key, value in temp_sample.items():
                        reversed_sample[key] = value
        
        # Remove augmentation metadata
        reversed_sample.pop('augmentations', None)
        reversed_sample.pop('shift_amount', None)
        reversed_sample.pop('is_reverse_complemented', None)
        
        return reversed_sample


def create_augmentation_transform(
    shift_range: int = 256,  # Bacterial-adapted default
    reverse_complement_prob: float = 0.5,
    enable_shift: bool = True,
    enable_reverse_complement: bool = True,
    circular_genome: bool = True
) -> BacterialSequenceAugmentation:
    """
    Factory function to create augmentation transform
    
    Args:
        shift_range: Maximum shift distance in bp
        reverse_complement_prob: Probability of reverse complement
        enable_shift: Whether to enable shift augmentation
        enable_reverse_complement: Whether to enable reverse complement
        circular_genome: Whether genome is circular (typical for bacteria)
        
    Returns:
        Configured augmentation transform
    """
    return BacterialSequenceAugmentation(
        shift_range=shift_range,
        reverse_complement_prob=reverse_complement_prob,
        enable_shift=enable_shift,
        enable_reverse_complement=enable_reverse_complement,
        circular_genome=circular_genome
    )


class AugmentationWrapper:
    """
    Wrapper to integrate augmentation with existing datasets
    """
    
    def __init__(self, dataset, augmentation_transform: Optional[BacterialSequenceAugmentation] = None):
        """
        Initialize wrapper
        
        Args:
            dataset: Base dataset to wrap
            augmentation_transform: Augmentation to apply (None for no augmentation)
        """
        self.dataset = dataset
        self.augmentation = augmentation_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original sample
        sample = self.dataset[idx]
        
        # Apply augmentation if enabled
        if self.augmentation is not None:
            sample = self.augmentation(sample)
        
        return sample
    
    def enable_augmentation(self):
        """Enable augmentation"""
        if self.augmentation is not None:
            self.augmentation.enable_shift = True
            self.augmentation.enable_reverse_complement = True
    
    def disable_augmentation(self):
        """Disable augmentation (e.g., for validation)"""
        if self.augmentation is not None:
            self.augmentation.enable_shift = False
            self.augmentation.enable_reverse_complement = False


def test_augmentation():
    """Test function to verify augmentation works correctly"""
    # Create test sample
    seq_len = 1000
    test_sample = {
        'dna': torch.randint(0, 5, (seq_len,)),
        'target_gene_expression': torch.randn(seq_len, 1),
        'target_gene_density': torch.randn(seq_len // 128, 1),
        'target_operon_membership': torch.randint(0, 2, (seq_len, 1)).float(),
        'organism_index': torch.tensor(0),
    }
    
    # Create augmentation
    augmentation = create_augmentation_transform(
        shift_range=100,
        reverse_complement_prob=1.0,  # Always apply for testing
    )
    
    # Apply augmentation
    augmented = augmentation(test_sample)
    
    print("Augmentation test results:")
    print(f"Original DNA shape: {test_sample['dna'].shape}")
    print(f"Augmented DNA shape: {augmented['dna'].shape}")
    print(f"Augmentations applied: {augmented.get('augmentations', [])}")
    
    # Test reverse
    reversed_sample = augmentation.reverse_augmentation(augmented)
    print(f"Reversibility test - DNA matches: {torch.equal(test_sample['dna'], reversed_sample['dna'])}")
    
    return True


if __name__ == "__main__":
    test_augmentation()