"""
Loss functions for bacterial genome modeling
Based on AlphaGenome loss function patterns adapted for bacterial data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


def soft_clip(x: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
    """
    AlphaGenome-style soft clipping for numerical stability
    """
    return torch.where(x > threshold, 2 * torch.sqrt(x * threshold) - threshold, x)


def multinomial_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    multinomial_resolution: int,
    segment_weight: float = 5.0
) -> torch.Tensor:
    """
    AlphaGenome-style multinomial loss over segments
    
    Args:
        pred: [batch, seq_len, tracks] - Predictions
        target: [batch, seq_len, tracks] - Targets  
        multinomial_resolution: Size of each segment
        segment_weight: Weight for multinomial component (default 5.0 like AlphaGenome)
    
    Returns:
        Combined Poisson + Multinomial loss
    """
    # Reshape into segments
    batch_size, seq_len, tracks = pred.shape
    num_segments = seq_len // multinomial_resolution
    
    if num_segments == 0:
        # Fallback for short sequences
        return F.poisson_nll_loss(pred.sum(dim=1), target.sum(dim=1), log_input=False)
    
    # Reshape to [batch, num_segments, segment_size, tracks]
    pred_segments = pred[:, :num_segments * multinomial_resolution].reshape(
        batch_size, num_segments, multinomial_resolution, tracks
    )
    target_segments = target[:, :num_segments * multinomial_resolution].reshape(
        batch_size, num_segments, multinomial_resolution, tracks
    )
    
    # Sum over each segment: [batch, num_segments, tracks]
    sum_pred = pred_segments.sum(dim=2)
    sum_target = target_segments.sum(dim=2)
    
    # Poisson loss on segment sums (encourages correct total counts)
    poisson_loss = F.poisson_nll_loss(
        sum_pred.flatten(), 
        sum_target.flatten(), 
        log_input=False,
        reduction='mean'
    )
    poisson_loss = poisson_loss / multinomial_resolution  # Scale by segment size
    
    # Multinomial loss on distributions within segments (encourages correct shape)
    pred_probs = pred_segments / (sum_pred.unsqueeze(2) + 1e-7)  # Normalize to probabilities
    multinomial_nll = -(target_segments * torch.log(pred_probs + 1e-7)).sum()
    multinomial_loss = multinomial_nll / (batch_size * num_segments * tracks)
    
    return poisson_loss + segment_weight * multinomial_loss


class PromoterStrengthLoss(nn.Module):
    """
    Promoter strength prediction loss - AlphaGenome style
    
    Based on AlphaGenome's RNA-seq loss: Poisson + Multinomial over segments
    Adapted for bacterial gene expression across conditions
    """
    
    def __init__(self, multinomial_resolution: int = 217, segment_weight: float = 5.0):
        super().__init__()
        self.multinomial_resolution = multinomial_resolution
        self.segment_weight = segment_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, num_conditions] - Predicted expression levels
            target: [batch, seq_len, num_conditions] - Target expression levels
        
        Returns:
            Combined loss encouraging correct counts and spatial distribution
        """
        return multinomial_loss(pred, target, self.multinomial_resolution, self.segment_weight)


class RBSEfficiencyLoss(nn.Module):
    """
    RBS efficiency prediction loss
    
    MSE on log-transformed efficiency ratios (similar to AlphaGenome scaling)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, 1] - Predicted RBS efficiency
            target: [batch, seq_len, 1] - Target RBS efficiency
        
        Returns:
            MSE loss on log-transformed values
        """
        # Log transform for better numerical properties
        log_pred = torch.log(pred + 1e-7)
        log_target = torch.log(target + 1e-7)
        
        return F.mse_loss(log_pred, log_target)


class OperonCoregulationLoss(nn.Module):
    """
    Operon co-regulation prediction loss
    
    Uses AlphaGenome-style multinomial loss for co-expression tracks
    """
    
    def __init__(self, multinomial_resolution: int = 128):
        super().__init__()
        self.multinomial_resolution = multinomial_resolution
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len//128, num_tracks] - Predicted co-expression
            target: [batch, seq_len//128, num_tracks] - Target co-expression
        
        Returns:
            Multinomial loss for co-expression patterns
        """
        # Use smaller resolution for gene-level features
        resolution = min(self.multinomial_resolution, pred.shape[1])
        return multinomial_loss(pred, target, resolution, segment_weight=5.0)


# Advanced regulation losses (Phase 2)

class RiboswitchBindingLoss(nn.Module):
    """
    Riboswitch ligand binding prediction loss
    Binary cross-entropy for binding probabilities
    """
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, num_ligands] - Predicted binding probabilities
            target: [batch, seq_len, num_ligands] - Target binding labels
        """
        return F.binary_cross_entropy(pred, target)


class SRNATargetLoss(nn.Module):
    """
    sRNA target prediction loss
    Ranking loss for target prioritization (following AlphaGenome's approach to ordering)
    """
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, num_srnas] - Predicted interaction strength
            target: [batch, seq_len, num_srnas] - Target interaction strength
        """
        # Use MSE as ranking loss proxy (higher scores = stronger interactions)
        return F.mse_loss(pred, target)


# Systems-level losses (Phase 3)

class TerminationLoss(nn.Module):
    """
    Transcription termination prediction loss
    Cross-entropy for termination type classification
    """
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, 3] - Predicted termination probabilities
            target: [batch, seq_len, 3] - Target termination labels (one-hot)
        """
        pred_flat = pred.view(-1, pred.shape[-1])
        target_flat = target.view(-1, target.shape[-1])
        
        # Convert one-hot to class indices
        target_indices = target_flat.argmax(dim=-1)
        
        return F.cross_entropy(pred_flat, target_indices)


class PathwayActivityLoss(nn.Module):
    """
    Pathway activity prediction loss
    Binary cross-entropy for pathway completeness
    """
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, num_pathways] - Predicted pathway activity
            target: [batch, num_pathways] - Target pathway activity
        """
        return F.binary_cross_entropy(pred, target)


class SecretionSignalLoss(nn.Module):
    """
    Secretion signal prediction loss
    Multi-label binary cross-entropy for secretion systems
    """
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, num_secretion_types] - Predicted secretion probabilities
            target: [batch, seq_len, num_secretion_types] - Target secretion labels
        """
        return F.binary_cross_entropy(pred, target)


class BacterialLossFunction(nn.Module):
    """
    Complete loss function for bacterial genome modeling
    
    Combines all modality-specific losses with AlphaGenome-style weighting
    (AlphaGenome uses uniform weighting across all heads - no additional coefficients)
    """
    
    def __init__(self, loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        # Default weights: Phase 1 modalities weighted higher
        self.loss_weights = loss_weights or {
            # Phase 1: Core expression control (full weight)
            'promoter_strength': 1.0,
            'rbs_efficiency': 1.0, 
            'operon_coregulation': 1.0,
            
            # Phase 2: Advanced regulation (reduced weight)
            'riboswitch_binding': 0.8,
            'srna_targets': 0.8,
            
            # Phase 3: Systems-level (full weight)
            'transcription_termination': 1.0,
            'pathway_activity': 1.0,
            'secretion_signals': 1.0,
        }
        
        # Loss function registry
        self.loss_functions = nn.ModuleDict({
            'promoter_strength': PromoterStrengthLoss(),
            'rbs_efficiency': RBSEfficiencyLoss(),
            'operon_coregulation': OperonCoregulationLoss(),
            'riboswitch_binding': RiboswitchBindingLoss(),
            'srna_targets': SRNATargetLoss(),
            'transcription_termination': TerminationLoss(),
            'pathway_activity': PathwayActivityLoss(),
            'secretion_signals': SecretionSignalLoss(),
        })
    
    def forward(
        self, 
        predictions: Dict[str, Dict[str, torch.Tensor]], 
        targets: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all active modalities
        
        Args:
            predictions: Nested dict [organism][modality] -> tensor
            targets: Nested dict [organism][modality] -> tensor
        
        Returns:
            Dictionary containing:
            - Individual modality losses
            - Total weighted loss
        """
        modality_losses = {}
        total_loss = 0.0
        
        for organism in predictions:
            if organism not in targets:
                continue
                
            for modality, pred in predictions[organism].items():
                if modality not in targets[organism]:
                    continue
                
                if modality not in self.loss_functions:
                    continue
                
                # Compute modality-specific loss
                target = targets[organism][modality]
                loss_fn = self.loss_functions[modality]
                loss = loss_fn(pred, target)
                
                # Apply weighting
                weight = self.loss_weights.get(modality, 1.0)
                weighted_loss = loss * weight
                
                # Accumulate losses
                if modality in modality_losses:
                    modality_losses[modality] += weighted_loss
                else:
                    modality_losses[modality] = weighted_loss
                
                total_loss += weighted_loss
        
        # Return all losses for monitoring
        result = modality_losses.copy()
        result['total_loss'] = total_loss
        
        return result


# Alias for backward compatibility
BacterialLossCollection = BacterialLossFunction