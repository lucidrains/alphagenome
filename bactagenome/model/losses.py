"""
Loss functions for bacterial genome modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class PromoterLoss(nn.Module):
    """Loss for promoter strength prediction"""
    
    def __init__(self, multinomial_weight: float = 1.0, poisson_weight: float = 0.1):
        super().__init__()
        self.multinomial_weight = multinomial_weight
        self.poisson_weight = poisson_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, num_conditions]
            target: [batch, seq_len, num_conditions]
        """
        # Multinomial loss for ratios
        pred_ratios = F.softmax(pred, dim=-1)
        target_ratios = F.softmax(target, dim=-1)
        multinomial_loss = F.kl_div(pred_ratios.log(), target_ratios, reduction='batchmean')
        
        # Poisson loss for counts
        pred_counts = pred.sum(dim=-1)
        target_counts = target.sum(dim=-1)
        poisson_loss = F.poisson_nll_loss(pred_counts, target_counts, log_input=False)
        
        return self.multinomial_weight * multinomial_loss + self.poisson_weight * poisson_loss


class RBSLoss(nn.Module):
    """Loss for RBS efficiency prediction"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, 1]
            target: [batch, seq_len, 1]
        """
        # MSE on log-transformed efficiency ratios
        log_pred = torch.log(pred + 1e-8)
        log_target = torch.log(target + 1e-8)
        return F.mse_loss(log_pred, log_target)


class OperonLoss(nn.Module):
    """Loss for operon co-regulation prediction"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, num_genes]
            target: [batch, seq_len, num_genes]
        """
        # Correlation loss between genes in same operon
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        # Compute correlation matrices
        pred_corr = torch.bmm(pred_norm.transpose(1, 2), pred_norm)
        target_corr = torch.bmm(target_norm.transpose(1, 2), target_norm)
        
        return F.mse_loss(pred_corr, target_corr)


class RiboswitchLoss(nn.Module):
    """Loss for riboswitch binding prediction"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, num_ligands]
            target: [batch, seq_len, num_ligands]
        """
        return F.binary_cross_entropy(pred, target)


class SRNALoss(nn.Module):
    """Loss for sRNA target prediction"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, num_srnas]
            target: [batch, seq_len, num_srnas]
        """
        # Ranking loss for target prioritization
        return F.mse_loss(pred, target)


class TerminationLoss(nn.Module):
    """Loss for transcription termination prediction"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, 3]
            target: [batch, seq_len, 3]
        """
        return F.cross_entropy(pred.view(-1, 3), target.view(-1, 3).argmax(dim=-1))


class PathwayLoss(nn.Module):
    """Loss for pathway activity prediction"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, num_pathways]
            target: [batch, num_pathways]
        """
        return F.binary_cross_entropy(pred, target)


class SecretionLoss(nn.Module):
    """Loss for secretion signal prediction"""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, seq_len, num_secretion_types]
            target: [batch, seq_len, num_secretion_types]
        """
        return F.binary_cross_entropy(pred, target)


class BacterialLossCollection(nn.Module):
    """Collection of all bacterial-specific losses"""
    
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        
        self.loss_weights = loss_weights or {
            'promoter_strength': 1.0,
            'rbs_efficiency': 1.0,
            'operon_coregulation': 1.0,
            'riboswitch_binding': 0.5,
            'srna_target': 0.5,
            'termination': 0.5,
            'pathway_activity': 0.5,
            'secretion_signal': 0.5,
        }
        
        self.losses = nn.ModuleDict({
            'promoter_strength': PromoterLoss(),
            'rbs_efficiency': RBSLoss(),
            'operon_coregulation': OperonLoss(),
            'riboswitch_binding': RiboswitchLoss(),
            'srna_target': SRNALoss(),
            'termination': TerminationLoss(),
            'pathway_activity': PathwayLoss(),
            'secretion_signal': SecretionLoss(),
        })
    
    def forward(
        self, 
        predictions: Dict[str, Dict[str, torch.Tensor]], 
        targets: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all modalities
        
        Args:
            predictions: Nested dict [organism][modality] -> tensor
            targets: Nested dict [organism][modality] -> tensor
        
        Returns:
            Dictionary of losses by modality
        """
        total_losses = {}
        
        for organism in predictions:
            for modality, pred in predictions[organism].items():
                if modality not in targets[organism]:
                    continue
                
                target = targets[organism][modality]
                loss_fn = self.losses[modality]
                loss = loss_fn(pred, target)
                
                # Weight the loss
                weighted_loss = loss * self.loss_weights[modality]
                
                if modality not in total_losses:
                    total_losses[modality] = weighted_loss
                else:
                    total_losses[modality] += weighted_loss
        
        # Add total loss
        total_losses['total'] = sum(total_losses.values())
        
        return total_losses