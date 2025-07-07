"""
Loss functions for bacterial genome modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BacterialLossFunction(nn.Module):
    """
    Combined loss function for Phase 1 bacterial modalities
    """
    
    def __init__(self, loss_weights=None):
        super().__init__()
        
        # Default loss weights for Phase 1 modalities
        self.loss_weights = loss_weights or {
            'promoter_strength': 1.0,
            'rbs_efficiency': 1.0,
            'operon_coregulation': 1.0,
        }
        
        # Loss functions for Phase 1 modalities
        self.loss_functions = {
            'promoter_strength': nn.MSELoss(),  # Regression for expression levels
            'rbs_efficiency': nn.MSELoss(),     # Regression for translation rates
            'operon_coregulation': nn.MSELoss(), # Regression for correlation values
        }
        
    def forward(self, predictions, targets, organism_name):
        """
        Compute combined loss for all modalities
        
        Args:
            predictions: Dict of model predictions by modality
            targets: Dict of target values by modality
            organism_name: Name of the organism
            
        Returns:
            Total loss and individual losses
        """
        total_loss = 0.0
        individual_losses = {}
        
        for modality, pred in predictions.items():
            if modality not in targets:
                continue
                
            target = targets[modality]
            loss_fn = self.loss_functions[modality]
            weight = self.loss_weights.get(modality, 1.0)
            
            # Apply appropriate loss function for Phase 1 modalities
            if modality in ['promoter_strength', 'rbs_efficiency', 'operon_coregulation']:
                # All Phase 1 modalities are regression tasks
                loss = loss_fn(pred, target)
            else:
                # Default to MSE for any unexpected modalities
                loss = nn.MSELoss()(pred, target)
                
            individual_losses[modality] = loss.item()
            total_loss += weight * loss
            
        return total_loss, individual_losses


class PromoterStrengthLoss(nn.Module):
    """Specialized loss for promoter strength prediction"""
    
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        # Base MSE loss
        mse = self.mse_loss(pred, target)
        
        # Add smoothness penalty
        pred_diff = pred[1:] - pred[:-1]
        smoothness_penalty = torch.mean(pred_diff ** 2)
        
        return mse + self.alpha * smoothness_penalty


class RBSEfficiencyLoss(nn.Module):
    """Specialized loss for RBS efficiency prediction"""
    
    def __init__(self, log_transform=True):
        super().__init__()
        self.log_transform = log_transform
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        if self.log_transform:
            # Work in log space for better numerical stability
            pred_log = torch.log(pred + 1e-8)
            target_log = torch.log(target + 1e-8)
            return self.mse_loss(pred_log, target_log)
        else:
            return self.mse_loss(pred, target)


class OperonCoregulationLoss(nn.Module):
    """Specialized loss for operon co-regulation prediction"""
    
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, pred, target):
        return self.ce_loss(pred, target.long())


class PathwayActivityLoss(nn.Module):
    """Specialized loss for pathway activity prediction"""
    
    def __init__(self, pathway_weights=None):
        super().__init__()
        self.pathway_weights = pathway_weights
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, target):
        if self.pathway_weights is not None:
            # Weight different pathways differently
            weights = self.pathway_weights.to(pred.device)
            weighted_mse = torch.mean((pred - target) ** 2 * weights)
            return weighted_mse
        else:
            return self.mse_loss(pred, target)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification tasks"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()