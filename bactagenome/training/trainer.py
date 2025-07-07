"""
Training utilities for BactaGenome
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
import logging
from tqdm import tqdm
from accelerate import Accelerator
from .losses import BacterialLossFunction


class BactaGenomeTrainer:
    """
    Training utility for BactaGenome models
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: Optional[BacterialLossFunction] = None,
        device: str = "cuda",
        accelerator: Optional[Accelerator] = None,
        log_interval: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function or BacterialLossFunction()
        self.device = device
        self.accelerator = accelerator
        self.log_interval = log_interval
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Organism mapping
        self.organism_names = [
            "E_coli_K12",
            "B_subtilis_168", 
            "Salmonella_enterica",
            "Pseudomonas_aeruginosa",
            "Mycobacterium_tuberculosis",
            "Streptococcus_pyogenes",
            "Synechocystis_sp"
        ]
        self.index_to_organism = {i: name for i, name in enumerate(self.organism_names)}
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of losses and metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        modality_losses = {}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Extract inputs
            dna = batch['dna']
            organism_index = batch['organism_index']
            
            # Forward pass
            self.optimizer.zero_grad()
            
            batch_loss = 0.0
            batch_modality_losses = {}
            
            # Process each organism in the batch
            for org_idx in organism_index.unique():
                mask = organism_index == org_idx
                org_name = self.index_to_organism[org_idx.item()]
                
                # Get predictions for this organism
                org_dna = dna[mask]
                org_organism_index = organism_index[mask]
                
                predictions = self.model(org_dna, org_organism_index)
                
                # Extract targets for this organism
                targets = self._extract_targets(batch, mask, org_name)
                
                # Compute loss
                if org_name in predictions:
                    loss, individual_losses = self.loss_function(
                        predictions[org_name], targets, org_name
                    )
                    batch_loss += loss
                    
                    # Accumulate individual losses
                    for modality, loss_val in individual_losses.items():
                        if modality not in batch_modality_losses:
                            batch_modality_losses[modality] = 0.0
                        batch_modality_losses[modality] += loss_val
            
            # Backward pass
            if self.accelerator is not None:
                self.accelerator.backward(batch_loss)
            else:
                batch_loss.backward()
                
            self.optimizer.step()
            
            # Update metrics
            total_loss += batch_loss.item()
            total_samples += len(dna)
            
            # Update modality losses
            for modality, loss_val in batch_modality_losses.items():
                if modality not in modality_losses:
                    modality_losses[modality] = 0.0
                modality_losses[modality] += loss_val
                
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{batch_loss.item():.4f}",
                    'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
                })
        
        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_modality_losses = {
            modality: loss / len(dataloader) 
            for modality, loss in modality_losses.items()
        }
        
        return {
            'total_loss': avg_loss,
            'modality_losses': avg_modality_losses,
            'samples_processed': total_samples
        }
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation losses and metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        modality_losses = {}
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            
            for batch in pbar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Extract inputs
                dna = batch['dna']
                organism_index = batch['organism_index']
                
                batch_loss = 0.0
                batch_modality_losses = {}
                
                # Process each organism in the batch
                for org_idx in organism_index.unique():
                    mask = organism_index == org_idx
                    org_name = self.index_to_organism[org_idx.item()]
                    
                    # Get predictions for this organism
                    org_dna = dna[mask]
                    org_organism_index = organism_index[mask]
                    
                    predictions = self.model(org_dna, org_organism_index)
                    
                    # Extract targets for this organism
                    targets = self._extract_targets(batch, mask, org_name)
                    
                    # Compute loss
                    if org_name in predictions:
                        loss, individual_losses = self.loss_function(
                            predictions[org_name], targets, org_name
                        )
                        batch_loss += loss
                        
                        # Accumulate individual losses
                        for modality, loss_val in individual_losses.items():
                            if modality not in batch_modality_losses:
                                batch_modality_losses[modality] = 0.0
                            batch_modality_losses[modality] += loss_val
                
                # Update metrics
                total_loss += batch_loss.item()
                total_samples += len(dna)
                
                # Update modality losses
                for modality, loss_val in batch_modality_losses.items():
                    if modality not in modality_losses:
                        modality_losses[modality] = 0.0
                    modality_losses[modality] += loss_val
        
        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_modality_losses = {
            modality: loss / len(dataloader) 
            for modality, loss in modality_losses.items()
        }
        
        return {
            'total_loss': avg_loss,
            'modality_losses': avg_modality_losses,
            'samples_processed': total_samples
        }
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device"""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
                
        return device_batch
    
    def _extract_targets(self, batch: Dict[str, Any], mask: torch.Tensor, 
                        org_name: str) -> Dict[str, torch.Tensor]:
        """Extract target values for a specific organism"""
        targets = {}
        
        target_keys = [
            'promoter_strength', 'rbs_efficiency', 'operon_coregulation'
        ]
        
        for key in target_keys:
            target_key = f'target_{key}'
            if target_key in batch:
                targets[key] = batch[target_key][mask]
                
        return targets
    
    def save_checkpoint(self, filepath: str, epoch: int, **kwargs):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **kwargs
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Loaded checkpoint: {filepath}")
        return checkpoint