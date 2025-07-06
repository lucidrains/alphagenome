"""
Utility functions for bactagenome model
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .config import BactaGenomeConfig


def count_parameters(model: nn.Module) -> int:
    """Count total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze the transformer backbone"""
    for name, param in model.named_parameters():
        if 'transformer_unet' in name:
            param.requires_grad = not freeze


def get_learning_rates(model: nn.Module, backbone_lr: float = 1e-4, head_lr: float = 1e-3) -> list:
    """Get parameter groups with different learning rates for backbone and heads"""
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'transformer_unet' in name or 'organism_embed' in name or 'outembed' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    return [
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ]


def calculate_model_size(model: nn.Module) -> Dict[str, Any]:
    """Calculate model size statistics"""
    total_params = count_parameters(model)
    
    # Calculate size of different components
    component_sizes = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'parameters'):
            component_params = sum(p.numel() for p in module.parameters())
            if component_params > 0:
                component_sizes[name] = component_params
    
    # Calculate memory usage (approximate)
    param_size = total_params * 4  # 4 bytes per float32
    
    return {
        'total_parameters': total_params,
        'total_size_mb': param_size / (1024 * 1024),
        'component_sizes': component_sizes
    }


def load_pretrained_backbone(
    model: nn.Module, 
    pretrained_path: str, 
    strict: bool = False
) -> None:
    """Load pretrained AlphaGenome backbone weights"""
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Extract only backbone weights
    backbone_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('transformer_unet') or key.startswith('organism_embed') or key.startswith('outembed'):
            backbone_state_dict[key] = value
    
    # Load weights
    model.load_state_dict(backbone_state_dict, strict=strict)
    print(f"Loaded pretrained backbone from {pretrained_path}")


def create_bacterial_species_mapping() -> Dict[str, int]:
    """Create mapping from species names to indices"""
    species = [
        "E_coli_K12",
        "B_subtilis_168", 
        "Salmonella_enterica",
        "Pseudomonas_aeruginosa",
        "Mycobacterium_tuberculosis",
        "Streptococcus_pyogenes",
        "Synechocystis_sp"
    ]
    
    return {species_name: idx for idx, species_name in enumerate(species)}


def get_phase_config(phase: str) -> Dict[str, Any]:
    """Get training configuration for different phases"""
    config = BactaGenomeConfig()
    
    if phase == "phase1":
        return config.get_phase1_config()
    elif phase == "phase2":
        return config.get_phase2_config()
    else:
        raise ValueError(f"Unknown phase: {phase}")


def validate_batch_shapes(batch: Dict[str, torch.Tensor]) -> None:
    """Validate that batch tensors have consistent shapes"""
    batch_size = None
    seq_len = None
    
    for key, tensor in batch.items():
        if batch_size is None:
            batch_size = tensor.shape[0]
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"Inconsistent batch size for {key}: {tensor.shape[0]} vs {batch_size}")
        
        if 'seq' in key or 'dna' in key:
            if seq_len is None:
                seq_len = tensor.shape[1]
            elif tensor.shape[1] != seq_len:
                raise ValueError(f"Inconsistent sequence length for {key}: {tensor.shape[1]} vs {seq_len}")


def format_model_summary(model: nn.Module) -> str:
    """Format a summary of the model architecture"""
    stats = calculate_model_size(model)
    
    summary = f"""
bactagenome Model Summary:
========================
Total Parameters: {stats['total_parameters']:,}
Model Size: {stats['total_size_mb']:.2f} MB

Architecture:
- Transformer Backbone: {stats['component_sizes'].get('transformer_unet', 0):,} parameters
- Organism Embeddings: {stats['component_sizes'].get('organism_embed', 0):,} parameters
- Output Embeddings: {sum(v for k, v in stats['component_sizes'].items() if 'outembed' in k):,} parameters
- Prediction Heads: {sum(v for k, v in stats['component_sizes'].items() if 'heads' in k):,} parameters
"""
    
    return summary