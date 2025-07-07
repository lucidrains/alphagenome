"""
Dummy training script for BactaGenome
"""

import os
import argparse
import yaml
import random
import numpy as np
import logging
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

from bactagenome import BactaGenome, BactaGenomeConfig
from bactagenome.data import DummyBacterialDataset, DummyBacterialTargetsDataset
from bactagenome.training import BactaGenomeTrainer, BacterialLossFunction


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train BactaGenome model")
    parser.add_argument("--config", type=str, default="configs/training/dummy_full.yaml",
                        help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(config: dict) -> BactaGenome:
    """Create BactaGenome model from configuration"""
    model_config = BactaGenomeConfig(
        dims=tuple(config['model']['dims']),
        context_length=config['model']['context_length'],
        num_organisms=config['model']['num_organisms'],
        transformer_kwargs=config['model'].get('transformer_kwargs', {})
    )
    
    model = BactaGenome(model_config)
    
    # Add bacterial heads for each organism
    for organism_name in config['organisms']:
        model.add_bacterial_heads(organism_name)
    
    return model


def create_datasets(config: dict):
    """Create training and validation datasets"""
    # Create dummy heads configuration for Phase 1 only
    heads_cfg = {}
    for organism_name in config['organisms']:
        heads_cfg[organism_name] = {
            'promoter_strength': {'num_conditions': 10},     # Phase 1 Head 1
            'rbs_efficiency': {},                           # Phase 1 Head 2
            'operon_coregulation': {'num_genes': 5}         # Phase 1 Head 3
        }
    
    # Create datasets
    targets_dataset = DummyBacterialTargetsDataset(
        heads_cfg=heads_cfg,
        seq_len=config['data']['seq_len'],
        global_seed=config['training']['seed']
    )
    
    train_dataset = DummyBacterialDataset(
        seq_len=config['data']['seq_len'],
        num_samples=config['data']['num_train_samples'],
        targets_dataset=targets_dataset,
        num_organisms=config['model']['num_organisms'],
        global_seed=config['training']['seed']
    )
    
    val_dataset = DummyBacterialDataset(
        seq_len=config['data']['seq_len'],
        num_samples=config['data']['num_val_samples'],
        targets_dataset=targets_dataset,
        num_organisms=config['model']['num_organisms'],
        global_seed=config['training']['seed'] + 1000
    )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: dict):
    """Create data loaders"""
    # Disable pin_memory on MPS (Apple Silicon) to avoid warnings
    pin_memory = not torch.backends.mps.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def main():
    """Main training function"""
    logger = setup_logging()
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # Create output directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
    logger.info(f"Train dataset: {len(train_dataset)} samples, Val dataset: {len(val_dataset)} samples")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Create loss function
    loss_function = BacterialLossFunction(
        loss_weights=config['training'].get('loss_weights', {})
    )
    
    # Setup accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        log_with="wandb" if config['training'].get('use_wandb', False) else None
    )
    
    # Prepare model, optimizer, and dataloaders
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Create trainer
    trainer = BactaGenomeTrainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=accelerator.device,
        accelerator=accelerator,
        log_interval=config['training'].get('log_interval', 10)
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = trainer.load_checkpoint(args.resume)
        start_epoch = checkpoint['epoch']
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        
        # Training
        train_metrics = trainer.train_epoch(train_loader, epoch + 1)
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        
        # Log individual modality losses
        for modality, loss in train_metrics['modality_losses'].items():
            logger.info(f"Train {modality}: {loss:.4f}")
        
        # Validation
        if (epoch + 1) % config['training']['val_interval'] == 0:
            val_metrics = trainer.validate_epoch(val_loader)
            logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Log individual modality losses
            for modality, loss in val_metrics['modality_losses'].items():
                logger.info(f"Val {modality}: {loss:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                best_model_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pt')
                trainer.save_checkpoint(best_model_path, epoch + 1, val_loss=best_val_loss)
                logger.info(f"New best model saved: {best_model_path}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(
                config['training']['checkpoint_dir'], 
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            trainer.save_checkpoint(checkpoint_path, epoch + 1)
        
        # Wait for all processes
        accelerator.wait_for_everyone()
    
    # Save final model
    final_model_path = os.path.join(config['training']['checkpoint_dir'], 'final_model.pt')
    trainer.save_checkpoint(final_model_path, config['training']['epochs'])
    logger.info(f"Final model saved: {final_model_path}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()