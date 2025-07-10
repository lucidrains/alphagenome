"""
Training script for BactaGenome with real RegulonDB data
"""

import os
import platform
import argparse
import yaml
import random
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict

# MPS fallback for Apple Silicon
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    print("✅ MPS fallback enabled for Apple Silicon.")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from bactagenome import BactaGenome, BactaGenomeConfig
from bactagenome.data import RegulonDBDataset, RegulonDBDataLoader, collate_regulondb_batch
from bactagenome.training import BactaGenomeTrainer
from bactagenome.model.losses import BacterialLossFunction


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training_regulondb.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train BactaGenome with RegulonDB data")
    parser.add_argument("--config", type=str, default="configs/training/phase1_regulondb.yaml",
                        help="Path to training configuration file")
    parser.add_argument("--regulondb-path", type=str, 
                        default="/Users/zhaoj/Project/BactaGenome/data/raw/RegulonDB",
                        help="Path to raw RegulonDB BSON files")
    parser.add_argument("--processed-data-dir", type=str,
                        default="/Users/zhaoj/Project/BactaGenome/data/processed/regulondb",
                        help="Directory for processed RegulonDB data")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--process-data-only", action="store_true",
                        help="Only process data, don't train")
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
    
    # Add bacterial heads for each organism (Phase 1: E. coli only)
    phase = config.get('phase', 1)
    for organism_name in config['organisms']:
        model.add_bacterial_heads(organism_name, phase=phase)
    
    return model


def create_regulondb_datasets(config: dict, regulondb_path: str, processed_data_dir: str):
    """Create RegulonDB training and validation datasets"""
    
    # Create datasets using chromosome-based splits
    train_dataset = RegulonDBDataset(
        data_dir=processed_data_dir,
        seq_len=config['data']['seq_len'],
        num_organisms=config['model']['num_organisms'],
        organism_name="E_coli_K12",
        split='train',
        process_if_missing=True,
        regulondb_raw_path=regulondb_path
    )
    
    val_dataset = RegulonDBDataset(
        data_dir=processed_data_dir,
        seq_len=config['data']['seq_len'],
        num_organisms=config['model']['num_organisms'],
        organism_name="E_coli_K12", 
        split='val',
        process_if_missing=False,  # Already processed by train_dataset
        regulondb_raw_path=None
    )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config: dict):
    """Create data loaders with custom collate function"""
    
    # Disable pin_memory on MPS (Apple Silicon) to avoid warnings
    pin_memory = not torch.backends.mps.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=pin_memory,
        collate_fn=collate_regulondb_batch,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=pin_memory,
        collate_fn=collate_regulondb_batch,
        drop_last=False
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
    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    # Create datasets
    logger.info("Creating RegulonDB datasets...")
    train_dataset, val_dataset = create_regulondb_datasets(
        config, args.regulondb_path, args.processed_data_dir
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Exit if only processing data
    if args.process_data_only:
        logger.info("Data processing complete. Exiting.")
        return
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    logger.info(f"Model created with {model.total_parameters:,} parameters")
    
    # Print model heads info
    for organism, heads in model.heads.items():
        logger.info(f"Organism {organism}: {list(heads.keys())}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )
    
    # Create loss function with RegulonDB-appropriate weights
    loss_weights = config['training'].get('loss_weights', {
        'promoter_strength': 1.0,
        'rbs_efficiency': 1.0,
        'operon_coregulation': 1.0
    })
    
    loss_function = BacterialLossFunction(loss_weights=loss_weights)
    
    # Setup accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        mixed_precision=config['training'].get('mixed_precision', 'no'),
        log_with="wandb" if config['training'].get('use_wandb', False) else None,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Prepare for distributed training
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
    logger.info("Starting RegulonDB training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        
        # Training
        train_metrics = trainer.train_epoch(train_loader, epoch + 1)
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        
        # Log individual modality losses
        modality_losses = {k: v for k, v in train_metrics.items() if k != 'total_loss'}
        for modality, loss in modality_losses.items():
            logger.info(f"Train {modality}: {loss:.4f}")
        
        # Validation
        if (epoch + 1) % config['training'].get('val_interval', 5) == 0:
            val_metrics = trainer.validate_epoch(val_loader)
            logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Log individual modality losses
            val_modality_losses = {k: v for k, v in val_metrics.items() if k != 'total_loss'}
            for modality, loss in val_modality_losses.items():
                logger.info(f"Val {modality}: {loss:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                best_model_path = os.path.join(config['training']['checkpoint_dir'], 'best_model_regulondb.pt')
                trainer.save_checkpoint(best_model_path, epoch + 1, val_loss=best_val_loss)
                logger.info(f"New best model saved: {best_model_path}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(
                config['training']['checkpoint_dir'],
                f'checkpoint_regulondb_epoch_{epoch + 1}.pt'
            )
            trainer.save_checkpoint(checkpoint_path, epoch + 1)
        
        # Wait for all processes
        accelerator.wait_for_everyone()
    
    # Save final model
    final_model_path = os.path.join(config['training']['checkpoint_dir'], 'final_model_regulondb.pt')
    trainer.save_checkpoint(final_model_path, config['training']['epochs'])
    logger.info(f"Final model saved: {final_model_path}")
    
    logger.info("RegulonDB training completed!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()