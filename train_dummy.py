from __future__ import annotations

import argparse
import os
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from alphagenome_pytorch import AlphaGenome, AlphaGenomeConfig, TargetScaler, MultinomialLoss, JunctionsLoss
from alphagenome_pytorch.data import DummyGenomeDataset, DummyTargetsDataset

from accelerate import Accelerator

# util

def exists(v):
    return v is not None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()
    return args
    
def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

# training

def train_one_epoch(model, dataloader, optimizer, loss_fns, target_scaler, index_to_organism, device, accelerator: Accelerator | None = None):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Move inputs to device
        dna = batch['dna'].to(device)
        organism_index = batch['organism_index'].to(device)
        splice_donor_idx = batch['splice_donor_idx'].to(device)
        splice_acceptor_idx = batch['splice_acceptor_idx'].to(device)

        losses = []

        for org_idx in organism_index.unique():
            idx = organism_index == org_idx
            org_name = index_to_organism[org_idx.item()]

            # Forward pass
            preds = model(
                dna[idx],
                organism_index[idx],
                splice_donor_idx=splice_donor_idx[idx],
                splice_acceptor_idx=splice_acceptor_idx[idx]
            )

            # Loss computation
            for head_name, pred_tensor in preds[org_name].items():
                target = batch[f'target_{head_name}'][idx].to(device)
                if head_name in target_scaler[org_name]:
                    target = target_scaler[org_name][head_name](target)
                losses.append(loss_fns[head_name](pred_tensor, target))

        loss = torch.stack(losses).sum()

        if exists(accelerator):
            accelerator.backward(loss)
        else:
            loss.backward()

        # Backpropagation

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss



def main():
    torch.autograd.set_detect_anomaly(True)

    # unpack config
    
    args = parse_args()
    config = load_config(args.config_file)
    
    seed = config.get('seed', 1234)
    seq_len = config.get('seq_len', 8192)
    num_samples = config.get('num_samples', 1000)
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 4)
    epochs = config.get('epochs', 10)
    lr = config.get('lr', 1e-4)
    checkpoint_freq = config.get('checkpoint_freq', 2)
    
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    output_dir = config.get('output_dir', './outputs')

    default_cfg = AlphaGenomeConfig()
    model_cfg = config.get('model', {})
    dims = tuple(model_cfg.get('dims', default_cfg.dims))
    basepairs = model_cfg.get('basepairs', default_cfg.basepairs)
    dna_embed_width = model_cfg.get('dna_embed_width', default_cfg.dna_embed_width)
    num_organisms = model_cfg.get('num_organisms', default_cfg.num_organisms)
    transformer_kwargs = model_cfg.get('transformer_kwargs', default_cfg.transformer_kwargs)
    heads_cfg = config.get('heads', default_cfg.head_specs)

    # init
    
    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # architecture
    
    model = AlphaGenome(dims, basepairs, dna_embed_width, num_organisms, transformer_kwargs)
    for organism, head_cfg in heads_cfg.items():
        model.add_heads(organism=organism, **head_cfg)
    print("Total model parameters:", model.total_parameters)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    # dataset
    
    targets_dataset = DummyTargetsDataset(heads_cfg, seq_len, dim_contacts=model.dim_contacts, global_seed=seed)
    train_dataset = DummyGenomeDataset(seq_len, num_samples, targets_dataset, global_seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    organism_list = sorted(heads_cfg.keys())  # assumes consistent order
    index_to_organism = {i: org for i, org in enumerate(organism_list)}

    # optimization
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # prepare accelerator for distributed

    accelerate_kwargs = {}

    accelerator = Accelerator(**accelerate_kwargs)

    model, train_loader, optimizer = accelerator.prepare(model, train_loader, optimizer)

    device = accelerator.device

    # scaler and losses

    target_scaler = {
        organism : {
            '1bp_tracks': TargetScaler(track_means = torch.ones(heads['num_tracks_1bp'])).to(device),
            '128bp_tracks': TargetScaler(track_means = torch.ones(heads['num_tracks_128bp'])).to(device)
        }
        for organism, heads in heads_cfg.items()
    }
    loss_fns = {
        '1bp_tracks' : MultinomialLoss(multinomial_resolution = seq_len // 1),
        '128bp_tracks' : MultinomialLoss(multinomial_resolution = seq_len // 128),
        'contact_head' : nn.MSELoss(),
        'splice_probs' : nn.CrossEntropyLoss(), # this should act on the splice logits
        'splice_usage' : nn.CrossEntropyLoss(),
        'splice_juncs' : JunctionsLoss()
    }

    # training loop
    
    for epoch in range(epochs):
        avg_train_loss = train_one_epoch(
            model = model,
            dataloader = train_loader,
            optimizer = optimizer,
            loss_fns = loss_fns,
            target_scaler = target_scaler,
            index_to_organism = index_to_organism,
            accelerator = accelerator,
            device = device
        )
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
    
        if (epoch + 1) % checkpoint_freq == 0 and accelerator.is_main_process:
            save_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pt')
            save_model(model, optimizer, epoch + 1, save_path)
            print(f"Saved checkpoint: {save_path}")

        accelerator.wait_for_everyone()

    # save final model
    
    save_path = os.path.join(output_dir, f'epoch_{epoch+1}.pt')
    save_model(model, optimizer, epochs, save_path)
    print(f"Saved final model: {save_path}")

if __name__ == "__main__":
    main()
