import argparse
import os
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from alphagenome_pytorch import AlphaGenome, AlphaGenomeConfig, TargetScaler, MultinomialLoss, JunctionsLoss

# util

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

# data

class DummyTargetsDataset(Dataset):
    def __init__(self, heads_cfg, seq_len, dim_contacts, n_splice_site_types = 5, n_splice_sites = 3, global_seed = 1234):
        
        self.heads_cfg = heads_cfg
        self.global_seed = global_seed
        
        self.len_1bp = seq_len
        self.len_128bp = seq_len // 128
        self.dim_contacts = dim_contacts

        self.n_splice_site_types = n_splice_site_types
        self.n_splice_sites = n_splice_sites

    def __getitem__(self, idx):
        np.random.seed(self.global_seed + idx)

        targets = {}
        for organism, config in self.heads_cfg.items():
        
            targets[organism] = {
                'target_1bp_tracks': torch.rand(self.len_1bp, config['num_tracks_1bp']).clamp(min=0.01),
                'target_128bp_tracks': torch.rand(self.len_128bp, config['num_tracks_128bp']).clamp(min=0.01),
                'target_contact_head': torch.rand(self.dim_contacts, self.dim_contacts, config['num_tracks_contacts']).clamp(min=0.01),
                'target_splice_probs': torch.nn.functional.one_hot(torch.randint(0, self.n_splice_site_types, (self.len_1bp,)), num_classes=self.n_splice_site_types).float(),
                'target_splice_usage': torch.bernoulli(torch.rand(self.len_1bp, config['num_splicing_contexts'])),
                'target_splice_juncs': torch.abs(torch.randn(self.n_splice_sites, self.n_splice_sites, config['num_splicing_contexts'])).clamp(min=0.01)

            }
        
        return targets

class DummyGenomeDataset(Dataset):
    def __init__(self, seq_len, num_samples, targets_dataset=None, global_seed=1234):
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.targets_dataset = targets_dataset
        self.global_seed = global_seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        np.random.seed(self.global_seed + idx)
        
        dna = torch.randint(0, 5, (self.seq_len,))
        organism_index = torch.randint(0, 2, (1,)).item()
        splice_donor_idx = torch.randint(0, self.seq_len, (3,))
        splice_acceptor_idx = torch.randint(0, self.seq_len, (3,))

        item = {
            'dna': dna,
            'organism_index': organism_index,
            'splice_donor_idx': splice_donor_idx,
            'splice_acceptor_idx': splice_acceptor_idx,
        }

        if self.targets_dataset is not None:
            targets = self.targets_dataset[idx]
            for target_organism, target_tensors in targets.items():
                for target_name, target_tensor in target_tensors.items():
                    item[target_name] = target_tensor
        return item

# training

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # dataset
    
    targets_dataset = DummyTargetsDataset(heads_cfg, seq_len, dim_contacts=model.dim_contacts, global_seed=seed)
    train_dataset = DummyGenomeDataset(seq_len, num_samples, targets_dataset, global_seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    organism_list = sorted(heads_cfg.keys())  # assumes consistent order
    index_to_organism = {i: org for i, org in enumerate(organism_list)}

    # optimization
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    target_scaler = {
        organism : {
                '1bp_tracks': TargetScaler(track_means = torch.ones(heads['num_tracks_1bp'])),
                '128bp_tracks': TargetScaler(track_means = torch.ones(heads['num_tracks_128bp']))
            }
        for organism, heads in heads_cfg.items()
    }
    loss_fns = {
        '1bp_tracks' : MultinomialLoss(multinomial_resolution = seq_len // 1),
        '128bp_tracks' : MultinomialLoss(multinomial_resolution = seq_len // 128),
        'contact_head' : nn.MSELoss(),
        'splice_probs' : nn.CrossEntropyLoss(),
        'splice_usage' : nn.CrossEntropyLoss(),
        'splice_juncs' : JunctionsLoss()
    }

    # training loop
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:

            # unpack inputs
            
            dna = batch['dna'].to(device)
            organism_index = batch['organism_index'].to(device)
            splice_donor_idx = batch['splice_donor_idx'].to(device)
            splice_acceptor_idx = batch['splice_acceptor_idx'].to(device)

            # subset batch by organism

            losses = []
            for org_idx in organism_index.unique():
                idx = organism_index == org_idx
                
                # predictions for this organism
                
                preds = model(dna[idx], organism_index[idx], splice_donor_idx=splice_donor_idx[idx], splice_acceptor_idx=splice_acceptor_idx[idx])

                # loss
                
                organism = index_to_organism[org_idx.item()] # only for the organism that we subsetted the batch for
                for head_name, pred_tensor in preds[organism].items():

                    # get corresponding targets

                    target = batch[f'target_{head_name}']
                    target = target[idx].to(device)
                    if head_name in target_scaler[organism].keys():
                        target = target_scaler[organism][head_name](target)

                    # compute loss
                    loss_organism = loss_fns[head_name](pred_tensor, target)
                    losses.append(loss_organism)
                
            loss = torch.stack(losses).sum()

            # back propagate
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # save checkpoint
        
        if (epoch + 1) % config['checkpoint_freq'] == 0:
            save_path = os.path.join(config['checkpoint_dir'], f'epoch_{epoch+1}.pt')
            save_model(model, optimizer, epoch + 1, save_path)
            print(f"Saved checkpoint: {save_path}")

    # save final model
    
    save_path = os.path.join(output_dir, f'epoch_{epoch+1}.pt')
    save_model(model, optimizer, epochs, save_path)


if __name__ == "__main__":
    main()
