import numpy as np
import torch
from torch.utils.data import Dataset

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

        item = {}
        for organism, config in self.heads_cfg.items():
        
            item[organism] = {
                'target_1bp_tracks': torch.rand(self.len_1bp, config['num_tracks_1bp']).clamp(min=0.01),
                'target_128bp_tracks': torch.rand(self.len_128bp, config['num_tracks_128bp']).clamp(min=0.01),
                'target_contact_head': torch.rand(self.dim_contacts, self.dim_contacts, config['num_tracks_contacts']).clamp(min=0.01),
                'target_splice_probs': torch.nn.functional.one_hot(torch.randint(0, self.n_splice_site_types, (self.len_1bp,)), num_classes=self.n_splice_site_types).float(),
                'target_splice_usage': torch.bernoulli(torch.rand(self.len_1bp, config['num_splicing_contexts'])),
                'target_splice_juncs': torch.abs(torch.randn(self.n_splice_sites, self.n_splice_sites, config['num_splicing_contexts'])).clamp(min=0.01)

            }
        
        return item

class DummyGenomeDataset(Dataset):
    def __init__(self, seq_len, num_samples, targets_dataset = None, global_seed = 1234):
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