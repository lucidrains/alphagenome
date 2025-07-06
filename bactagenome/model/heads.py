"""
Bacterial-specific prediction heads for bactagenome
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from einops import repeat, reduce


class PromoterStrengthHead(nn.Module):
    """
    Promoter strength prediction head
    Predicts expression levels across different conditions
    """
    
    def __init__(self, dim_1bp: int, dim_128bp: int, num_conditions: int):
        super().__init__()
        self.to_1bp = Linear(dim_1bp, num_conditions)
        self.to_128bp = Linear(dim_128bp, num_conditions)
        self.scale = nn.Parameter(torch.ones(num_conditions))
    
    def forward(self, embeds_1bp: torch.Tensor, embeds_128bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
            embeds_128bp: [batch, seq_len//128, dim_128bp]
        
        Returns:
            [batch, seq_len, num_conditions] - Expression levels
        """
        x_1bp = self.to_1bp(embeds_1bp)
        x_128bp = self.to_128bp(embeds_128bp)
        
        # Repeat 128bp predictions to match 1bp resolution
        x_128bp_expanded = repeat(x_128bp, 'b n c -> b (n repeat) c', repeat=128)
        
        # Ensure shapes match
        seq_len = x_1bp.shape[1]
        x_128bp_expanded = x_128bp_expanded[:, :seq_len]
        
        # Combine predictions
        combined = (x_1bp + x_128bp_expanded) / 2
        
        # Apply learnable scale and softplus activation
        return F.softplus(combined) * F.softplus(self.scale)


class RBSEfficiencyHead(nn.Module):
    """
    Ribosome binding site (RBS) translation efficiency prediction
    """
    
    def __init__(self, dim_1bp: int):
        super().__init__()
        self.linear = Linear(dim_1bp, 1)
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, embeds_1bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
        
        Returns:
            [batch, seq_len, 1] - Translation efficiency
        """
        x = self.linear(embeds_1bp)
        return F.softplus(x) * F.softplus(self.scale)


class OperonCoregulationHead(nn.Module):
    """
    Operon co-regulation prediction head
    Predicts gene co-expression within operons
    """
    
    def __init__(self, dim_128bp: int, dim_pairwise: int, num_genes: int):
        super().__init__()
        self.local_context = Linear(dim_128bp, num_genes)
        self.pair_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pair_proj = Linear(dim_pairwise, num_genes)
    
    def forward(self, embeds_128bp: torch.Tensor, embeds_pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_128bp: [batch, seq_len//128, dim_128bp]
            embeds_pair: [batch, seq_len//2048, seq_len//2048, dim_pairwise]
        
        Returns:
            [batch, seq_len//128, num_genes] - Co-expression scores
        """
        local_features = self.local_context(embeds_128bp)
        
        # Pool pairwise features
        pair_features = self.pair_pool(embeds_pair.permute(0, 3, 1, 2))  # [B, D, 1, 1]
        pair_features = pair_features.squeeze(-1).squeeze(-1)  # [B, D]
        pair_contribution = self.pair_proj(pair_features)  # [B, num_genes]
        
        # Broadcast pair contribution to sequence length
        seq_len = local_features.shape[1]
        pair_contribution = pair_contribution.unsqueeze(1).expand(-1, seq_len, -1)
        
        return F.softplus(local_features + pair_contribution)


class RiboswitchBindingHead(nn.Module):
    """
    Riboswitch ligand binding prediction
    """
    
    def __init__(self, dim_1bp: int, dim_128bp: int, num_ligands: int):
        super().__init__()
        self.motif_conv = Conv1d(dim_1bp, num_ligands, kernel_size=15, padding=7)
        self.context_proj = Linear(dim_128bp, num_ligands)
    
    def forward(self, embeds_1bp: torch.Tensor, embeds_128bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
            embeds_128bp: [batch, seq_len//128, dim_128bp]
        
        Returns:
            [batch, seq_len, num_ligands] - Binding probabilities
        """
        # Motif features from 1bp resolution
        x_1bp = embeds_1bp.transpose(1, 2)  # [B, D, L]
        motif_features = self.motif_conv(x_1bp).transpose(1, 2)  # [B, L, num_ligands]
        
        # Context features from 128bp resolution
        context_features = self.context_proj(embeds_128bp)
        context_expanded = repeat(context_features, 'b n c -> b (n repeat) c', repeat=128)
        
        # Ensure shapes match
        seq_len = motif_features.shape[1]
        context_expanded = context_expanded[:, :seq_len]
        
        # Combine and apply sigmoid
        binding_logits = motif_features + context_expanded
        return torch.sigmoid(binding_logits)


class SRNATargetHead(nn.Module):
    """
    Small RNA target prediction head
    """
    
    def __init__(self, dim_1bp: int, dim_128bp: int, num_srnas: int):
        super().__init__()
        self.base_pair_proj = Linear(dim_1bp, num_srnas)
        self.context_proj = Linear(dim_128bp, num_srnas)
    
    def forward(self, embeds_1bp: torch.Tensor, embeds_128bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
            embeds_128bp: [batch, seq_len//128, dim_128bp]
        
        Returns:
            [batch, seq_len, num_srnas] - Interaction strength
        """
        base_pair_features = self.base_pair_proj(embeds_1bp)
        context_features = self.context_proj(embeds_128bp)
        
        # Expand context features
        context_expanded = repeat(context_features, 'b n c -> b (n repeat) c', repeat=128)
        
        # Ensure shapes match
        seq_len = base_pair_features.shape[1]
        context_expanded = context_expanded[:, :seq_len]
        
        interaction_strength = base_pair_features + context_expanded
        return F.softplus(interaction_strength)


class TerminationHead(nn.Module):
    """
    Transcription termination prediction
    """
    
    def __init__(self, dim_1bp: int):
        super().__init__()
        self.terminator_proj = Linear(dim_1bp, 3)  # Intrinsic, Rho-dependent, None
    
    def forward(self, embeds_1bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
        
        Returns:
            [batch, seq_len, 3] - Termination probabilities
        """
        terminator_logits = self.terminator_proj(embeds_1bp)
        return F.softmax(terminator_logits, dim=-1)


class PathwayActivityHead(nn.Module):
    """
    Metabolic pathway activity prediction
    """
    
    def __init__(self, dim_128bp: int, dim_pairwise: int, num_pathways: int):
        super().__init__()
        self.gene_pool = nn.AdaptiveAvgPool1d(1)
        self.gene_proj = Linear(dim_128bp, num_pathways)
        
        self.pair_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pair_proj = Linear(dim_pairwise, num_pathways)
        
        self.final_proj = Linear(num_pathways * 2, num_pathways)
    
    def forward(self, embeds_128bp: torch.Tensor, embeds_pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_128bp: [batch, seq_len//128, dim_128bp]
            embeds_pair: [batch, seq_len//2048, seq_len//2048, dim_pairwise]
        
        Returns:
            [batch, num_pathways] - Pathway activity scores
        """
        # Gene activity features
        gene_features = self.gene_pool(embeds_128bp.transpose(1, 2)).squeeze(-1)  # [B, D]
        gene_activities = self.gene_proj(gene_features)  # [B, num_pathways]
        
        # Organization features from pairwise
        pair_features = self.pair_pool(embeds_pair.permute(0, 3, 1, 2))  # [B, D, 1, 1]
        pair_features = pair_features.squeeze(-1).squeeze(-1)  # [B, D]
        organization_features = self.pair_proj(pair_features)  # [B, num_pathways]
        
        # Combine features
        combined = torch.cat([gene_activities, organization_features], dim=-1)
        pathway_scores = self.final_proj(combined)
        
        return torch.sigmoid(pathway_scores)


class SecretionSignalHead(nn.Module):
    """
    Protein secretion signal prediction
    """
    
    def __init__(self, dim_1bp: int, num_secretion_types: int):
        super().__init__()
        self.signal_proj = Linear(dim_1bp, num_secretion_types)
    
    def forward(self, embeds_1bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
        
        Returns:
            [batch, seq_len, num_secretion_types] - Secretion type probabilities
        """
        signal_logits = self.signal_proj(embeds_1bp)
        return torch.sigmoid(signal_logits)  # Multi-label classification