"""
Bacterial-specific prediction heads for BactaGenome
Based on AlphaGenome architecture patterns adapted for bacterial biology and RegulonDB data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d, Parameter
from einops import repeat, reduce
import math


def tracks_scaled_predictions(embeddings: torch.Tensor, num_tracks: int, head: nn.Module, scale_param: Parameter) -> torch.Tensor:
    """
    AlphaGenome-style track prediction with learnable scaling
    """
    x = head(embeddings)  # Linear projection
    return F.softplus(x) * F.softplus(scale_param)


def targets_scaling(targets: torch.Tensor, track_means: torch.Tensor, apply_squashing: bool = False) -> torch.Tensor:
    """
    AlphaGenome-style target scaling for numerical stability
    """
    targets = targets / (track_means + 1e-7)
    if apply_squashing:  # For RNA-seq like data
        targets = targets ** 0.75
    return torch.where(targets > 10.0, 2 * torch.sqrt(targets * 10.0) - 10.0, targets)


def predictions_scaling(predictions: torch.Tensor, track_means: torch.Tensor, apply_squashing: bool = False) -> torch.Tensor:
    """
    Inverse scaling for evaluation against original experimental data
    """
    x = torch.where(predictions > 10.0, (predictions + 10.0) ** 2 / (4 * 10.0), predictions)
    if apply_squashing:
        x = x ** (1.0 / 0.75)
    return x * track_means


class PromoterStrengthHead(nn.Module):
    """
    Promoter strength prediction head - AlphaGenome style
    
    Based on RegulonDB data:
    - geneExpression.bson (2.6GB): Expression levels under different conditions  
    - transcriptionStartSite.bson (49MB): 68,044 TSS positions with promoter data
    
    Predicts gene expression levels across conditions using both 1bp and 128bp embeddings.
    Follows AlphaGenome's RNA-seq head architecture with multi-resolution inputs.
    """
    
    def __init__(self, dim_1bp: int, dim_128bp: int, num_conditions: int = 50):
        super().__init__()
        self.num_conditions = num_conditions
        
        # Separate linear heads for each resolution (AlphaGenome pattern)
        self.head_1bp = Linear(dim_1bp, num_conditions)
        self.head_128bp = Linear(dim_128bp, num_conditions)
        
        # Learnable per-track scaling (AlphaGenome pattern)
        self.scale = Parameter(torch.zeros(num_conditions))  # Init to 0 for softplus
        
        # Track means for scaling (set during data preprocessing)
        self.register_buffer('track_means', torch.ones(num_conditions))
    
    def forward(self, embeds_1bp: torch.Tensor, embeds_128bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp] - 1bp resolution embeddings
            embeds_128bp: [batch, seq_len//128, dim_128bp] - 128bp resolution embeddings
        
        Returns:
            [batch, seq_len, num_conditions] - Expression levels per condition
        """
        # Process each resolution separately (AlphaGenome pattern)
        x_1bp = self.head_1bp(embeds_1bp)
        x_128bp = self.head_128bp(embeds_128bp)
        
        # Upsample 128bp to 1bp resolution
        x_128bp_upsampled = repeat(x_128bp, 'b n c -> b (n repeat) c', repeat=128)
        
        # Handle length mismatches
        seq_len = x_1bp.shape[1]
        if x_128bp_upsampled.shape[1] != seq_len:
            x_128bp_upsampled = x_128bp_upsampled[:, :seq_len]
        
        # Combine multi-resolution predictions
        combined = x_1bp + x_128bp_upsampled
        
        # Apply softplus + learnable scaling (AlphaGenome pattern)
        return F.softplus(combined) * F.softplus(self.scale)


class RBSEfficiencyHead(nn.Module):
    """
    RBS (Ribosome Binding Site) translation efficiency prediction head
    
    Based on RegulonDB data:
    - geneDatamart.bson (142MB): 4,747 genes with 5' UTR sequences containing RBS
    - geneExpression.bson: Translation efficiency inferred from mRNA vs protein ratios
    
    Predicts translation initiation rates at single nucleotide resolution.
    Uses only 1bp embeddings for precise RBS motif detection.
    """
    
    def __init__(self, dim_1bp: int):
        super().__init__()
        # Single output track for translation efficiency
        self.efficiency_head = Linear(dim_1bp, 1)
        
        # Learnable scaling parameter (AlphaGenome pattern)
        self.scale = Parameter(torch.zeros(1))  # Init to 0 for softplus
        
        # Track mean for scaling
        self.register_buffer('track_means', torch.ones(1))
    
    def forward(self, embeds_1bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp] - 1bp resolution embeddings
        
        Returns:
            [batch, seq_len, 1] - Translation efficiency scores
        """
        # Linear projection to efficiency scores
        efficiency_logits = self.efficiency_head(embeds_1bp)
        
        # Softplus activation + learnable scaling (AlphaGenome pattern)
        return F.softplus(efficiency_logits) * F.softplus(self.scale)


class OperonCoregulationHead(nn.Module):
    """
    Operon co-regulation prediction head
    
    Based on RegulonDB data:
    - operonDatamart.bson (64MB): 2,609 operons with detailed structure
    - geneCoexpressions.bson (2.0GB): Co-expression patterns between genes
    - regulatoryNetworkDatamart.bson (54MB): Regulatory interactions
    
    Predicts gene co-expression within operons using both local context and long-range interactions.
    Uses 128bp embeddings for gene-level features + pairwise embeddings for operon structure.
    """
    
    def __init__(self, dim_128bp: int, dim_pairwise: int, num_coexpression_tracks: int = 20):
        super().__init__()
        self.num_tracks = num_coexpression_tracks
        
        # Local gene context head (128bp resolution)
        self.local_head = Linear(dim_128bp, num_coexpression_tracks)
        
        # Long-range operon structure processor (pairwise embeddings)
        self.pair_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling over sequence
            nn.Flatten(),
            Linear(dim_pairwise, num_coexpression_tracks)
        )
        
        # Learnable per-track scaling (AlphaGenome pattern)
        self.scale = Parameter(torch.zeros(num_coexpression_tracks))
        
        # Track means for scaling
        self.register_buffer('track_means', torch.ones(num_coexpression_tracks))
    
    def forward(self, embeds_128bp: torch.Tensor, embeds_pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_128bp: [batch, seq_len//128, dim_128bp] - Gene-level features
            embeds_pair: [batch, seq_len//2048, seq_len//2048, dim_pairwise] - Pairwise interactions
        
        Returns:
            [batch, seq_len//128, num_tracks] - Co-expression predictions
        """
        # Local gene-level predictions
        local_predictions = self.local_head(embeds_128bp)
        
        # Global operon organization features
        pair_features = embeds_pair.permute(0, 3, 1, 2)  # [batch, dim, height, width]
        global_features = self.pair_processor(pair_features)  # [batch, num_tracks]
        
        # Broadcast global features to sequence length
        seq_len = local_predictions.shape[1]
        global_broadcast = global_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine local and global predictions
        combined = local_predictions + global_broadcast
        
        # Apply softplus + learnable scaling (AlphaGenome pattern)
        return F.softplus(combined) * F.softplus(self.scale)


# Advanced regulation heads (Phase 2)

class RiboswitchBindingHead(nn.Module):
    """
    Riboswitch ligand binding prediction head
    
    Based on riboswitch databases:
    - 55+ characterized riboswitch classes with ligand specificity
    - 200,000+ putative riboswitch sequences from Rfam
    
    Predicts binding probabilities for different metabolites.
    Uses 1bp embeddings for motif detection + 128bp for context.
    """
    
    def __init__(self, dim_1bp: int, dim_128bp: int, num_ligands: int = 30):
        super().__init__()
        self.num_ligands = num_ligands
        
        # Motif detection with 1D convolution (riboswitch aptamer domains are ~30-40 nt)
        self.motif_conv = Conv1d(dim_1bp, num_ligands, kernel_size=35, padding=17)
        
        # Context features from 128bp resolution
        self.context_head = Linear(dim_128bp, num_ligands)
        
    def forward(self, embeds_1bp: torch.Tensor, embeds_128bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
            embeds_128bp: [batch, seq_len//128, dim_128bp]
        
        Returns:
            [batch, seq_len, num_ligands] - Binding probabilities
        """
        # Motif features from 1bp resolution
        x_1bp = embeds_1bp.transpose(1, 2)  # [batch, dim, seq_len]
        motif_features = self.motif_conv(x_1bp).transpose(1, 2)  # [batch, seq_len, num_ligands]
        
        # Context features from 128bp resolution
        context_features = self.context_head(embeds_128bp)
        context_upsampled = repeat(context_features, 'b n c -> b (n repeat) c', repeat=128)
        
        # Handle length mismatches
        seq_len = motif_features.shape[1]
        if context_upsampled.shape[1] != seq_len:
            context_upsampled = context_upsampled[:, :seq_len]
        
        # Combine motif and context features
        binding_logits = motif_features + context_upsampled
        
        # Sigmoid for binding probabilities
        return torch.sigmoid(binding_logits)


class SRNATargetHead(nn.Module):
    """
    Small RNA target prediction head
    
    Based on bacterial sRNA data:
    - sRNAdb: 1,500+ bacterial sRNAs with targets  
    - CopraRNA: 50,000+ sRNA-target interactions
    - PAR-CLIP/CLASH: Hfq-mediated interaction data
    
    Predicts interaction strength with known sRNAs.
    """
    
    def __init__(self, dim_1bp: int, dim_128bp: int, num_srnas: int = 100):
        super().__init__()
        self.num_srnas = num_srnas
        
        # Base-pairing features (1bp resolution)
        self.base_pair_head = Linear(dim_1bp, num_srnas)
        
        # Context features (128bp resolution)  
        self.context_head = Linear(dim_128bp, num_srnas)
        
        # Learnable scaling
        self.scale = Parameter(torch.zeros(num_srnas))
    
    def forward(self, embeds_1bp: torch.Tensor, embeds_128bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
            embeds_128bp: [batch, seq_len//128, dim_128bp]
        
        Returns:
            [batch, seq_len, num_srnas] - Interaction strength scores
        """
        # Base-pairing predictions
        base_pair_features = self.base_pair_head(embeds_1bp)
        
        # Context features
        context_features = self.context_head(embeds_128bp)
        context_upsampled = repeat(context_features, 'b n c -> b (n repeat) c', repeat=128)
        
        # Handle length mismatches
        seq_len = base_pair_features.shape[1]
        if context_upsampled.shape[1] != seq_len:
            context_upsampled = context_upsampled[:, :seq_len]
        
        # Combine features
        interaction_logits = base_pair_features + context_upsampled
        
        # Softplus for positive interaction strength
        return F.softplus(interaction_logits) * F.softplus(self.scale)


# Systems-level heads (Phase 3)

class TerminationHead(nn.Module):
    """
    Transcription termination prediction head
    
    Based on termination data:
    - TransTermHP: 50,000+ intrinsic terminators
    - Rho termination: 1,200+ Rho-dependent sites
    - Term-seq: Genome-wide termination mapping
    
    Predicts termination probability and mechanism type.
    """
    
    def __init__(self, dim_1bp: int):
        super().__init__()
        # 3 classes: Intrinsic, Rho-dependent, No termination
        self.termination_head = Linear(dim_1bp, 3)
    
    def forward(self, embeds_1bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
        
        Returns:
            [batch, seq_len, 3] - Termination type probabilities
        """
        termination_logits = self.termination_head(embeds_1bp)
        return F.softmax(termination_logits, dim=-1)


class PathwayActivityHead(nn.Module):
    """
    Metabolic pathway activity prediction head
    
    Based on pathway databases:
    - KEGG Pathways: 478+ bacterial reference pathways
    - BioCyc Database: 20,000+ organism-specific variants
    - MetaCyc: 3,000+ experimentally validated pathways
    
    Predicts pathway completeness scores using sequence-level features.
    """
    
    def __init__(self, dim_128bp: int, dim_pairwise: int, num_pathways: int = 200):
        super().__init__()
        self.num_pathways = num_pathways
        
        # Gene activity aggregation
        self.gene_processor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Pool over sequence length
            nn.Flatten(),
            Linear(dim_128bp, num_pathways)
        )
        
        # Genome organization features
        self.organization_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Pool over 2D pairwise
            nn.Flatten(), 
            Linear(dim_pairwise, num_pathways)
        )
        
        # Combining layer
        self.combiner = Linear(num_pathways * 2, num_pathways)
    
    def forward(self, embeds_128bp: torch.Tensor, embeds_pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_128bp: [batch, seq_len//128, dim_128bp]
            embeds_pair: [batch, seq_len//2048, seq_len//2048, dim_pairwise]
        
        Returns:
            [batch, num_pathways] - Pathway activity scores
        """
        # Gene activity features
        gene_features = self.gene_processor(embeds_128bp.transpose(1, 2))  # [batch, num_pathways]
        
        # Organization features
        org_features = self.organization_processor(embeds_pair.permute(0, 3, 1, 2))  # [batch, num_pathways]
        
        # Combine and predict
        combined = torch.cat([gene_features, org_features], dim=-1)
        pathway_scores = self.combiner(combined)
        
        return torch.sigmoid(pathway_scores)


class SecretionSignalHead(nn.Module):
    """
    Protein secretion signal prediction head
    
    Based on secretion databases:
    - SignalP: 100,000+ signal peptides across species
    - MacSyFinder: Secretion system component models
    - SecretomeP: Non-classical secretion predictions
    
    Predicts secretion system type (multi-label classification).
    """
    
    def __init__(self, dim_1bp: int, num_secretion_types: int = 8):
        super().__init__()
        # Common bacterial secretion systems: T1SS, T2SS, T3SS, T4SS, T5SS, T6SS, Tat, Sec
        self.signal_head = Linear(dim_1bp, num_secretion_types)
    
    def forward(self, embeds_1bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
        
        Returns:
            [batch, seq_len, num_secretion_types] - Secretion type probabilities
        """
        signal_logits = self.signal_head(embeds_1bp)
        return torch.sigmoid(signal_logits)  # Multi-label classification