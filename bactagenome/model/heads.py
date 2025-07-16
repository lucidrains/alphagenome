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


def tracks_scaled_predictions(embeddings: torch.Tensor, head: nn.Module, scale_param: Parameter) -> torch.Tensor:
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


def soft_clip(x: torch.Tensor) -> torch.Tensor:
    """Soft clipping for numerical stability in loss calculations"""
    return torch.where(x > 10.0, 2 * torch.sqrt(x * 10.0) - 10.0, x)


def multinomial_poisson_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                           multinomial_resolution: int = 128,
                           multinomial_weight: float = 5.0) -> torch.Tensor:
    """
    AlphaGenome-style loss combining multinomial and Poisson components
    """
    batch_size, seq_len, num_tracks = predictions.shape
    
    # Adjust resolution if sequence length is not divisible
    if seq_len % multinomial_resolution != 0:
        # Use adaptive resolution or fall back to simple segments
        num_segments = max(1, seq_len // 64)  # At least 1 segment, up to seq_len//64
        segment_size = seq_len // num_segments
    else:
        segment_size = multinomial_resolution
        num_segments = seq_len // segment_size
    
    # If sequence is too small, fall back to MSE loss
    if num_segments < 2:
        mse_loss = torch.nn.MSELoss()
        return mse_loss(predictions, targets)
    
    # Reshape for segment-wise loss calculation
    # Truncate to make evenly divisible
    effective_length = num_segments * segment_size
    pred_truncated = predictions[:, :effective_length]
    target_truncated = targets[:, :effective_length]
    
    pred_segments = pred_truncated.reshape(batch_size, num_segments, segment_size, num_tracks)
    target_segments = target_truncated.reshape(batch_size, num_segments, segment_size, num_tracks)
    
    # Sum over segment positions
    sum_pred = torch.sum(pred_segments, dim=2, keepdim=True)  # [batch, num_segments, 1, num_tracks]
    sum_target = torch.sum(target_segments, dim=2, keepdim=True)
    
    # Poisson loss on segment totals
    poisson_loss = torch.sum(sum_pred - sum_target * torch.log(sum_pred + 1e-7))
    
    # Multinomial loss on within-segment distributions
    multinomial_prob = pred_segments / (sum_pred + 1e-7)
    positional_loss = torch.sum(-target_segments * torch.log(multinomial_prob + 1e-7))
    
    return poisson_loss / segment_size + multinomial_weight * positional_loss


class GeneExpressionHead(nn.Module):
    """
    Gene expression prediction head using realistic RegulonDB data
    
    Based on actual RegulonDB analysis and AlphaGenome techniques:
    - Uses log-normalized TPM/FPKM values from geneExpression.bson
    - Predicts continuous expression levels with learnable scaling
    - AlphaGenome-style softplus activation and scaling parameters
    """
    
    def __init__(self, dim_1bp: int, num_tracks: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_tracks = num_tracks
        
        # Linear projection to track outputs
        self.linear = Linear(dim_1bp, num_tracks)
        
        # Learnable per-track scaling parameters (AlphaGenome style)
        self.scale = Parameter(torch.zeros(num_tracks))
        
        # Track means for scaling (will be set during training)
        self.register_buffer('track_means', torch.ones(num_tracks))
        
        # Initialize with Xavier uniform for stable training
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, embeds_1bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp] - 1bp resolution embeddings
        
        Returns:
            [batch, seq_len, num_tracks] - Expression values with learnable scaling
        """
        return tracks_scaled_predictions(embeds_1bp, self.linear, self.scale)
    
    def set_track_means(self, track_means: torch.Tensor):
        """Set track means for proper scaling"""
        self.track_means.data = track_means


class GeneDensityHead(nn.Module):
    """
    Gene density prediction head - counts genes per genomic region
    
    Based on RegulonDB gene distribution and AlphaGenome techniques:
    - Predicts number of genes in 128bp bins
    - Uses AlphaGenome-style count modeling with softplus activation
    - Learnable scaling for better count predictions
    """
    
    def __init__(self, dim_128bp: int, num_tracks: int = 1, dropout: float = 0.1):
        super().__init__()
        self.num_tracks = num_tracks
        
        # Linear projection to track outputs
        self.linear = Linear(dim_128bp, num_tracks)
        
        # Learnable per-track scaling parameters (AlphaGenome style)
        self.scale = Parameter(torch.zeros(num_tracks))
        
        # Track means for scaling
        self.register_buffer('track_means', torch.ones(num_tracks))
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.linear.weight)
    
    def forward(self, embeds_128bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_128bp: [batch, seq_len//128, dim_128bp] - 128bp resolution embeddings
        
        Returns:
            [batch, seq_len//128, num_tracks] - Gene counts with learnable scaling
        """
        return tracks_scaled_predictions(embeds_128bp, self.linear, self.scale)
    
    def set_track_means(self, track_means: torch.Tensor):
        """Set track means for proper scaling"""
        self.track_means.data = track_means


class OperonMembershipHead(nn.Module):
    """
    Operon membership prediction head - binary classification
    
    Based on RegulonDB operon structure:
    - Predicts whether a gene region is part of an operon (binary)
    - Uses 1bp resolution for precise gene boundary detection
    - Simple binary classification task achievable with available data
    """
    
    def __init__(self, dim_1bp: int, dropout: float = 0.1):
        super().__init__()
        
        # Binary classification head for operon membership
        self.projection = Linear(dim_1bp, 64)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = Linear(64, 1)
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
    
    def forward(self, embeds_1bp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp] - 1bp resolution embeddings
        
        Returns:
            [batch, seq_len, 1] - Operon membership probabilities [0,1]
        """
        x = self.projection(embeds_1bp)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        # Sigmoid for binary classification probabilities
        return torch.sigmoid(x)


# Compatibility wrapper classes for existing code
class PromoterStrengthHead(GeneExpressionHead):
    """Compatibility wrapper for GeneExpressionHead"""
    def __init__(self, dim_1bp: int, dim_128bp: int = None, num_conditions: int = 50, **kwargs):
        # Use num_conditions as num_tracks for multi-condition expression
        super().__init__(dim_1bp, num_tracks=num_conditions, **kwargs)

class RBSEfficiencyHead(GeneDensityHead):
    """Compatibility wrapper for GeneDensityHead"""
    def __init__(self, dim_1bp: int, **kwargs):
        # Convert 1bp to 128bp for gene density (typically dim_1bp is actually 128bp embedding)
        super().__init__(dim_128bp=dim_1bp, **kwargs)

class OperonCoregulationHead(OperonMembershipHead):
    """Compatibility wrapper for OperonMembershipHead"""
    def __init__(self, dim_128bp: int, dim_pairwise: int = None, num_coexpression_tracks: int = 20, **kwargs):
        # Convert 128bp to 1bp for membership classification
        super().__init__(dim_1bp=dim_128bp, **kwargs)

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


class RegulonDBHeadManager(nn.ModuleDict):
    """
    Head manager for RegulonDB-based bacterial genomics targets
    Uses achievable prediction tasks based on actual RegulonDB data analysis
    Compatible with existing BactaGenome model interface by inheriting from ModuleDict
    """
    
    def __init__(self, dim_1bp: int, dim_128bp: int, organism_name: str):
        # Initialize with RegulonDB-based heads
        heads_dict = {
            'gene_expression': GeneExpressionHead(dim_1bp),
            'gene_density': GeneDensityHead(dim_128bp), 
            'operon_membership': OperonMembershipHead(dim_1bp)
        }
        super().__init__(heads_dict)
        self.organism_name = organism_name
    
    @property
    def heads(self):
        """Return self for compatibility"""
        return self
        
    def forward(self, embeds_1bp: torch.Tensor, embeds_128bp: torch.Tensor) -> dict:
        """
        Forward pass through all heads
        
        Args:
            embeds_1bp: [batch, seq_len, dim_1bp]
            embeds_128bp: [batch, seq_len//128, dim_128bp]
            
        Returns:
            Dict of predictions for each target type
        """
        return {
            'gene_expression': self['gene_expression'](embeds_1bp),
            'gene_density': self['gene_density'](embeds_128bp),
            'operon_membership': self['operon_membership'](embeds_1bp)
        }
    
    def get_target_info(self) -> dict:
        """Return information about each target type"""
        return {
            'gene_expression': {
                'type': 'regression',
                'resolution': '1bp', 
                'loss': 'MSE',
                'description': 'Log-normalized TPM/FPKM values'
            },
            'gene_density': {
                'type': 'count_regression',
                'resolution': '128bp',
                'loss': 'MSE', 
                'description': 'Number of genes per genomic bin'
            },
            'operon_membership': {
                'type': 'binary_classification',
                'resolution': '1bp',
                'loss': 'BCE',
                'description': 'Binary operon membership'
            }
        }


class RegulonDBLossFunction(nn.Module):
    """
    Loss function for RegulonDB-based bacterial genomics targets
    Uses AlphaGenome-inspired loss functions for better training
    """
    
    def __init__(self, loss_weights: dict = None, use_alphgenome_loss: bool = True):
        super().__init__()
        
        # Default weights for each target
        self.loss_weights = loss_weights or {
            'gene_expression': 1.0,
            'gene_density': 1.0, 
            'operon_membership': 1.0
        }
        
        self.use_alphgenome_loss = use_alphgenome_loss
        
        # Standard loss functions as fallback
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions: dict, targets: dict, organism_name: str) -> tuple:
        """
        Compute losses for all targets
        
        Args:
            predictions: Dict of model predictions
            targets: Dict of target tensors
            organism_name: Name of organism (for compatibility)
            
        Returns:
            Tuple of (total_loss, individual_losses_dict)
        """
        individual_losses = {}
        total_loss = 0.0
        
        # Gene expression loss - use AlphaGenome-style for count-like data
        if 'gene_expression' in predictions and 'gene_expression' in targets:
            pred = predictions['gene_expression']
            target = targets['gene_expression']
            
            # Ensure compatible shapes
            if pred.shape != target.shape:
                min_len = min(pred.shape[1], target.shape[1])
                pred = pred[:, :min_len]
                target = target[:, :min_len]
            
            if self.use_alphgenome_loss and len(pred.shape) == 3 and pred.shape[1] >= 128:
                # Use multinomial+Poisson loss for sequence-level predictions
                loss = multinomial_poisson_loss(pred, target, multinomial_resolution=128)
            else:
                # Fallback to MSE for simple cases
                loss = self.mse_loss(pred, target)
            
            individual_losses['gene_expression'] = loss.item()
            total_loss += self.loss_weights['gene_expression'] * loss
        
        # Gene density loss - AlphaGenome-style for count data
        if 'gene_density' in predictions and 'gene_density' in targets:
            pred = predictions['gene_density']
            target = targets['gene_density']
            
            # Ensure compatible shapes
            if pred.shape != target.shape:
                min_len = min(pred.shape[1], target.shape[1])
                pred = pred[:, :min_len]
                target = target[:, :min_len]
            
            if self.use_alphgenome_loss and len(pred.shape) == 3 and pred.shape[1] >= 64:
                # Use multinomial+Poisson loss for count predictions
                resolution = min(64, pred.shape[1] // 8)  # Adaptive resolution
                loss = multinomial_poisson_loss(pred, target, multinomial_resolution=resolution)
            else:
                # Fallback to MSE
                loss = self.mse_loss(pred, target)
            
            individual_losses['gene_density'] = loss.item()
            total_loss += self.loss_weights['gene_density'] * loss
        
        # Operon membership loss (BCE for binary classification)
        if 'operon_membership' in predictions and 'operon_membership' in targets:
            pred = predictions['operon_membership']
            target = targets['operon_membership']
            
            # Ensure compatible shapes
            if pred.shape != target.shape:
                min_len = min(pred.shape[1], target.shape[1])
                pred = pred[:, :min_len]
                target = target[:, :min_len]
            
            loss = self.bce_loss(pred, target)
            individual_losses['operon_membership'] = loss.item()
            total_loss += self.loss_weights['operon_membership'] * loss
        
        return total_loss, individual_losses


def create_regulondb_bacterial_heads(dim_1bp: int, dim_128bp: int, organism_name: str) -> RegulonDBHeadManager:
    """
    Factory function to create RegulonDB-based bacterial heads
    
    Args:
        dim_1bp: Dimension of 1bp embeddings
        dim_128bp: Dimension of 128bp embeddings
        organism_name: Name of organism
        
    Returns:
        RegulonDBHeadManager instance
    """
    return RegulonDBHeadManager(
        dim_1bp=dim_1bp,
        dim_128bp=dim_128bp,
        organism_name=organism_name
    )


def integrate_regulondb_heads_with_model(model, organism_name: str):
    """
    Replace model heads with RegulonDB-based bacterial heads
    
    Args:
        model: BactaGenome model instance
        organism_name: Name of organism to replace heads for
    """
    if hasattr(model, 'heads') and organism_name in model.heads:
        # Get input dimensions from model (after output embedding doubling)
        if hasattr(model, 'dim_1bp') and hasattr(model, 'dim_128bp'):
            dim_1bp = model.dim_1bp
            dim_128bp = model.dim_128bp
        elif hasattr(model, 'config') and hasattr(model.config, 'dims'):
            # Calculate dimensions with output embedding doubling
            first_dim = model.config.dims[0]
            last_dim = model.config.dims[-1]
            dim_1bp = 2 * first_dim
            dim_128bp = 2 * last_dim
        else:
            # Default fallback
            dim_1bp = 1536
            dim_128bp = 3072
        
        # Keep the existing head configuration metadata
        existing_head_config = {
            'head_forward_arg_names': model.head_forward_arg_names[organism_name] if hasattr(model, 'head_forward_arg_names') else {},
            'head_forward_arg_maps': model.head_forward_arg_maps[organism_name] if hasattr(model, 'head_forward_arg_maps') else {}
        }
        
        # Replace with RegulonDB-based heads
        regulondb_head_manager = create_regulondb_bacterial_heads(dim_1bp, dim_128bp, organism_name)
        model.heads[organism_name] = regulondb_head_manager
        
        # Update head configuration for new heads
        if hasattr(model, 'head_forward_arg_names'):
            model.head_forward_arg_names[organism_name] = {
                'gene_expression': ['embeds_1bp'],
                'gene_density': ['embeds_128bp'],
                'operon_membership': ['embeds_1bp']
            }
        
        if hasattr(model, 'head_forward_arg_maps'):
            model.head_forward_arg_maps[organism_name] = {
                'gene_expression': {},
                'gene_density': {},
                'operon_membership': {}
            }
        
        print(f"Replaced {organism_name} heads with RegulonDB-based bacterial heads (1bp={dim_1bp}, 128bp={dim_128bp})")
    else:
        print(f"Warning: No existing heads found for {organism_name}")