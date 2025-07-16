"""
Core bactagenome model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

# Import AlphaGenome components
from ..model.components import (
    TransformerUnet,
    OrganismEmbedding,
    OutputEmbedding,
    OutputPairEmbedding,
    Embeds,
    get_function_arg_names,
    is_disjoint
)

from .config import BactaGenomeConfig
from .heads import (
    PromoterStrengthHead,
    RBSEfficiencyHead,
    OperonCoregulationHead,
    RiboswitchBindingHead,
    SRNATargetHead,
    TerminationHead,
    PathwayActivityHead,
    SecretionSignalHead,
)

from collections import defaultdict
from torch.nn import ModuleDict


class BactaGenome(nn.Module):
    """
    bactagenome: Bacterial genome modeling with AlphaGenome architecture
    
    Adapts AlphaGenome's encoder-decoder architecture for bacterial-specific
    output modalities optimized for synthetic biology applications.
    """
    
    def __init__(
        self,
        config: Optional[BactaGenomeConfig] = None,
        **kwargs
    ):
        super().__init__()
        
        self.config = config or BactaGenomeConfig(**kwargs)
        
        # Core transformer-unet backbone
        self.transformer_unet = TransformerUnet(
            dims=self.config.dims,
            basepairs=self.config.basepairs,
            dna_embed_width=self.config.dna_embed_width,
            transformer_kwargs=self.config.transformer_kwargs
        )
        
        # Organism embeddings for bacterial species
        first_dim, *_, last_dim = self.config.dims
        self.organism_embed = OrganismEmbedding(last_dim, self.config.num_organisms)
        
        # Output embeddings
        self.outembed_128bp = OutputEmbedding(last_dim, self.config.num_organisms)
        self.outembed_1bp = OutputEmbedding(first_dim, self.config.num_organisms, skip_dim=2*last_dim)
        self.outembed_pair = OutputPairEmbedding(
            self.transformer_unet.transformer.dim_pairwise, 
            self.config.num_organisms
        )
        
        # Dimensions for heads (after output embedding doubles them)
        self.num_organisms = self.config.num_organisms
        self.dim_1bp = 2 * first_dim  # After output embedding: 2 * 64 = 128
        self.dim_128bp = 2 * last_dim  # After output embedding: 2 * 96 = 192
        self.dim_contacts = self.transformer_unet.transformer.dim_pairwise
        
        # Bacterial-specific prediction heads
        self.heads = ModuleDict()
        self.head_forward_arg_names = defaultdict(dict)
        self.head_forward_arg_maps = defaultdict(dict)
    
    @property
    def total_parameters(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def add_head(
        self,
        organism: str,
        head_name: str,
        head: nn.Module,
        head_input_kwarg_names: Optional[str | tuple[str, ...]] = None
    ):
        """Add a prediction head for a specific organism"""
        if isinstance(head_input_kwarg_names, str):
            head_input_kwarg_names = (head_input_kwarg_names,)
        
        if organism not in self.heads:
            self.heads[organism] = ModuleDict()
        
        self.heads[organism][head_name] = head
        
        # Store input argument names
        head_forward_arg_names = get_function_arg_names(head.forward)
        self.head_forward_arg_names[organism][head_name] = (
            head_input_kwarg_names or head_forward_arg_names
        )
        
        # Create argument mapping
        head_forward_arg_map = {}
        if head_input_kwarg_names:
            head_forward_arg_map = dict(zip(head_input_kwarg_names, head_forward_arg_names))
        
        self.head_forward_arg_maps[organism][head_name] = head_forward_arg_map
    
    def add_bacterial_heads(self, organism: str, phase: int = 0):
        """Add bacterial-specific prediction heads for an organism"""
        
        if phase == 0:
            # Phase 0: RegulonDB-based heads (gene expression, density, operon membership)
            from .heads import GeneExpressionHead, GeneDensityHead, OperonMembershipHead
            
            bacterial_heads = [
                ("gene_expression", GeneExpressionHead(self.dim_1bp)),
                ("gene_density", GeneDensityHead(self.dim_128bp)),
                ("operon_membership", OperonMembershipHead(self.dim_1bp))
            ]
            
            for head_name, head in bacterial_heads:
                self.add_head(organism, head_name, head)
            
        elif phase == 1:
            # Phase 1: Core expression control heads (based on RegulonDB data)
            bacterial_heads = [
                ("gene_expression_control", PromoterStrengthHead(
                    self.dim_1bp, self.dim_128bp, num_conditions=50  # RegulonDB conditions
                )),
                ("translation_efficiency", RBSEfficiencyHead(self.dim_1bp)),
                ("gene_coregulation", OperonCoregulationHead(
                    self.dim_128bp, self.dim_contacts, num_coexpression_tracks=20
                )),
            ]
            
            for head_name, head in bacterial_heads:
                self.add_head(organism, head_name, head)
                
        elif phase == 2:
            # Phase 2: Add advanced regulation heads
            bacterial_heads = [
                ("gene_expression_control", PromoterStrengthHead(
                    self.dim_1bp, self.dim_128bp, num_conditions=50
                )),
                ("translation_efficiency", RBSEfficiencyHead(self.dim_1bp)),
                ("gene_coregulation", OperonCoregulationHead(
                    self.dim_128bp, self.dim_contacts, num_coexpression_tracks=20
                )),
                ("riboswitch_binding", RiboswitchBindingHead(
                    self.dim_1bp, self.dim_128bp, num_ligands=30
                )),
                ("srna_target_prediction", SRNATargetHead(
                    self.dim_1bp, self.dim_128bp, num_srnas=100
                )),
            ]
            
            for head_name, head in bacterial_heads:
                self.add_head(organism, head_name, head)
                
        elif phase == 3:
            # Phase 3: All heads including systems-level
            bacterial_heads = [
                ("gene_expression_control", PromoterStrengthHead(
                    self.dim_1bp, self.dim_128bp, num_conditions=50
                )),
                ("translation_efficiency", RBSEfficiencyHead(self.dim_1bp)),
                ("gene_coregulation", OperonCoregulationHead(
                    self.dim_128bp, self.dim_contacts, num_coexpression_tracks=20
                )),
                ("riboswitch_binding", RiboswitchBindingHead(
                    self.dim_1bp, self.dim_128bp, num_ligands=30
                )),
                ("srna_target_prediction", SRNATargetHead(
                    self.dim_1bp, self.dim_128bp, num_srnas=100
                )),
                ("transcription_termination", TerminationHead(self.dim_1bp)),
                ("metabolic_pathway_activity", PathwayActivityHead(
                    self.dim_128bp, self.dim_contacts, num_pathways=200
                )),
                ("protein_secretion_signals", SecretionSignalHead(
                    self.dim_1bp, num_secretion_types=8
                )),
            ]
            
            for head_name, head in bacterial_heads:
                self.add_head(organism, head_name, head)
                
        else:
            raise ValueError(f"Unsupported phase: {phase}. Must be 0, 1, 2, or 3.")
    
    def get_embeds(
        self,
        seq: torch.Tensor,
        organism_index: torch.Tensor
    ) -> Embeds:
        """Get embeddings at different resolutions"""
        organism_embed = self.organism_embed(organism_index)
        
        # Forward through transformer-unet
        unet_out, single, pairwise = self.transformer_unet(
            seq, pre_attend_embed=organism_embed
        )
        
        # Organism-specific output embeddings
        embeds_128bp = self.outembed_128bp(single, organism_index)
        embeds_1bp = self.outembed_1bp(unet_out, organism_index, embeds_128bp)
        embeds_pair = self.outembed_pair(pairwise, organism_index)
        
        return Embeds(embeds_1bp, embeds_128bp, embeds_pair)
    
    def forward(
        self,
        seq: torch.Tensor,
        organism_index: int | torch.Tensor,
        return_embeds: bool = False,
        **head_kwargs
    ) -> Dict[str, Any] | Embeds:
        """
        Forward pass through bactagenome
        
        Args:
            seq: DNA sequence tensor [batch, length]
            organism_index: Organism indices [batch] or single int
            return_embeds: Whether to return embeddings only
            **head_kwargs: Additional arguments for prediction heads
            
        Returns:
            Dictionary of predictions by organism and head, or embeddings
        """
        # Handle integer organism index
        if isinstance(organism_index, int):
            batch = seq.shape[0]
            organism_index = torch.full((batch,), organism_index, device=seq.device)
        
        # Get multi-resolution embeddings
        embeds = self.get_embeds(seq, organism_index)
        
        # Return embeddings if requested or no heads available
        if return_embeds or len(self.heads) == 0:
            return embeds
        
        # Prepare inputs for prediction heads
        embeds_1bp, embeds_128bp, embeds_pair = embeds
        
        head_inputs = {
            "embeds_1bp": embeds_1bp,
            "embeds_128bp": embeds_128bp,
            "embeds_pair": embeds_pair,
            "organism_index": organism_index,
        }
        
        # Ensure no argument conflicts
        assert is_disjoint(set(head_inputs.keys()), set(head_kwargs.keys()))
        head_inputs.update(**head_kwargs)
        
        # Run prediction heads
        predictions = {}
        
        for organism, heads in self.heads.items():
            organism_predictions = {}
            
            for head_name, head in heads.items():
                # Get required inputs for this head
                head_arg_names = self.head_forward_arg_names[organism][head_name]
                head_arg_map = self.head_forward_arg_maps[organism][head_name]
                
                # Extract inputs
                head_kwargs_filtered = {
                    head_arg: head_inputs[head_arg] for head_arg in head_arg_names
                }
                
                # Apply argument mapping
                head_kwargs_filtered = {
                    (head_arg_map.get(k, k)): v 
                    for k, v in head_kwargs_filtered.items()
                }
                
                # Forward through head
                head_output = head(**head_kwargs_filtered)
                organism_predictions[head_name] = head_output
            
            predictions[organism] = organism_predictions
        
        return predictions