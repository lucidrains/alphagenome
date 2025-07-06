"""
Configuration classes for bactagenome model
"""
import inspect
from typing import Dict, Any, Optional


class BactaGenomeConfig:
    """Configuration for bactagenome model"""
    
    model_type = "bactagenome"
    
    def __init__(
        self,
        dims=(768, 896, 1024, 1152, 1280, 1408, 1536),
        basepairs=5,
        dna_embed_width=15,
        num_organisms=7,  # 7 bacterial species
        context_length=98304,  # 100K bp context
        output_1bp_bins=98304,  # Full length output
        output_128bp_bins=768,  # 98304/128 = 768 bins
        pairwise_size=48,  # 98304/2048 = 48
        transformer_kwargs=None,
        head_specs=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.dims = tuple(dims)
        self.basepairs = basepairs
        self.dna_embed_width = dna_embed_width
        self.num_organisms = num_organisms
        self.context_length = context_length
        self.output_1bp_bins = output_1bp_bins
        self.output_128bp_bins = output_128bp_bins
        self.pairwise_size = pairwise_size
        
        self.transformer_kwargs = transformer_kwargs or {
            "depth": 8,
            "heads": 6,
            "dim_head_qk": 128,
            "dim_head_v": 192,
            "dropout": 0.1,
            "ff_expansion_factor": 2.0,
            "max_positions": 8192,  # Sufficient for 100K context (98304/16 = 6144)
            "dim_pairwise": 128,
            "pairwise_every_num_single_blocks": 2,
            "single_to_pairwise_heads": 32,
            "pool_size": 16,
        }
        
        self.head_specs = head_specs or {
            "E_coli_K12": {
                "num_conditions": 50,  # Different growth conditions
                "num_pathways": 100,   # KEGG pathways
                "num_srnas": 200,      # Known sRNAs
                "num_secretion_types": 8,  # T1SS, T2SS, etc.
            },
            "B_subtilis_168": {
                "num_conditions": 40,
                "num_pathways": 80,
                "num_srnas": 150,
                "num_secretion_types": 6,
            },
            "Salmonella_enterica": {
                "num_conditions": 45,
                "num_pathways": 90,
                "num_srnas": 180,
                "num_secretion_types": 7,
            },
            "Pseudomonas_aeruginosa": {
                "num_conditions": 35,
                "num_pathways": 85,
                "num_srnas": 160,
                "num_secretion_types": 8,
            },
            "Mycobacterium_tuberculosis": {
                "num_conditions": 30,
                "num_pathways": 70,
                "num_srnas": 120,
                "num_secretion_types": 5,
            },
            "Streptococcus_pyogenes": {
                "num_conditions": 25,
                "num_pathways": 60,
                "num_srnas": 100,
                "num_secretion_types": 4,
            },
            "Synechocystis_sp": {
                "num_conditions": 20,
                "num_pathways": 75,
                "num_srnas": 90,
                "num_secretion_types": 3,
            },
        }
    
    def get_head_spec(self, organism: str) -> Dict[str, Any]:
        """Get head specifications for a specific organism"""
        if organism not in self.head_specs:
            raise ValueError(f"Organism '{organism}' not found in head_specs.")
        return {**self.head_specs[organism]}
    
    def get_phase1_config(self) -> Dict[str, Any]:
        """Get configuration for Phase 1 training (proof of concept)"""
        return {
            "species": ["E_coli_K12"],
            "modalities": ["promoter_strength", "rbs_efficiency", "operon_coregulation"],
            "context_length": self.context_length,
            "output_length": self.output_1bp_bins,
            "batch_size": 16,
            "learning_rate": 0.002,
            "total_steps": 8000,
        }
    
    def get_phase2_config(self) -> Dict[str, Any]:
        """Get configuration for Phase 2 training (multi-species)"""
        return {
            "species": list(self.head_specs.keys()),
            "modalities": [
                "promoter_strength", "rbs_efficiency", "operon_coregulation",
                "riboswitch_binding", "srna_target", "termination",
                "pathway_activity", "secretion_signal"
            ],
            "context_length": self.context_length,
            "output_length": self.output_1bp_bins,
            "batch_size": 24,
            "learning_rate": 0.003,
            "total_steps": 12000,
        }


def get_function_arg_names(fn):
    signature = inspect.signature(fn)
    parameters = signature.parameters.values()
    return [p.name for p in parameters]


def is_disjoint(a: set, b: set):
    return not any(a.intersection(b))

