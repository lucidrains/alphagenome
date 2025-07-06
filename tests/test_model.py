"""
Tests for bactagenome model components
"""

import pytest
import torch
from bactagenome import BactaGenome, BactaGenomeConfig
from bactagenome.model.heads import (
    PromoterStrengthHead,
    RBSEfficiencyHead, 
    OperonCoregulationHead
)


def test_bactagenome_config():
    """Test bactagenome configuration"""
    config = BactaGenomeConfig()
    
    assert config.context_length == 98304
    assert config.output_1bp_bins == 98304
    assert config.num_organisms == 7
    assert len(config.dims) == 7
    
    # Test head specs
    ecoli_spec = config.get_head_spec("E_coli_K12")
    assert "num_conditions" in ecoli_spec
    assert "num_pathways" in ecoli_spec


def test_bactagenome_model_creation():
    """Test bactagenome model creation"""
    config = BactaGenomeConfig(
        dims=(256, 384, 512),  # Smaller for testing
        num_organisms=2
    )
    
    model = BactaGenome(config)
    
    assert isinstance(model, BactaGenome)
    assert model.num_organisms == 2
    assert model.config.context_length == 98304


def test_bactagenome_forward_pass():
    """Test bactagenome forward pass"""
    config = BactaGenomeConfig(
        dims=(256, 384, 512),  # Smaller for testing
        num_organisms=2
    )
    
    model = BactaGenome(config)
    
    # Test embeddings only
    seq = torch.randint(0, 5, (2, 1024))  # Smaller sequence for testing
    organism_index = torch.tensor([0, 1])
    
    embeds = model(seq, organism_index, return_embeds=True)
    
    assert embeds.embeds_1bp.shape[0] == 2  # batch size
    assert embeds.embeds_128bp.shape[0] == 2
    assert embeds.embeds_pair.shape[0] == 2


def test_bacterial_heads():
    """Test bacterial prediction heads"""
    dim_1bp, dim_128bp, dim_pair = 512, 512, 128
    
    # Test promoter head
    promoter_head = PromoterStrengthHead(dim_1bp, dim_128bp, num_conditions=10)
    
    embeds_1bp = torch.randn(2, 128, dim_1bp)
    embeds_128bp = torch.randn(2, 1, dim_128bp)
    
    output = promoter_head(embeds_1bp, embeds_128bp)
    assert output.shape == (2, 128, 10)
    
    # Test RBS head
    rbs_head = RBSEfficiencyHead(dim_1bp)
    output = rbs_head(embeds_1bp)
    assert output.shape == (2, 128, 1)
    
    # Test operon head
    operon_head = OperonCoregulationHead(dim_128bp, dim_pair, num_genes=20)
    embeds_pair = torch.randn(2, 4, 4, dim_pair)
    
    output = operon_head(embeds_128bp, embeds_pair)
    assert output.shape == (2, 1, 20)


def test_add_bacterial_heads():
    """Test adding bacterial heads to model"""
    config = BactaGenomeConfig(
        dims=(256, 384, 512),
        num_organisms=1
    )
    
    model = BactaGenome(config)
    model.add_bacterial_heads("E_coli_K12")
    
    # Check that heads were added
    assert "E_coli_K12" in model.heads
    assert "promoter_strength" in model.heads["E_coli_K12"]
    assert "rbs_efficiency" in model.heads["E_coli_K12"]


def test_model_with_heads():
    """Test model forward pass with prediction heads"""
    config = BactaGenomeConfig(
        dims=(256, 384, 512),
        num_organisms=1
    )
    
    model = BactaGenome(config)
    model.add_bacterial_heads("E_coli_K12")
    
    seq = torch.randint(0, 5, (1, 1024))
    organism_index = torch.tensor([0])
    
    # Forward pass with heads
    predictions = model(seq, organism_index)
    
    assert "E_coli_K12" in predictions
    assert "promoter_strength" in predictions["E_coli_K12"]
    assert "rbs_efficiency" in predictions["E_coli_K12"]


def test_phase_configs():
    """Test phase-specific configurations"""
    config = BactaGenomeConfig()
    
    phase1_config = config.get_phase1_config()
    assert phase1_config["species"] == ["E_coli_K12"]
    assert len(phase1_config["modalities"]) == 3
    
    phase2_config = config.get_phase2_config()
    assert len(phase2_config["species"]) == 7
    assert len(phase2_config["modalities"]) == 8


if __name__ == "__main__":
    pytest.main([__file__])