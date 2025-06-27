import pytest
import torch
from alphagenome_pytorch.alphagenome import TransformerTower

def test_attention():

    transformer = TransformerTower(dim = 768, dim_pairwise = 128)

    single = torch.randn(2, 512, 768)

    single_repr, pairwise_repr = transformer(single)

    assert single_repr.shape == (2, 512, 768)
    assert pairwise_repr.shape == (2, 512 // 16, 512 // 16, 128)
