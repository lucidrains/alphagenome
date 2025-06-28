import pytest

import torch
from alphagenome_pytorch.alphagenome import TransformerTower

def test_attention():

    transformer = TransformerTower(dim = 768, dim_pairwise = 128)

    single = torch.randn(2, 512, 768)

    single_repr, pairwise_repr = transformer(single)

    assert single_repr.shape == (2, 512, 768)
    assert pairwise_repr.shape == (2, 512 // 16, 512 // 16, 128)

def test_down_up():
    from alphagenome_pytorch.alphagenome import DownresBlock, UpresBlock
    down = DownresBlock(64)
    up = UpresBlock(64 + 128)

    x = torch.randn(1, 64, 8)
    assert up(down(x), x).shape == x.shape
