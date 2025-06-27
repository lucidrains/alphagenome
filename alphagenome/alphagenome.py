from __future__ import annotations
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList

from einops.layers.torch import Rearrange
from einops import rearrange, einsum

# ein notation

# b - batch
# h - heads
# n - sequence
# d - feature dimension

# constants

LinearNoBias = partial(Linear, bias = False)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        self.merge_heads = Rearraneg('b h n d -> b n (h d)')

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

    def forward(
        self,
        x
    ):

        qkv = self.to_qkv(x)
        q, k, v = self.split_heads(qkv)

        q = q * self.scale
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        return self.to_out(out)

# classes

class AlphaGenome(Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        seq
    ):
        return seq
