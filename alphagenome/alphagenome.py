from __future__ import annotations
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Module, ModuleList

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

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(num):
    return not divisible_by(num, 2)

def is_even(num):
    return divisible_by(num, 2)

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value = 5.):
    return (t / value).tanh() * value

# sandwich norm

class SandwichNorm(Module):
    def __init__(
        self,
        dim,
        block: Module,
        dropout = 0.
    ):
        super().__init__()
        self.block = block
        self.pre_rmsnorm = nn.RMSNorm(dim) # they use an interesting variant of batchnorm, batch-rmsnorm. craft later and make sure it works distributed

        self.post_block_dropout = nn.Dropout(dropout)
        self.post_rmsnorm = nn.RMSNorm(dim)

    def forward(
        self,
        x,
        **kwargs
    ):
        residual = x
        x = self.pre_rmsnorm(x)
        out = self.block(x, **kwargs)
        out = self.post_block_dropout(out)
        return self.post_rmsnorm(out) + residual

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dim_head_qk = 128,
        dim_head_v = 192,
        softclamp_value = 5. # they employ attention softclamping
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        qkv_proj_dim_out = (dim_head_qk * heads, dim_head_qk, dim_head_v)

        # splitting and merging of attention heads

        self.split_q_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # projections

        self.to_qkv = LinearNoBias(dim, sum(qkv_proj_dim_out))
        self.to_out = LinearNoBias(dim_head_v * heads, dim)

        # they add layernorms to queries, keys, and interestingly enough, values as well. first time i've seen this

        self.q_norm = nn.LayerNorm(dim_head_qk, bias = False)
        self.k_norm = nn.LayerNorm(dim_head_qk, bias = False)
        self.v_norm = nn.LayerNorm(dim_head_v, bias = False)

        # to attention bias

        self.to_attn_bias = Sequential(
            nn.RMSNorm(dim), # replace with BatchRMSNorm once crafted
            nn.GELU(),
            LinearNoBias(dim, heads),
            Rearrange('b i j h -> b h i j')
        )
        # variables

        self.qkv_dim_splits = qkv_proj_dim_out
        self.softclamp_value = softclamp_value

    def forward(
        self,
        x,
        pairwise_feats = None # Float['b i j dp']
    ):

        q, k, v = self.to_qkv(x).split(self.qkv_dim_splits, dim = -1)

        # they use multi-query attention, with only 1 key / value head - pretty unconventional, but maybe enough for genomic modeling

        q = self.split_q_heads(q)

        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)

        q = q * self.scale
        sim = einsum(q, k, 'b h i d, b j d -> b h i j')

        # add attention bias + softclamping

        if exists(pairwise_feats):
            attn_bias = self.to_attn_bias(pairwise_feats)

            sim = softclamp(sim + attn_bias, clamp_value = self.softclamp_value)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b j d -> b h i d')

        out = self.merge_heads(out)
        return self.to_out(out)

# feedforward

def FeedForward(
    dim,
    *,
    dropout = 0.,
    expansion_factor = 2.,  # they only do expansion factor of 2, no glu
):
    dim_inner = int(dim * expansion_factor)

    return Sequential(
        Linear(dim, dim_inner),
        nn.ReLU(),
        nn.Dropout(dropout),
        Linear(dim_inner, dim)
    )

# transformer

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth = 8,
        heads = 8,
        dim_head_qk = 128,
        dim_head_v = 192,
        dropout = 0.,
        ff_expansion_factor = 2.,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        layers = []

        for _ in range(depth):
            attn = Attention(dim = dim, dim_head_qk = dim_head_qk, dim_head_v = dim_head_v, heads = heads)

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

            layers.append(ModuleList([
                SandwichNorm(dim = dim, block = attn, dropout = dropout),
                SandwichNorm(dim = dim, block = ff, dropout = dropout),
            ]))

        self.layers = ModuleList(layers)

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x

# embedding

class DNAEmbed(Module):
    def __init__(
        self,
        dim,
        dim_input = 5, # 5 basepairs
        width = 15
    ):
        super().__init__()
        assert is_odd(width)
        self.conv = nn.Conv1d(dim_input, dim, width, padding = width // 2)
        self.pointwise = nn.Conv1d(dim, dim, 1)

    def forward(
        self,
        seq # Int['b n']
    ):
        onehot = F.one_hot(seq, num_classes = 5).float()
        x = rearrange(onehot, 'b n d -> b d n')

        out = self.conv(x)
        out = out + self.pointwise(out)
        return rearrange(out, 'b d n -> b n d')

# classes

class AlphaGenome(Module):
    def __init__(
        self,
        dim = 768,
        basepairs = 5,
        dna_embed_width = 15,
        transformer_kwargs: dict = dict()
    ):
        super().__init__()
        assert is_odd(dna_embed_width)

        self.to_dna_embed = DNAEmbed(dim, dim_input = basepairs, width = dna_embed_width)

        self.transformer = Transformer(
            dim = dim,
            **transformer_kwargs
        )

    def forward(
        self,
        seq
    ):

        dna_embed = self.to_dna_embed(seq)

        attended = self.transformer(dna_embed)

        return attended
