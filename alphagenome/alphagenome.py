from __future__ import annotations
from functools import partial

import torch
from torch import nn, cat, stack, arange
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Module, ModuleList

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, einsum

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

# rotary, but with attenuation of short relative distance frequencies

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        max_positions = 8192
    ):
        super().__init__()
        num_freqs = dim // 2
        inv_freq = 1. / (arange(num_freqs).float() + torch.logspace(1, max_positions - num_freqs + 1, num_freqs))
        self.register_buffer('inv_freq', inv_freq)

    def forward(
        self,
        seq_len
    ):
        device = self.inv_freq.device
        t = arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = einsum(t, self.inv_freq, 'i , j -> i j')
        return cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# prenorm and sandwich norm - they use sandwich norm for single rep, prenorm for pairwise rep

class NormWrapper(Module):
    def __init__(
        self,
        dim,
        block: Module,
        dropout = 0.,
        sandwich = False
    ):
        super().__init__()
        self.block = block
        self.pre_rmsnorm = nn.RMSNorm(dim) # they use an interesting variant of batchnorm, batch-rmsnorm. craft later and make sure it works distributed

        self.post_block_dropout = nn.Dropout(dropout)
        self.post_rmsnorm = nn.RMSNorm(dim) if sandwich else nn.Identity()

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
        dim_pairwise = None,
        softclamp_value = 5. # they employ attention softclamping
    ):
        super().__init__()
        dim_pairwise = default(dim_pairwise, dim)

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
            nn.RMSNorm(dim_pairwise), # replace with BatchRMSNorm once crafted
            nn.GELU(),
            LinearNoBias(dim_pairwise, heads),
            Rearrange('b i j h -> b h i j')
        )
        # variables

        self.qkv_dim_splits = qkv_proj_dim_out
        self.softclamp_value = softclamp_value

    def forward(
        self,
        x,
        pairwise = None, # Float['b i j dp']
        rotary_emb = None
    ):

        q, k, v = self.to_qkv(x).split(self.qkv_dim_splits, dim = -1)

        # they use multi-query attention, with only 1 key / value head - pretty unconventional, but maybe enough for genomic modeling

        q = self.split_q_heads(q)

        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)

        q = q * self.scale

        # maybe rotary

        if exists(rotary_emb):
            q, k = tuple(apply_rotary_pos_emb(rotary_emb, t) for t in (q, k))

        # similarities

        sim = einsum(q, k, 'b h i d, b j d -> b h i j')

        # add attention bias + softclamping

        if exists(pairwise):
            attn_bias = self.to_attn_bias(pairwise)

            assert divisible_by(sim.shape[-1], attn_bias.shape[-1])
            expand_factor = sim.shape[-1] // attn_bias.shape[-1]

            attn_bias = repeat(attn_bias, 'b h i j -> b h (i r1) (j r2)', r1 = expand_factor, r2 = expand_factor)

            sim = softclamp(sim + attn_bias, value = self.softclamp_value)

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        out = einsum(attn, v, 'b h i j, b j d -> b h i d')

        out = self.merge_heads(out)
        return self.to_out(out)

# pairwise attention is a single headed attention across rows, they said columns did not help

class PairwiseRowAttention(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** -0.5

        self.to_qk = LinearNoBias(dim, dim * 2)
        self.to_v = Linear(dim, dim)

    def forward(
        self,
        x
    ):

        q, k = self.to_qk(x).chunk(2, dim = -1)
        v = self.to_v(x)

        # similarity

        sim = einsum(q, k, 'b n i d, b n j d -> b n i j')

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        return einsum(attn, v, 'b n i j, b n j d -> b n i d')

# feedforward for both single and pairwise

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
        max_positions = 8192,
        dim_pairwise = None,
        pairwise_every_num_single_blocks = 2, # how often to do a pairwise block
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        dim_pairwise = default(dim_pairwise, dim)

        layers = []

        self.pairwise_every = pairwise_every_num_single_blocks

        self.rotary_emb = RotaryEmbedding(dim_head_qk, max_positions = max_positions)

        for layer_index in range(depth):

            attn = Attention(dim = dim, dim_head_qk = dim_head_qk, dim_head_v = dim_head_v, heads = heads, dim_pairwise = dim_pairwise)

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

            attn = NormWrapper(dim = dim, block = attn, dropout = dropout, sandwich = True)
            ff = NormWrapper(dim = dim, block = ff, dropout = dropout, sandwich = True)

            # maybe pairwise

            pairwise_attn, pairwise_ff = None, None

            if divisible_by(layer_index, self.pairwise_every):
                pairwise_attn = PairwiseRowAttention(dim_pairwise)
                pairwise_ff = FeedForward(dim = dim_pairwise, expansion_factor = ff_expansion_factor)

                pairwise_attn = NormWrapper(dim = dim_pairwise, block = pairwise_attn)
                pairwise_ff = NormWrapper(dim = dim_pairwise, block = pairwise_ff)

            # add to layers

            layers.append(ModuleList([
                attn,
                ff,
                pairwise_attn,
                pairwise_ff
            ]))


        self.layers = ModuleList(layers)

    def forward(
        self,
        single,
        pairwise = None
    ):

        seq_len = single.shape[1]

        rotary_emb = self.rotary_emb(seq_len)

        for (
            attn,
            ff,
            maybe_pairwise_attn,
            maybe_pairwise_ff
        ) in self.layers:

            single = attn(single, rotary_emb = rotary_emb, pairwise = pairwise)
            single = ff(single)

            if exists(maybe_pairwise_attn):
                pairwise = maybe_pairwise_attn(pairwise)
                pairwise = maybe_pairwise_ff(pairwise)

        return single, pairwise

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
        self.dim_input = dim_input
        self.conv = nn.Conv1d(dim_input, dim, width, padding = width // 2)
        self.pointwise = nn.Conv1d(dim, dim, 1)

    def forward(
        self,
        seq # Int['b n']
    ):
        onehot = F.one_hot(seq, num_classes = self.dim_input).float()
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
        dim_pairwise = None,
        transformer_kwargs: dict = dict()
    ):
        super().__init__()
        assert is_odd(dna_embed_width)

        self.to_dna_embed = DNAEmbed(dim, dim_input = basepairs, width = dna_embed_width)

        self.transformer = Transformer(
            dim = dim,
            dim_pairwise = dim_pairwise,
            **transformer_kwargs
        )

    def forward(
        self,
        seq,
        pairwise
    ):

        dna_embed = self.to_dna_embed(seq)

        attended = self.transformer(dna_embed, pairwise)

        return attended
