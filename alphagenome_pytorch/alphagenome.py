from __future__ import annotations

import inspect
from functools import partial
from collections import namedtuple

import torch
import torch.distributed as dist
from torch import tensor, arange, nn, cat, stack, arange, logspace
import torch.nn.functional as F
from torch.nn import Conv1d, Linear, Sequential, Module, ModuleList, LayerNorm, RMSNorm, ModuleDict

from torch.nn.utils.parametrize import register_parametrization

from einx import add, multiply, greater
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat, einsum

# ein notation

# b - batch
# h - heads
# n - sequence
# p - relative positions
# d - feature dimension

# constants

LinearNoBias = partial(Linear, bias = False)

Embeds = namedtuple('Embeds', [
    'embeds_1bp',
    'embeds_128bp',
    'embeds_pair',
])

# functions

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def last(arr):
    return arr[-1]

def is_odd(num):
    return not divisible_by(num, 2)

def is_even(num):
    return divisible_by(num, 2)

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value = 5.):
    return (t / value).tanh() * value

def symmetrize(t):
    return add('b i j ..., b j i ... -> b i j ...', t, t) * 0.5

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

# batch rmsnorm

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def get_maybe_dist_var(
    t,
    distributed = False
):
    t = rearrange(t, '... d -> (...) d')

    calc_distributed_var = distributed and is_distributed()

    if not calc_distributed_var:
        return t.var(dim = 0)

    device = t.device
    numel = tensor(t[..., 0].numel(), device = device)
    dist.all_reduce(numel)

    summed = stats.sum(dim = 0)
    dist.all_reduce(summed)

    mean = summed / numel
    centered = (t - mean)

    centered_squared_sum = centered.square().sum(dim = 0)
    dist.all_reduce(centered_squared_sum)

    return centered_squared_sum / (numel - 1) # with correction

class BatchRMSNorm(Module):
    def __init__(
        self,
        dim_feat,
        channel_first = False,
        distributed = True,
        momentum = 0.9,
        eps = 1e-5,
    ):
        super().__init__()
        self.scale = dim_feat ** 0.5

        self.eps = eps
        self.momentum = 1. - momentum
        self.channel_first = channel_first
        self.distributed = distributed

        self.gamma = nn.Parameter(torch.zeros(dim_feat))
        self.beta = nn.Parameter(torch.zeros(dim_feat))

        self.register_buffer('running_var', torch.ones((dim_feat,)))

    def forward(
        self,
        x,
        update_running_var = None
    ):
        gamma, beta, running_var, channel_first = self.gamma, self.beta, self.running_var, self.channel_first

        update_running_var = default(update_running_var, self.training)

        if update_running_var:

            to_reduce = rearrange(x, 'b d ... -> b ... d') if channel_first else x

            batch_var = get_maybe_dist_var(to_reduce, distributed = self.distributed)

            running_var.lerp_(batch_var, self.momentum)

        # get denominator

        std = running_var.clamp(min = self.eps).sqrt()

        # handle channel first

        if channel_first:
            std, gamma, beta = tuple(append_dims(t, x.ndim - 2) for t in (std, gamma, beta))

        # norm

        batch_rmsnormed = x / std * self.scale

        # scale and offset

        return batch_rmsnormed * (gamma + 1) + beta

# convolutional unet related

class WeightStandardConv(Conv1d):
    def __init__(
        self,
        dim,
        dim_out,
        width,
        *args,
        **kwargs
    ):
        super().__init__(dim, dim_out, width, *args, **kwargs)

        register_parametrization(self, 'weight', LayerNorm(self.weight.shape, elementwise_affine = False))

class ConvBlock(Module):
    def __init__(
        self,
        dim,
        width = 5,
        dim_out = None
    ):
        super().__init__()
        assert is_odd(width)
        dim_out = default(dim_out, dim)

        conv_klass = Conv1d if width == 1 else WeightStandardConv

        self.net = nn.Sequential(
            BatchRMSNorm(dim, channel_first = True),
            nn.GELU(),
            conv_klass(dim, dim_out, width, padding = width // 2)
        )

    def forward(self, x):
        return self.net(x)

class DownresBlock(Module):
    def __init__(
        self,
        dim,
        channels_to_add = 128 # this is new as well? instead of doubling channels, they add 128 at a time, and use padding or slicing for the residual
    ):
        super().__init__()

        dim_out = dim + channels_to_add
        self.pad = channels_to_add

        self.conv = ConvBlock(dim, width = 1, dim_out = dim_out)
        self.conv_out = ConvBlock(dim_out, width = 1)

        self.max_pool = Reduce('b d (n pool) -> b d n', 'max', pool = 2)

    def forward(self, x):

        residual = F.pad(x, (0, 0, 0, self.pad), value = 0.)

        out = self.conv(x) + residual

        out = self.conv_out(out) + out

        return self.max_pool(out)

class UpresBlock(Module):
    def __init__(
        self,
        dim,
        channels_to_remove = 128,
        residual_scale_init = .9,
        has_skip = True
    ):
        super().__init__()

        dim_out = dim - channels_to_remove
        self.pad = channels_to_remove

        self.conv = ConvBlock(dim, width = 1, dim_out = dim_out)
        self.unet_conv = ConvBlock(dim_out, width = 1) if has_skip else None

        self.conv_out = ConvBlock(dim_out, width = 1)

        self.residual_scale = nn.Parameter(torch.ones(1,) * residual_scale_init)

    def forward(
        self,
        x,
        skip = None
    ):
        length = x.shape[1]
        residual = x[:, :(length - self.pad)]

        out = self.conv(x) + residual

        out = repeat(out, 'b c n -> b c (n upsample)', upsample = 2) * self.residual_scale

        if exists(self.unet_conv):
            assert exists(skip)
            out = out + self.unet_conv(skip)

        return self.conv_out(out) + out

# position related

def relative_shift(t):
    *leading_dims, seq_len, dim = t.shape
    t = F.pad(t, (1, 0), value = 0.)
    t = t.reshape(*leading_dims, dim + 1, seq_len)
    t = t[..., 1:, :].reshape(*leading_dims, seq_len, dim)
    return t[..., :, :seq_len]

# rotary, but with attenuation of short relative distance frequencies

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        max_positions = 8192
    ):
        super().__init__()
        num_freqs = dim // 2
        inv_freq = 1. / (arange(num_freqs).float() + logspace(1, max_positions - num_freqs + 1, num_freqs))
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

# 'central mask features' - relative positions for constituting pairwise rep

class RelativePosFeatures(Module):
    def __init__(self, pool_size = 16):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, single):

        _, seq_len, dim = single.shape

        seq_len //= self.pool_size
        half_dim = dim // 2

        rel_pos = arange(2 * seq_len - 1) - (seq_len - 1)

        center_widths = (
            arange(half_dim) +
            logspace(1, seq_len - half_dim + 1, half_dim + 1)[:-1] # endpoint = False
        )

        abs_rel_pos, rel_pos_sign = rel_pos.abs(), rel_pos.sign()
        embeds = greater('j, i -> i j', center_widths, abs_rel_pos).float()

        return cat((embeds, multiply('i, i j', rel_pos_sign, embeds)), dim = -1)

# prenorm and sandwich norm - they use sandwich norm for single rep, prenorm for pairwise rep

class NormWrapper(Module):
    def __init__(
        self,
        dim,
        block: Module,
        dropout = 0.,
        sandwich = False,
        use_batch_rmsnorm = True
    ):
        super().__init__()
        norm_klass = BatchRMSNorm if use_batch_rmsnorm else RMSNorm

        self.block = block
        self.pre_rmsnorm = norm_klass(dim)

        self.post_block_dropout = nn.Dropout(dropout)
        self.post_rmsnorm = norm_klass(dim) if sandwich else nn.Identity()

    def forward(
        self,
        x,
        *args,
        **kwargs
    ):
        x = self.pre_rmsnorm(x)
        out = self.block(x, *args, **kwargs)
        out = self.post_block_dropout(out)
        return self.post_rmsnorm(out)

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        kv_heads = 1,
        dim_head_qk = 128,
        dim_head_v = 192,
        dim_pairwise = None,
        softclamp_value = 5., # they employ attention softclamping
        use_qk_rmsnorm = True
    ):
        super().__init__()
        dim_pairwise = default(dim_pairwise, dim)

        self.scale = dim_head ** -0.5

        qkv_proj_dim_out = (dim_head_qk * heads, dim_head_qk, dim_head_v)

        # splitting and merging of attention heads

        assert divisible_by(heads, kv_heads)
        groups = heads // kv_heads

        self.split_q_heads = Rearrange('b n (g h d) -> b g h n d', h = kv_heads, g = groups)
        self.split_kv_heads = Rearrange('b n (h d) -> b h n d', h = kv_heads)

        self.merge_heads = Rearrange('b g h n d -> b n (g h d)')

        # projections

        self.to_qkv = LinearNoBias(dim, sum(qkv_proj_dim_out))
        self.to_out = LinearNoBias(dim_head_v * heads, dim)

        # they add layernorms to queries, keys, and interestingly enough, values as well. first time i've seen this

        norm_klass = RMSNorm if use_qk_rmsnorm else partial(LayerNorm, bias = False)
        self.q_norm = norm_klass(dim_head_qk)
        self.k_norm = norm_klass(dim_head_qk)
        self.v_norm = norm_klass(dim_head_v)

        # to attention bias

        self.to_attn_bias = Sequential(
            BatchRMSNorm(dim_pairwise),
            nn.GELU(),
            LinearNoBias(dim_pairwise, heads),
            Rearrange('b i j (g h) -> b g h i j', g = groups)
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
        k, v = tuple(self.split_kv_heads(t) for t in (k, v))

        q, k, v = self.q_norm(q), self.k_norm(k), self.v_norm(v)

        q = q * self.scale

        # maybe rotary

        if exists(rotary_emb):
            q, k = tuple(apply_rotary_pos_emb(rotary_emb, t) for t in (q, k))

        # similarities

        sim = einsum(q, k, 'b g h i d, b h j d -> b g h i j')

        # add attention bias + softclamping

        if exists(pairwise):
            attn_bias = self.to_attn_bias(pairwise)

            assert divisible_by(sim.shape[-1], attn_bias.shape[-1])
            expand_factor = sim.shape[-1] // attn_bias.shape[-1]

            attn_bias = repeat(attn_bias, 'b g h i j -> b g h (i r1) (j r2)', r1 = expand_factor, r2 = expand_factor)

            sim = softclamp(sim + attn_bias, value = self.softclamp_value)

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        out = einsum(attn, v, 'b g h i j, b h j d -> b g h i d')

        out = self.merge_heads(out)
        return self.to_out(out)

# single to pairwise

class SingleToPairwise(Module):
    def __init__(
        self,
        dim,
        pool_size = 16,
        dim_pairwise = 128,
        heads = 32
    ):
        super().__init__()
        self.avg_pool = Reduce('b (n pool) d -> b n d', 'mean', pool = pool_size)

        dim_inner = heads * dim_pairwise

        self.split_heads = Rearrange('... (h d) -> ... h d', h = heads)

        self.to_outer_sum = Sequential(
            nn.GELU(),
            LinearNoBias(dim, dim_pairwise * 2),
        )

        self.to_qk = LinearNoBias(dim, dim_inner * 2)
        self.qk_to_pairwise = Linear(heads, dim_pairwise)

        # relative position related

        self.to_rel_pos_encoding = Linear(dim, heads * dim_pairwise)
        self.qk_rel_pos_bias = nn.Parameter(torch.zeros(2, 1, 1, heads, dim_pairwise))

    def forward(
        self,
        single,
        rel_pos_feats = None
    ):

        single = self.avg_pool(single)

        pool_seq_len = single.shape[1]

        q, k = self.to_qk(single).chunk(2, dim = -1)
        q, k = tuple(self.split_heads(t) for t in (q, k))

        sim = einsum(q, k, 'b i h d, b j h d -> b i j h')

        if exists(rel_pos_feats):
            rel_pos_encoding = self.to_rel_pos_encoding(rel_pos_feats)
            rel_pos_encoding = self.split_heads(rel_pos_encoding)

            q_rel_bias, k_rel_bias = self.qk_rel_pos_bias

            rel_q = relative_shift(einsum(q + q_rel_bias, rel_pos_encoding, 'b n h d, p h d -> b h n p'))
            rel_k = relative_shift(einsum(k + k_rel_bias, rel_pos_encoding, 'b n h d, p h d -> b h n p'))

            rel_sim = add('b h i j, b h j i -> b i j h', rel_q, rel_k) * 0.5

            sim = sim + rel_sim

        pairwise_from_sim = self.qk_to_pairwise(sim)

        outer_q, outer_k = self.to_outer_sum(single).chunk(2, dim = -1)

        outer_sum = add('b i d, b j d -> b i j d', outer_q, outer_k)

        return outer_sum

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

class TransformerTower(Module):
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
        dim_pairwise = 128,
        pairwise_every_num_single_blocks = 2,   # how often to do a pairwise block
        single_to_pairwise_heads = 32,          # they did 32
        pool_size = 16,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        dim_pairwise = default(dim_pairwise, dim)

        layers = []

        self.pairwise_every = pairwise_every_num_single_blocks

        self.rel_pos_features = RelativePosFeatures(pool_size)

        self.rotary_emb = RotaryEmbedding(dim_head_qk, max_positions = max_positions)

        for layer_index in range(depth):

            attn = Attention(dim = dim, dim_head_qk = dim_head_qk, dim_head_v = dim_head_v, heads = heads, dim_pairwise = dim_pairwise)

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

            attn = NormWrapper(dim = dim, block = attn, dropout = dropout, sandwich = True)
            ff = NormWrapper(dim = dim, block = ff, dropout = dropout, sandwich = True)

            # maybe pairwise

            single_to_pairwise, pairwise_attn, pairwise_ff = None, None, None

            if divisible_by(layer_index, self.pairwise_every):
                single_to_pairwise = SingleToPairwise(dim = dim, dim_pairwise = dim_pairwise, heads = single_to_pairwise_heads, pool_size = pool_size)
                pairwise_attn = PairwiseRowAttention(dim_pairwise)
                pairwise_ff = FeedForward(dim = dim_pairwise, expansion_factor = ff_expansion_factor)

                single_to_pairwise = NormWrapper(dim = dim, block = single_to_pairwise, dropout = dropout, use_batch_rmsnorm = False)
                pairwise_attn = NormWrapper(dim = dim_pairwise, block = pairwise_attn, dropout = dropout, use_batch_rmsnorm = False)
                pairwise_ff = NormWrapper(dim = dim_pairwise, block = pairwise_ff, dropout = dropout, use_batch_rmsnorm = False)

            # add to layers

            layers.append(ModuleList([
                attn,
                ff,
                single_to_pairwise,
                pairwise_attn,
                pairwise_ff
            ]))


        self.layers = ModuleList(layers)

    def forward(
        self,
        single
    ):

        seq_len = single.shape[1]

        pairwise = None

        rel_pos_feats = self.rel_pos_features(single)

        rotary_emb = self.rotary_emb(seq_len)

        for (
            attn,
            ff,
            maybe_single_to_pair,
            maybe_pairwise_attn,
            maybe_pairwise_ff
        ) in self.layers:

            if exists(maybe_single_to_pair):
                pairwise = maybe_single_to_pair(single, rel_pos_feats) + default(pairwise, 0.)
                pairwise = maybe_pairwise_attn(pairwise) + pairwise
                pairwise = maybe_pairwise_ff(pairwise) + pairwise

            single = attn(single, rotary_emb = rotary_emb, pairwise = pairwise) + single
            single = ff(single) + single

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
        self.conv = Conv1d(dim_input, dim, width, padding = width // 2)
        self.pointwise = Conv1d(dim, dim, 1)

        self.pool = Reduce('b d (n pool) -> b d n', 'max', pool = 2)

    def forward(
        self,
        seq # Int['b n']
    ):
        onehot = F.one_hot(seq, num_classes = self.dim_input).float()
        x = rearrange(onehot, 'b n d -> b d n')

        out = self.conv(x)
        out = out + self.pointwise(out)
        pooled = self.pool(out) # think they downsample for dna embed block

        return pooled, x

class OrganismEmbedding(Module):
    def __init__(
        self,
        dim,
        num_organisms
    ):
        super().__init__()
        self.embed = nn.Embedding(num_organisms, dim)

    def forward(
        self,
        x,
        organism_index
    ):
        organism_embed = self.embed(organism_index)
        return add('b n d, b d', x, organism_embed)

class OutputEmbedding(Module):
    def __init__(
        self,
        input_dim,
        num_organisms = 2,
        skip_dim = None
    ):
        super().__init__()
        self.double_features = Linear(input_dim, 2 * input_dim)
        self.skip_proj = LinearNoBias(skip_dim, 2 * input_dim) if exists(skip_dim) else None
        self.norm = BatchRMSNorm(2 * input_dim)
        self.embed = nn.Embedding(num_organisms, 2 * input_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        x,
        organism_index,
        skip_x = None
    ):
        seq_len = x.shape[1]
        x = self.double_features(x)  # double the input features

        if exists(skip_x) and exists(self.skip_proj):
            skip = self.skip_proj(skip_x)
            assert divisible_by(seq_len, skip.shape[1])

            repeat_factor = seq_len // skip.shape[1]
            skip = repeat(skip, 'b n d -> b (n repeat) d', repeat = repeat_factor)

            x = x + skip

        emb = self.embed(organism_index)

        x = add('b n d, b d', self.norm(x), emb)

        return self.activation(x)

class OutputPairEmbedding(Module):
    def __init__(
        self,
        pair_dim = 128,
        num_organisms = 2
    ):
        super().__init__()
        self.norm = RMSNorm(pair_dim)
        self.embed = nn.Embedding(num_organisms, pair_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        x,  # Float["b n n d"]
        organism_index
    ):
        x = symmetrize(x)

        organism_embed = self.embed(organism_index)
        embed = add('b i j d, b d', x, organism_embed)

        x = self.norm(x) + embed
        return self.activation(x)

# some reflection to make adding prediction heads easy

def get_function_arg_names(fn):
    signature = inspect.signature(fn)
    parameters = signature.parameters.values()
    return [p.name for p in parameters]

def is_disjoint(a: set, b: set):
    return not any(a.intersection(b))

# output heads

class SimpleOutputHead(Module):
    def __init__(
        self,
        input_dim,
        out_dim
    ):
        super().__init__()
        self.linear = Linear(input_dim, out_dim)
        self.scale = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.linear(x)
        return F.softplus(x) * F.softplus(self.scale)

# 1bp and 128bp share the same SimpleOutputHead but with different arg names to route the right embedding

class SimpleOutputHead1bp(SimpleOutputHead):
    def forward(self, embeds_1bp):
        return super().forward(embeds_1bp)

class SimpleOutputHead128bp(SimpleOutputHead):
    def forward(self, embeds_128bp):
        return super().forward(embeds_128bp)

class ContactMapHead(Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        num_organisms
    ):
        super().__init__()
        self.norm = RMSNorm(input_dim)
        self.embed = nn.Embedding(num_organisms, input_dim)
        self.gelu = nn.GELU()
        self.proj = Linear(input_dim, out_dim)
        
    def forward(
        self,
        embeds_pair,
        organism_index
    ):

        x = symmetrize(embeds_pair)
        x = self.norm(x)

        organism_embed = self.embed(organism_index)
        x = add('b i j d, b d', x, organism_embed)

        x = self.gelu(x)
        return self.proj(x)

class SpliceSiteClassifier(Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = Linear(input_dim, 5)  # Donor+, Acceptor+, Donor-, Acceptor-, None

    def forward(
        self,
        embeds_1bp
    ):
        return self.linear(embeds_1bp).softmax(dim = -1)

class SpliceSiteUsage(Module):
    def __init__(self, input_dim, n_contexts):
        super().__init__()
        self.linear = Linear(input_dim, n_contexts)

    def forward(
        self,
        embeds_1bp
    ):
        return self.linear(embeds_1bp).sigmoid()

class SpliceJunctionHead(Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_contexts
    ):
        super().__init__()
        self.project = Linear(input_dim, hidden_dim)
        self.scale = nn.Parameter(torch.ones(n_contexts, hidden_dim))
        self.offset = nn.Parameter(torch.zeros(n_contexts, hidden_dim))
        self.rope = RotaryEmbedding(hidden_dim)
        self.n_contexts = n_contexts

    def tissue_scaled_rope(
        self,
        embeds_1bp, # Float['b n h']
        indices # Int['b p']
    ):
    
        batch, device = embeds_1bp.shape[0], embeds_1bp.device
        n_contexts = self.n_contexts
    
        # index and rescale
        
        batch_indices = rearrange(arange(batch, device = device), 'b -> b 1')

        x = embeds_1bp[batch_indices, indices]  # [B, P, H]

        x = multiply('b p h, t h, t h -> b p t h', x, self.scale, self.offset)
    
        # get rotary embeddings [T, H]
        
        rotary_emb = self.rope(n_contexts)
    
        # apply
        
        x = apply_rotary_pos_emb(rotary_emb, x)

        return x

    def forward(
        self,
        embeds_1bp,
        splice_donor_idx,
        splice_acceptor_idx
    ):
        x_proj = self.project(embeds_1bp)
        donor_embed = self.tissue_scaled_rope(x_proj, splice_donor_idx)
        acceptor_embed = self.tissue_scaled_rope(x_proj, splice_acceptor_idx)
        scores = einsum(donor_embed, acceptor_embed, 'b d t h, b a t h -> b d a t')
        return F.softplus(scores)
        
# classes

class AlphaGenome(Module):
    def __init__(
        self,
        dims: tuple[int, ...] = (
            768,
            896,
            1024,
            1152,
            1280,
            1408,
            1536
        ),
        basepairs = 5,
        dna_embed_width = 15,
        num_organisms = 2,
        transformer_kwargs: dict = dict(),
    ):
        super().__init__()

        assert is_odd(dna_embed_width)

        assert len(dims) >= 2
        first_dim, *_, last_dim = dims

        self.dna_embed = DNAEmbed(first_dim, dim_input = basepairs, width = dna_embed_width)

        dim_with_input = (basepairs, *dims)
        dim_pairs = zip(dim_with_input[:-1], dim_with_input[1:])

        downs = []
        ups = []

        for layer_num, (dim_in, dim_out) in enumerate(dim_pairs, start = 1):
            is_first = layer_num == 1

            channel_diff = dim_out - dim_in

            assert channel_diff > 0

            if not is_first:
                down = DownresBlock(dim_in, channels_to_add = channel_diff)
                downs.append(down)

            up = UpresBlock(
                dim_out,
                channels_to_remove = channel_diff if not is_first else 0,
                has_skip = not is_first
            )

            ups.insert(0, up)

        self.downs = ModuleList(downs)
        self.ups = ModuleList(ups)

        self.transformer = TransformerTower(
            dim = last_dim,
            **transformer_kwargs
        )

        # organism embed and output embed

        self.organism_embed = OrganismEmbedding(last_dim, num_organisms)

        self.outembed_128bp = OutputEmbedding(last_dim, num_organisms)
        self.outembed_1bp = OutputEmbedding(first_dim, num_organisms, skip_dim=2*last_dim)
        self.outembed_pair = OutputPairEmbedding(2**len(dims), num_organisms)

        # heads

        self.num_organisms = num_organisms
        self.dim_1bp = last_dim
        self.dim_128bp = 2 * last_dim
        self.dim_contacts = 2 ** len(dims)

        self.heads = ModuleDict()
        self.head_forward_arg_names = dict()

    def add_splice_heads(
        self,
        organism,
        num_tracks_1bp,
        num_tracks_128bp,
        num_splicing_contexts,
        hidden_dim_splice_juncs = 512
    ):
        # head related

        organism_heads = ModuleDict({

            # RNA-seq, CAGE, ATAC, DNAse and PRO-Cap
            "1bp_tracks": SimpleOutputHead1bp(self.dim_1bp, num_tracks_1bp),
            
            # TF ChIP-seq and Histone ChIP-seq
            "128bp_tracks": SimpleOutputHead128bp(self.dim_128bp, num_tracks_128bp),

            # Contact Maps
            "contact_head": ContactMapHead(self.dim_contacts, self.dim_contacts, self.num_organisms),

            # Splicing
            "splice_probs": SpliceSiteClassifier(self.dim_1bp),
            "splice_usage": SpliceSiteUsage(self.dim_1bp, num_splicing_contexts),
            "splice_juncs": SpliceJunctionHead(self.dim_1bp, hidden_dim_splice_juncs, num_splicing_contexts)
        })

        self.heads[organism] = organism_heads
        self.head_forward_arg_names[organism] = {head_name: get_function_arg_names(head.forward) for head_name, head in organism_heads.items()}

    def get_embeds(
        self,
        seq, # Int['b n']
        organism_index
    ):

        skips = []

        # embed with one hot and add skip

        dna_embed, skip = self.dna_embed(seq)
        skips.append(skip)

        # downs

        x = dna_embed

        for down in self.downs:
            skips.append(x)
            x = down(x)

        x = rearrange(x, 'b d n -> b n d')
        
        # embed organism

        x = self.organism_embed(x, organism_index)

        # attention
        
        single, pairwise = self.transformer(x) # 1D 128bp resolution, 2D contact pairs
        
        # ups with skips from down
        
        x = rearrange(single, 'b n d -> b d n')

        for up in self.ups:
            x = up(x, skip = skips.pop())
            
        pred = rearrange(x, 'b d n -> b n d') # 1D 1bp resolution

        # embed organism to outputs

        embeds_128bp = self.outembed_128bp(single, organism_index)
        embeds_1bp = self.outembed_1bp(pred, organism_index, embeds_128bp)
        embeds_pair = self.outembed_pair(pairwise, organism_index)

        return Embeds(embeds_1bp, embeds_128bp, embeds_pair)
    
    def forward(
        self,
        seq,
        organism_index,
        return_embeds = False,
        **head_kwargs
    ):

        # process sequence
        
        embeds = self.get_embeds(seq, organism_index)

        # early return embeds, if specified

        if return_embeds or len(self.heads) == 0:
            return embeds

        # get output tracks

        embeds_1bp, embeds_128bp, embeds_pair = embeds

        head_inputs = dict(
            embeds_1bp = embeds_1bp,
            embeds_128bp = embeds_128bp,
            embeds_pair = embeds_pair,
            organism_index = organism_index,
        )

        assert is_disjoint(set(head_inputs.keys()), set(head_kwargs.keys()))

        head_inputs.update(**head_kwargs)

        out = dict()

        for organism, heads in self.heads.items():

            organism_head_args = self.head_forward_arg_names[organism]

            organism_out = dict()

            for head_name, head in heads.items():

                # get the inputs needed for the head, which is determined by the arg / kwarg name on the forward

                head_args = organism_head_args[head_name]

                head_kwargs = {head_arg: head_inputs[head_arg] for head_arg in head_args}

                # forward gathered inputs through the specific head

                head_out = head(**head_kwargs)

                organism_out[head_name] = head_out

            out[organism] = organism_out
        
        return out
