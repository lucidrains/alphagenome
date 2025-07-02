from __future__ import annotations

import inspect
from functools import partial
from collections import namedtuple, defaultdict

import torch
import torch.distributed as dist
from torch import tensor, is_tensor, arange, nn, cat, stack, arange, logspace, Tensor
import torch.nn.functional as F
from torch.nn import Conv1d, Linear, Sequential, Module, ModuleList, LayerNorm, ModuleDict

from torch.nn.utils.parametrize import register_parametrization

from einx import add, multiply, greater
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat, reduce, einsum

# ein notation

# b - batch
# h - heads
# n - sequence
# p - relative positions
# d - feature dimension
# t - tracks

# constants

LinearNoBias = partial(Linear, bias = False)

TransformerUnetOutput = namedtuple('TransformerUnetOutput', [
    'output',
    'single_repr',
    'pairwise_repr'
])

Embeds = namedtuple('Embeds', [
    'embeds_1bp',
    'embeds_128bp',
    'embeds_pair',
])

publication_heads_config = {
    'human': {
        'organism' : 'human',
        'num_tracks_1bp' : 3165, # 1174 total RNA-seq + 961 polyA plus RNA-seq + 516 hCAGE + 30 LQhCAGE + 167 ATAC-seq + 305 DNase-seq + 12 PRO-cap
        'num_tracks_128bp' : 2733, # 1617 TF ChIP-seq + 1116 Histone ChIP-seq
        'num_tracks_contacts' : 28, # 24 in situ Hi-C + 3 Micro-C + 1 Dilution Hi-C
        'num_splicing_contexts' : 282, # from Sup. Tab. 2
        'hidden_dim_splice_juncs' : 512 # max number of splice junctions to consider
    },
    'mouse': {
        'organism' : 'mouse',
        'num_tracks_1bp' : 730, # 168 total RNA-seq + 365 polyA plus RNA-seq + 172 hCAGE + 16 LQhCAGE + 18 ATAC-seq + 67 DNase-seq
        'num_tracks_128bp' : 310, # 127 TF ChIP-seq + 183 Histone ChIP-seq
        'num_tracks_contacts' : 8, # 8 in situ Hi-C
        'num_splicing_contexts' : 75, # from Sup. Tab. 2
        'hidden_dim_splice_juncs' : 512 # max number of splice junctions to consider
    }
}

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

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def softclamp(t, value = 5.):
    return (t / value).tanh() * value

def symmetrize(t):
    return add('b i j ..., b j i ... -> b i j ...', t, t) * 0.5

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def log(t, eps = 1e-7):
    return t.clamp(min = eps).log()

def safe_div(num, den, eps = 1e-7):
    return num / den.clamp(min = eps)

# losses

class MultinomialLoss(Module):
    def __init__(
        self,
        multinomial_resolution,
        positional_loss_weight = 5.,
        eps = 1e-7
    ):
        super().__init__()
        self.split_res = Rearrange('... (n resolution) t -> ... n resolution t', resolution = multinomial_resolution)

        self.eps = eps
        self.log = partial(log, eps = eps)
        self.resolution = multinomial_resolution
        self.positional_loss_weight = positional_loss_weight

    def forward(self, pred, target):
        pred = self.split_res(pred)
        target = self.split_res(target)

        pred_sum = reduce(pred, '... n resolution t -> ... n 1 t', 'sum')
        target_sum = reduce(target, '... n resolution t -> ... n 1 t', 'sum')

        poisson_loss = (pred_sum - target_sum * self.log(pred_sum)).sum()
        multinomial_prob = pred / pred_sum.clamp(min = self.eps)

        positional_loss = (-target * self.log(multinomial_prob)).sum()

        return poisson_loss / self.resolution + positional_loss * self.positional_loss_weight


class SoftClip(Module):
    def __init__(
        self, 
        threshold = 10.0
    ):
        super().__init__()
        self.threshold = threshold

    def forward(
        self, 
        x: torch.Tensor
    ):
        x = torch.where(
            x > self.threshold,
            2 * torch.sqrt(x * self.threshold) - self.threshold,
            x
        )
        return x

class MultinomialCrossEntropy(Module):
    def __init__(
        self,
        dim = -1,
        eps = 1e-7
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, pred, target):
        pred_ratios = safe_div(pred, pred.sum(dim = self.dim, keepdim = True), eps = self.eps)
        target_ratios = safe_div(target, target.sum(dim = self.dim, keepdim = True), eps = self.eps)
        return -(target_ratios * log(pred_ratios, eps = self.eps)).sum()

class PoissonLoss(Module):
    def __init__(
        self,
        dim = -1,
        eps = 1e-7,
        softclip_threshold = 10.
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.softclip = SoftClip(softclip_threshold)

    def forward(self, pred, target):
        sum_pred = pred.sum(dim = self.dim)
        sum_target = self.softclip(target.sum(dim = self.dim))
        return (sum_pred - sum_target * log(sum_pred, eps = self.eps)).sum()

class JunctionsLoss(Module):
    def __init__(
        self, 
        softclip_threshold = 10.
    ):
        super().__init__()
        self.multinomial_cross_entropy_row = MultinomialCrossEntropy(dim = 0)
        self.multinomial_cross_entropy_col = MultinomialCrossEntropy(dim = 1)

        self.poisson_loss_row = PoissonLoss(dim = 0, softclip_threshold = softclip_threshold)
        self.poisson_loss_col = PoissonLoss(dim = 1, softclip_threshold = softclip_threshold)

    def forward(
        self, 
        pred: Tensor, 
        target: Tensor
    ):
        loss = 0.0

        for pred_c, target_c in zip(pred.unbind(dim = -1), target.unbind(dim = -1)):

            ratios = self.multinomial_cross_entropy_row(pred_c, target_c) + \
                     self.multinomial_cross_entropy_col(pred_c, target_c)

            counts = self.poisson_loss_row(pred_c, target_c) + \
                     self.poisson_loss_col(pred_c, target_c)

            loss = loss + ( 0.2 * ratios + 0.04 * counts )

        return loss

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
        update_running_var = True
    ):
        super().__init__()
        self.scale = dim_feat ** 0.5

        self.eps = eps
        self.momentum = 1. - momentum
        self.channel_first = channel_first
        self.distributed = distributed

        self.gamma = nn.Parameter(torch.zeros(dim_feat))
        self.beta = nn.Parameter(torch.zeros(dim_feat))

        self.update_running_var = update_running_var
        self.register_buffer('running_var', torch.ones((dim_feat,)))

    def forward(
        self,
        x,
        update_running_var = None
    ):
        gamma, beta, running_var, channel_first = self.gamma, self.beta, self.running_var, self.channel_first

        update_running_var = default(update_running_var, self.update_running_var, self.training)

        if update_running_var:

            with torch.no_grad():
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

# for easier time freezing the running variance

def set_update_running_var(
    root: Module,
    update_running_var
):
    for mod in root.modules():
        if isinstance(mod, BatchRMSNorm):
            mod.update_running_var = update_running_var

# channel first rmsnorm

class ChannelFirstRMSNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = append_dims(self.gamma, x.ndim - 2)
        return l2norm(x, dim = 1) * self.scale * (gamma + 1)

# function for determining which rmsnorm

def RMSNorm(dim, channel_first = False, batch = False):

    if not batch and not channel_first:
        return nn.RMSNorm(dim)
    elif not batch and channel_first:
        return ChannelFirstRMSNorm(dim)
    else:
        return BatchRMSNorm(dim, channel_first = channel_first)

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
        dim_out = None,
        batch_rmsnorm = True
    ):
        super().__init__()
        assert is_odd(width)
        dim_out = default(dim_out, dim)

        conv_klass = Conv1d if width == 1 else WeightStandardConv

        self.net = nn.Sequential(
            RMSNorm(dim, channel_first = True, batch = batch_rmsnorm),
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
        norm_klass = partial(RMSNorm, batch = use_batch_rmsnorm)

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
        use_qk_rmsnorm = True,
        attn_bias_use_batch_rmsnorm = True
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

        norm_klass = nn.RMSNorm if use_qk_rmsnorm else partial(LayerNorm, bias = False)
        self.q_norm = norm_klass(dim_head_qk)
        self.k_norm = norm_klass(dim_head_qk)
        self.v_norm = norm_klass(dim_head_v)

        # to attention bias

        self.to_attn_bias = Sequential(
            RMSNorm(dim_pairwise, batch = attn_bias_use_batch_rmsnorm),
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

        # accessible attributes
        self.dim_pairwise = dim_pairwise

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

# transformer unet

class TransformerUnet(Module):
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
        transformer_kwargs: dict = dict()
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

    def forward(
        self,
        seq,
        pre_attend_embed = None
    ):
        skips = []

        # embed with one hot and add skip

        dna_embed, skip = self.dna_embed(seq)

        # downs

        x = dna_embed

        for down in self.downs:
            skips.append(x)
            x = down(x)

        x = rearrange(x, 'b d n -> b n d')

        # embed organism

        if exists(pre_attend_embed):
            x = add('b n d, b d', x, pre_attend_embed)

        # attention
        
        single, pairwise = self.transformer(x) # 1D 128bp resolution, 2D contact pairs
        
        # ups with skips from down
        
        x = rearrange(single, 'b n d -> b d n')

        for i, up in enumerate(self.ups):
            is_last = i == (len(self.ups) - 1)
            skip = skips.pop() if not is_last else None

            x = up(x, skip = skip)
            
        out = rearrange(x, 'b d n -> b n d') # 1D 1bp resolution

        return TransformerUnetOutput(out, single, pairwise) # the final output, as well as single and pairwise repr from the transformer

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
        organism_index
    ):
        return self.embed(organism_index)

class OutputEmbedding(Module):
    def __init__(
        self,
        input_dim,
        num_organisms = 2,
        skip_dim = None,
        use_batch_rmsnorm = True
    ):
        super().__init__()
        self.double_features = Linear(input_dim, 2 * input_dim)
        self.skip_proj = LinearNoBias(skip_dim, 2 * input_dim) if exists(skip_dim) else None
        self.norm = RMSNorm(2 * input_dim, batch = use_batch_rmsnorm)
        self.embed = nn.Embedding(num_organisms, 2 * input_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        x,
        organism_index,
        skip = None
    ):
        seq_len = x.shape[1]
        x = self.double_features(x)  # double the input features

        if exists(skip) and exists(self.skip_proj):
            skip = self.skip_proj(skip)
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
        x,  # Float['b n n d']
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

class ContactMapHead(Module):
    def __init__(
        self,
        input_dim,
        num_tracks,
        num_organisms
    ):
        super().__init__()
        self.norm = RMSNorm(input_dim)
        self.embed = nn.Embedding(num_organisms, input_dim)
        self.gelu = nn.GELU()
        self.proj = Linear(input_dim, num_tracks)
        
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
        n_contexts,
        rope_max_position = 2 ** 20
    ):
        super().__init__()
        self.project = Linear(input_dim, hidden_dim)
        self.scale = nn.Parameter(torch.ones(n_contexts, hidden_dim))
        self.offset = nn.Parameter(torch.zeros(n_contexts, hidden_dim))
        self.rope = RotaryEmbedding(hidden_dim, max_positions = rope_max_position)
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

class TracksScaledPrediction(Module):
    def __init__(
        self,
        dim,
        num_tracks
    ):
        super().__init__()
        self.to_pred = Linear(dim, num_tracks)
        self.scale = nn.Parameter(torch.zeros(num_tracks))

    def forward(
        self,
        x
    ):
        track_pred = self.to_pred(x)
        return F.softplus(track_pred) * F.softplus(self.scale)

class SoftClip(Module):
    def __init__(
        self,
        scale = 2.,
        gamma = 10.,
        threshold = 10.
    ):
        super().__init__()
        self.scale = scale
        self.gamma = gamma
        self.threshold = threshold

    def inverse(self, clipped):
        threshold, scale, gamma = self.threshold, self.scale, self.gamma
        return torch.where(clipped > threshold, ((clipped + threshold) ** 2) / (gamma * scale ** 2), clipped)

    def forward(self, x):
        threshold, scale, gamma = self.threshold, self.scale, self.gamma
        return torch.where(x > threshold, scale * torch.sqrt(x * gamma).sqrt() - threshold, x)

class TargetScaler(Module):
    def __init__(
        self,
        track_means: list[float] | Tensor,
        apply_squashing = False, # for rna-seq they squash the signal. not intimate enough with these assays to understand why yet
        squashing_factor = 0.75,
        softclip_scale = 2.,
        softclip_gamma = 10.,
        softclip_threshold = 10.
    ):
        super().__init__()

        if not is_tensor(track_means):
            track_means = tensor(track_means).float()

        self.register_buffer('track_means', track_means)

        self.apply_squashing = apply_squashing
        self.squashing_factor = squashing_factor

        self.softclip = SoftClip(threshold = softclip_threshold, scale = softclip_scale, gamma = softclip_gamma)

    def inverse(self, normalized):
        x = self.softclip.inverse(normalized)

        if self.apply_squashing:
            x = x ** (1. / self.squashing_factor)

        return x * self.track_means

    def forward(self, tracks):
        tracks = tracks / self.track_means

        if self.apply_squashing:
            tracks = tracks ** self.squashing_factor

        return self.softclip(tracks)

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

        self.transformer_unet = TransformerUnet(
            dims = dims,
            basepairs = basepairs,
            dna_embed_width = dna_embed_width,
            transformer_kwargs = transformer_kwargs
        )

        # organism embed and output embed

        first_dim, *_, last_dim = dims

        self.organism_embed = OrganismEmbedding(last_dim, num_organisms)

        self.outembed_128bp = OutputEmbedding(last_dim, num_organisms)
        self.outembed_1bp = OutputEmbedding(first_dim, num_organisms, skip_dim = 2*last_dim)
        self.outembed_pair = OutputPairEmbedding(self.transformer_unet.transformer.dim_pairwise, num_organisms)

        # heads

        self.num_organisms = num_organisms
        self.dim_1bp = last_dim
        self.dim_128bp = 2 * last_dim
        self.dim_contacts = self.transformer_unet.transformer.dim_pairwise

        self.heads = ModuleDict()
        self.head_forward_arg_names = defaultdict(dict)
        self.head_forward_arg_maps = defaultdict(dict) # contains a map of forward argument names to the embed name to be used

    @property
    def total_parameters(self):
        return sum([p.numel() for p in self.parameters()])

    def add_head(
        self,
        organism,
        head_name,
        head: Module,
        head_input_kwarg_names: str | tuple[str, ...] | None = None
    ):
        if isinstance(head_input_kwarg_names, str):
            head_input_kwarg_names = (head_input_kwarg_names,)

        if organism not in self.heads:
            self.heads[organism] = ModuleDict()

        self.heads[organism][head_name] = head

        # store the list of inputs to the head

        head_forward_arg_names = get_function_arg_names(head.forward)

        self.head_forward_arg_names[organism][head_name] = default(head_input_kwarg_names, head_forward_arg_names)

        # create a dict[str, str] for if the custom head contains inputs that are named explicitly through `head_input_kwarg_names` in the order presented in function

        head_forwared_arg_map = dict()

        if exists(head_input_kwarg_names):
            head_forwared_arg_map = dict(zip(head_input_kwarg_names, head_forward_arg_names))

        self.head_forward_arg_maps[organism][head_name] = head_forwared_arg_map

    def add_heads(
        self,
        organism,
        num_tracks_1bp,
        num_tracks_128bp,
        num_tracks_contacts,
        num_splicing_contexts,
        hidden_dim_splice_juncs = 512
    ):
        # splice head related

        organism_heads = (

            # RNA-seq, CAGE, ATAC, DNAse and PRO-Cap
            ('1bp_tracks', TracksScaledPrediction(self.dim_1bp, num_tracks_1bp), ('embeds_1bp',)),
            
            # TF ChIP-seq and Histone ChIP-seq
            ('128bp_tracks', TracksScaledPrediction(self.dim_128bp, num_tracks_128bp), ('embeds_128bp',)),

            # Contact Maps
            ('contact_head', ContactMapHead(self.dim_contacts, num_tracks_contacts, self.num_organisms)),

            # Splicing
            ('splice_probs', SpliceSiteClassifier(self.dim_1bp)),
            ('splice_usage', SpliceSiteUsage(self.dim_1bp, num_splicing_contexts)),
            ('splice_juncs', SpliceJunctionHead(self.dim_1bp, hidden_dim_splice_juncs, num_splicing_contexts))
        )

        for add_head_args in organism_heads:
            self.add_head(organism, *add_head_args)

    def get_embeds(
        self,
        seq, # Int['b n']
        organism_index
    ):

        organism_embed = self.organism_embed(organism_index)

        unet_out, single, pairwise = self.transformer_unet(seq, pre_attend_embed = organism_embed)

        # embed organism to outputs

        embeds_128bp = self.outembed_128bp(single, organism_index)
        embeds_1bp = self.outembed_1bp(unet_out, organism_index, embeds_128bp)
        embeds_pair = self.outembed_pair(pairwise, organism_index)

        return Embeds(embeds_1bp, embeds_128bp, embeds_pair)
    
    def forward(
        self,
        seq,
        organism_index: int | Tensor,
        return_embeds = False,
        **head_kwargs
    ):

        # handle int organism_index (0 for human, 1 for mouse)

        if isinstance(organism_index, int):
            batch = seq.shape[0]
            organism_index = torch.full((batch,), organism_index, device = seq.device)

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

            organism_head_arg_map = self.head_forward_arg_maps[organism]

            organism_out = dict()

            for head_name, head in heads.items():

                # get the inputs needed for the head, which is determined by the arg / kwarg name on the forward

                head_args = organism_head_args[head_name]
                head_arg_map = organism_head_arg_map[head_name]

                head_kwargs = {head_arg: head_inputs[head_arg] for head_arg in head_args}

                # remap the kwargs

                head_kwargs = {(head_arg_map.get(k, k)): v for k, v in head_kwargs.items()}

                # forward gathered inputs through the specific head

                head_out = head(**head_kwargs)

                organism_out[head_name] = head_out

            out[organism] = organism_out
        
        return out
