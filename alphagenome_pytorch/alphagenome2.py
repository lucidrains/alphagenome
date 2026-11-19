from __future__ import annotations

from inspect import signature
from itertools import zip_longest
from collections import defaultdict
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, is_tensor, nn, tensor

import einx
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PoPE_pytorch import PoPE, compute_attn_similarity

from torch.nn import Conv1d, Linear, Module, ModuleDict, ModuleList
from torch_einops_utils import (
    Sequential,
    batched_index_select,
    clamp,
    masked_mean,
    pack_with_inverse,
    pad_right_at_dim,
    pad_right_ndim_to,
    slice_left_at_dim,
    tree_map_tensor,
)
from torch_einops_utils.shape import is_shape, size

from alphagenome_pytorch.alphagenome import Embeds, TransformerUnetOutput

# ein notation

# b - batch
# h - heads
# n - sequence
# p - pairwise positions
# d - feature dimension
# c - channels / tracks
# s - strands
# g - genes
# l - depth / stage history

# constants

LinearNoBias = partial(Linear, bias = False)

DNA_CHAR_MAP = dict(A = 0, C = 1, G = 2, T = 3)

# 14 shared heads in stages 1-3, +1 denoised scRNA head in stage 4

ag2_genome_track_heads = dict(
    rna_seq = (12, (1, 128)),
    cage = (4, (1, 128)),
    dnase = (2, (1, 128)),
    atac = (2, (1, 128)),
    sc_atac = (5, (1, 128)),
    chip_tf = (911, (128,)),
    chip_histone = (43, (128,)),
)

ag2_contact_map_channels = 1
ag2_methylation_channels = 2
ag2_splice_usage_channels = 6
ag2_splice_junction_hidden_dim = 256
ag2_splice_junction_num_assays = 3
ag2_bulk_gene_expr_channels = 12
ag2_sc_gene_expr_channels = 14
ag2_denoised_sc_gene_expr_channels = 2

ag2_assay_scales = dict(
    splice_junctions = 2.,
    bulk_gene_expression = 42.,
    sc_gene_expression = 48.,
    denoised_sc_gene_expression = 48.,
)

# helpers

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(num):
    return not divisible_by(num, 2)

def softclamp(t, value = 10.):
    return (t / value).tanh() * value

def symmetrize(t):
    return einx.add('b i j ..., b j i ... -> b i j ...', t, t) * 0.5

def log(t, eps = 1e-7):
    return clamp(t, lo = eps).log()

def safe_div(num, den, eps = 1e-7):
    return num / clamp(den, lo = eps)

def gelu_1702(x):
    return torch.sigmoid(1.702 * x) * x

def geomspace(start, stop, num, device = None):
    if num == 1:
        return tensor([start], device = device)

    return start * (stop / start) ** torch.linspace(0, 1, num, device = device)

def poisson_nll(pred, target, eps = 1e-7):
    return pred - target * log(pred, eps = eps)

def parse_dna_seq(seq, device = None):
    if isinstance(seq, str):
        seq = [seq]

    if isinstance(seq, (list, tuple)) and len(seq) > 0 and isinstance(seq[0], str):
        seq = tensor([[DNA_CHAR_MAP.get(ch, -1) for ch in s.upper()] for s in seq], dtype = torch.long, device = device)

    assert is_tensor(seq), 'sequence inputs must be DNA strings or torch.Tensors'

    is_int_onehot = (
        not seq.dtype.is_floating_point
        and (is_shape(seq, 'b n 4') or (is_shape(seq, 'n 4') and size(seq, '[n] 4') > 4))
        and (seq.dtype == torch.bool or (seq.min().item() >= 0 and seq.max().item() <= 1))
    )
    if is_int_onehot:
        seq = seq.argmax(dim = -1)

    if (is_shape(seq, 'n') and not seq.dtype.is_floating_point) or (is_shape(seq, 'n d') and seq.dtype.is_floating_point):
        seq = rearrange(seq, '... -> 1 ...')

    return seq

# target scaling and losses

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
        return torch.where(x > threshold, scale * torch.sqrt(x * gamma) - threshold, x)

class TargetScaler(Module):
    def __init__(
        self,
        scale = 1.,
        bin_width = 1.,
        apply_squashing = False,
        squashing_factor = 0.75,
        log1p_scale = False,
        softclip_scale = 2.,
        softclip_gamma = 10.,
        softclip_threshold = 10.
    ):
        super().__init__()
        if not is_tensor(scale):
            scale = tensor(scale).float()

        self.register_buffer('scale', scale)
        self.bin_width = bin_width
        self.apply_squashing = apply_squashing
        self.squashing_factor = squashing_factor
        self.log1p_scale = log1p_scale
        self.softclip = SoftClip(threshold = softclip_threshold, scale = softclip_scale, gamma = softclip_gamma)

    def inverse(self, normalized):
        denom = self.scale * self.bin_width

        if self.log1p_scale:
            return torch.expm1(normalized) * denom

        x = self.softclip.inverse(normalized)

        if self.apply_squashing:
            x = x ** (1. / self.squashing_factor)

        return x * denom

    def forward(self, tracks):
        tracks = tracks / (self.scale * self.bin_width)

        if self.log1p_scale:
            return torch.log1p(tracks)

        if self.apply_squashing:
            tracks = tracks ** self.squashing_factor

        return self.softclip(tracks)

class MultinomialLoss(Module):
    def __init__(
        self,
        multinomial_resolution = 2 ** 17,
        positional_loss_weight = 5.,
        poisson_loss_weight = 1.,
        eps = 1e-7
    ):
        super().__init__()
        self.resolution = multinomial_resolution
        self.positional_loss_weight = positional_loss_weight
        self.poisson_loss_weight = poisson_loss_weight
        self.eps = eps

    def forward(self, pred, target, mask = None):
        if exists(mask):
            pred, target = pred * mask, target * mask

        pred = rearrange(pred, '... (n resolution) c -> ... n resolution c', resolution = self.resolution)
        target = rearrange(target, '... (n resolution) c -> ... n resolution c', resolution = self.resolution)

        pred_sum = reduce(pred, '... n resolution c -> ... n 1 c', 'sum')
        target_sum = reduce(target, '... n resolution c -> ... n 1 c', 'sum')

        count_loss = poisson_nll(pred_sum, target_sum, eps = self.eps).sum()
        multinomial_prob = safe_div(pred, pred_sum, eps = self.eps)
        positional_loss = (-target * log(multinomial_prob, eps = self.eps)).sum()

        return (count_loss / self.resolution) * self.poisson_loss_weight + positional_loss * self.positional_loss_weight

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
        dim = None,
        eps = 1e-7
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, pred, target, mask = None):
        if exists(mask):
            pred, target = pred * mask, target * mask

        if exists(self.dim):
            pred, target = pred.sum(dim = self.dim), target.sum(dim = self.dim)

        return poisson_nll(pred, target, eps = self.eps).sum()

class JunctionsLoss(Module):
    def __init__(
        self,
        ratio_weight = 0.5,
        count_weight = 0.1,
        scale_targets = False,
        target_scale = 2.,
        squashing_factor = 0.75,
        eps = 1e-7
    ):
        super().__init__()
        self.ratio_weight = ratio_weight
        self.count_weight = count_weight
        self.target_scaler = TargetScaler(
            scale = target_scale,
            apply_squashing = True,
            squashing_factor = squashing_factor
        ) if scale_targets else None

        # trailing layout: (..., donors, acceptors, channels)

        self.donor_ce = MultinomialCrossEntropy(dim = -3, eps = eps)
        self.acceptor_ce = MultinomialCrossEntropy(dim = -2, eps = eps)
        self.donor_poisson = PoissonLoss(dim = -3, eps = eps)
        self.acceptor_poisson = PoissonLoss(dim = -2, eps = eps)

    def forward(self, pred, target, mask = None):
        if exists(self.target_scaler):
            target = self.target_scaler(target)

        if exists(mask):
            pred, target = pred * mask, target * mask

        ratios = self.donor_ce(pred, target) + self.acceptor_ce(pred, target)
        counts = self.donor_poisson(pred, target) + self.acceptor_poisson(pred, target)

        return self.ratio_weight * ratios + self.count_weight * counts

class BinomialLoss(Module):
    def __init__(self, eps = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_prob, methylated_counts, total_counts, mask = None):
        unmethylated_counts = total_counts - methylated_counts
        nll = -(methylated_counts * log(pred_prob, eps = self.eps) + unmethylated_counts * log(1. - pred_prob, eps = self.eps))

        if exists(mask):
            nll = torch.where(mask.bool(), nll, 0.)

        return masked_mean(nll, mask = mask, eps = self.eps)

# normalization and conditioning

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-5,
        elementwise_affine = True
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.scale = nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        self.offset = nn.Parameter(torch.zeros(dim)) if elementwise_affine else None

    def forward(self, x):
        normed = x * torch.rsqrt(x.float().square().mean(dim = -1, keepdim = True) + self.eps).type_as(x)

        if not self.elementwise_affine:
            return normed

        return normed * self.scale.type_as(x) + self.offset.type_as(x)

class ConditioningEmbedder(Module):
    def __init__(
        self,
        dim = 256,
        num_taxonomies = 2,
        num_conditions = 2048,
        dim_cell_embed = 256,
        dim_adapter_hidden = 8192
    ):
        super().__init__()
        self.taxonomy_emb = nn.Embedding(num_taxonomies, dim)
        self.condition_emb = nn.Embedding(num_conditions, dim)

        self.cell_adapter = Sequential(
            nn.LayerNorm(dim_cell_embed),
            Linear(dim_cell_embed, dim_adapter_hidden),
            nn.GELU(),
            Linear(dim_adapter_hidden, dim)
        )

        self.norm = nn.LayerNorm(dim, elementwise_affine = False)

    @torch.no_grad()
    def clamp_condition_emb_rms_(self, max_rms = 1.):
        rms = self.condition_emb.weight.square().mean(dim = -1, keepdim = True).sqrt()
        self.condition_emb.weight.div_(clamp(rms / max_rms, lo = 1.))

    def forward(
        self,
        taxonomy_index,
        condition_index = None,
        cell_embed = None
    ):
        assert not (exists(condition_index) and exists(cell_embed)), 'pass either `condition_index` or `cell_embed`, not both'

        cond = self.taxonomy_emb(taxonomy_index)

        if exists(cell_embed):
            if is_shape(cell_embed, 'b k d'):
                cell_embed = reduce(cell_embed, 'b k d -> b d', 'mean')

            cond = cond + self.cell_adapter(cell_embed)

        elif exists(condition_index):
            cond_emb = self.condition_emb(clamp(condition_index, lo = 0))
            cond = cond + einx.where('b, b d, -> b d', condition_index >= 0, cond_emb, 0.)

        return self.norm(cond)

class FiLM(Module):
    def __init__(
        self,
        dim_cond,
        dim
    ):
        super().__init__()
        self.input_scale = nn.Parameter(torch.empty(dim_cond))
        self.input_offset = nn.Parameter(torch.zeros(dim_cond))
        self.to_scale = Linear(dim_cond, dim)
        self.to_offset = Linear(dim_cond, dim)

        nn.init.trunc_normal_(self.input_scale)
        nn.init.trunc_normal_(self.to_scale.weight, std = 1e-8)
        nn.init.trunc_normal_(self.to_offset.weight, std = 1e-8)
        nn.init.ones_(self.to_scale.bias)
        nn.init.zeros_(self.to_offset.bias)

    def forward(self, x, cond = None):
        if not exists(cond):
            return x

        cond = F.gelu(cond * self.input_scale + self.input_offset)
        scale, offset = self.to_scale(cond).tanh(), self.to_offset(cond)

        return einx.add('b ... d, b d -> b ... d', einx.multiply('b ... d, b d -> b ... d', x, scale), offset)

class FiLMNorm(Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        layer_norm = False
    ):
        super().__init__()
        has_cond = exists(cond_dim)
        norm_klass = nn.LayerNorm if layer_norm else RMSNorm

        self.norm = norm_klass(dim, elementwise_affine = not has_cond)
        self.film = FiLM(cond_dim, dim) if has_cond else None

    def forward(self, x, cond = None):
        x = self.norm(x)
        return self.film(x, cond) if exists(self.film) else x

# single-cell expression autoencoder (Stage 4 expression conditioning)

def AutoencoderMLPBlock(dim_in, dim_hidden = 4096, dim_out = 2048):
    return Sequential(
        nn.LayerNorm(dim_in),
        Linear(dim_in, dim_hidden),
        nn.GELU(),
        Linear(dim_hidden, dim_out)
    )

class SingleCellAutoencoder(Module):
    def __init__(
        self,
        num_genes = 39_323,
        dim = 2048,
        dim_hidden = 4096,
        dim_latent = 256,
        enc_depth = 2,
        dropout = 0.1,
        eps = 1e-7
    ):
        super().__init__()
        self.eps = eps
        self.input_dropout = nn.Dropout(dropout)
        self.to_hidden = Linear(num_genes, dim)

        self.enc_blocks = ModuleList([AutoencoderMLPBlock(dim, dim_hidden, dim) for _ in range(enc_depth)])
        self.to_latent = Sequential(
            nn.LayerNorm(dim),
            Linear(dim, dim_latent)
        )

        self.dec_block = Sequential(
            AutoencoderMLPBlock(dim_latent, dim_hidden, dim),
            nn.LayerNorm(dim)
        )
        self.to_gene_logits = Linear(dim, num_genes)
        self.theta_logit = nn.Parameter(torch.zeros(num_genes))

    def encode(self, counts, gene_mask = None):
        library_size = counts.sum(dim = -1, keepdim = True)
        x_norm = self.input_dropout(torch.log1p(1e4 * safe_div(counts, library_size, eps = self.eps)))

        if exists(gene_mask):
            x_norm = x_norm * gene_mask

        h = self.to_hidden(x_norm)

        for block in self.enc_blocks:
            h = block(h) + h

        return self.to_latent(h)

    def decode(self, z, library_size = 1e6):
        h_dec = self.dec_block(z)
        return self.to_gene_logits(h_dec).softmax(dim = -1) * library_size

    def nb_nll(self, counts, mean, theta):
        eps = self.eps
        log_theta_mu = log(theta + mean, eps = eps)
        return (
            torch.lgamma(theta)
            + torch.lgamma(counts + 1.)
            - torch.lgamma(counts + theta)
            + theta * (log_theta_mu - log(theta, eps = eps))
            + counts * (log_theta_mu - log(mean, eps = eps))
        )

    def forward(self, counts, gene_mask = None):
        z = self.encode(counts, gene_mask = gene_mask)
        h_dec = self.dec_block(z)
        theta = self.theta_logit.exp()
        library_size = counts.sum(dim = -1, keepdim = True)

        if not exists(gene_mask):
            mean = self.to_gene_logits(h_dec).softmax(dim = -1) * library_size
            return self.nb_nll(counts, mean, theta).sum(dim = -1).mean()

        # held-out gene masking; stop-grad fit on held-out rows

        gene_mask_bool = gene_mask.bool()
        neg_inf = -torch.finfo(h_dec.dtype).max

        logits_train = self.to_gene_logits(h_dec).masked_fill(~gene_mask_bool, neg_inf)
        mean_train = logits_train.softmax(dim = -1) * (counts * gene_mask).sum(dim = -1, keepdim = True)
        loss_train = (self.nb_nll(counts, mean_train, theta) * gene_mask).sum(dim = -1)

        mean_all = self.to_gene_logits(h_dec.detach()).softmax(dim = -1) * library_size
        loss_all = self.nb_nll(counts, mean_all, theta).sum(dim = -1)

        return (loss_train + loss_all).mean()

# convolutional unet

class ConvBlock(Module):
    def __init__(
        self,
        dim,
        width = 5,
        dim_out = None,
        cond_dim = None
    ):
        super().__init__()
        assert is_odd(width)
        dim_out = default(dim_out, dim)

        self.norm = FiLMNorm(dim, cond_dim)
        self.conv = Conv1d(dim, dim_out, width, padding = width // 2)

    def forward(self, x, cond = None):
        x = rearrange(x, 'b c n -> b n c')
        x = gelu_1702(self.norm(x, cond))
        return self.conv(rearrange(x, 'b n c -> b c n'))

class DNAEmbed(Module):
    def __init__(
        self,
        dim,
        dim_input = 4,
        width = 15,
        cond_dim = None
    ):
        super().__init__()
        assert is_odd(width)
        self.dim_input = dim_input
        self.conv = Conv1d(dim_input, dim, width, padding = width // 2)
        self.conv_block = ConvBlock(dim, width = 5, cond_dim = cond_dim)

    def forward(self, seq, cond = None):
        if seq.dtype.is_floating_point:
            onehot = seq
        else:
            onehot = F.one_hot(clamp(seq, lo = 0), num_classes = self.dim_input).float()
            onehot = einx.where('b n, b n d, -> b n d', seq >= 0, onehot, 0.)

        out = self.conv(rearrange(onehot, 'b n d -> b d n'))
        return out + self.conv_block(out, cond)

class DownresBlock(Module):
    def __init__(
        self,
        dim,
        channels_to_add = 128,
        cond_dim = None
    ):
        super().__init__()
        dim_out = dim + channels_to_add
        self.pad = channels_to_add
        self.conv = ConvBlock(dim, width = 5, dim_out = dim_out, cond_dim = cond_dim)
        self.conv_out = ConvBlock(dim_out, width = 5, cond_dim = cond_dim)

    def forward(self, x, cond = None):
        x = reduce(x, 'b c (n pool) -> b c n', 'max', pool = 2)
        out = self.conv(x, cond) + pad_right_at_dim(x, self.pad, dim = 1)
        return self.conv_out(out, cond) + out

class UpresBlock(Module):
    def __init__(
        self,
        dim,
        channels_to_remove = 128,
        residual_scale_init = 0.9,
        has_skip = True,
        cond_dim = None
    ):
        super().__init__()
        dim_out = dim - channels_to_remove
        self.dim_out = dim_out

        self.conv = ConvBlock(dim, width = 5, dim_out = dim_out, cond_dim = cond_dim)
        self.unet_conv = ConvBlock(dim_out, width = 1, cond_dim = cond_dim) if has_skip else None
        self.conv_out = ConvBlock(dim_out, width = 5, cond_dim = cond_dim)
        self.residual_scale = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(
        self,
        x,
        skip = None,
        cond = None
    ):
        out = self.conv(x, cond) + slice_left_at_dim(x, self.dim_out, dim = 1)
        out = repeat(out, 'b c n -> b c (n upsample)', upsample = 2) * self.residual_scale

        if exists(self.unet_conv):
            assert exists(skip)
            out = out + self.unet_conv(skip, cond)

        return self.conv_out(out, cond) + out

# positions

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        max_positions = 8192
    ):
        super().__init__()
        num_freqs = dim // 2
        inv_freq = 1. / (torch.arange(num_freqs).float() + geomspace(1, max_positions - num_freqs + 1, num_freqs))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        device, dtype = self.inv_freq.device, self.inv_freq.dtype
        t = positions.to(device = device, dtype = dtype) if is_tensor(positions) else torch.arange(positions, device = device, dtype = dtype)
        return einx.multiply('... , j -> ... j', t, self.inv_freq).repeat_interleave(2, dim = -1)

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim = -1).flatten(-2)

def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# attention and pairwise blocks

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        kv_heads = 1,
        dim_head_qk = 128,
        dim_head_v = 192,
        dim_pairwise = None,
        softclamp_value = 10.
    ):
        super().__init__()
        dim_pairwise = default(dim_pairwise, dim)

        assert divisible_by(heads, kv_heads)
        groups = heads // kv_heads

        self.scale = dim_head_qk ** -0.5
        self.heads = heads
        self.kv_heads = kv_heads
        self.groups = groups
        self.softclamp_value = softclamp_value
        self.qkv_dim_splits = (dim_head_qk * heads, dim_head_qk * kv_heads, dim_head_v * kv_heads)

        self.to_qkv = LinearNoBias(dim, sum(self.qkv_dim_splits))
        self.to_out = Linear(dim_head_v * heads, dim)

        self.to_attn_bias = Sequential(
            nn.LayerNorm(dim_pairwise),
            nn.GELU(),
            LinearNoBias(dim_pairwise, heads),
            Rearrange('b i j (g h) -> b g h i j', g = groups)
        )

    def forward(
        self,
        x,
        pairwise = None,
        rotary_emb = None,
        polar_emb = None,
        attn_mask = None
    ):
        q, k, v = self.to_qkv(x).split(self.qkv_dim_splits, dim = -1)

        q = rearrange(q, 'b n (g h d) -> b g h n d', g = self.groups, h = self.kv_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.kv_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.kv_heads)

        if exists(rotary_emb):
            q, k = apply_rotary_pos_emb(rotary_emb, q), apply_rotary_pos_emb(rotary_emb, k)

        if exists(polar_emb):
            q, inverse_pack = pack_with_inverse(q, 'b * n d')
            sim = inverse_pack(compute_attn_similarity(q, k, polar_emb), 'b * i j')
        else:
            sim = einx.dot('b g h i d, b h j d -> b g h i j', q, k)

        sim = sim * self.scale

        if exists(pairwise):
            attn_bias = self.to_attn_bias(pairwise)
            expand_factor = size(sim, 'b g h i [j]') // size(attn_bias, 'b g h i [j]')
            sim = sim + repeat(attn_bias, 'b g h i j -> b g h (i r1) (j r2)', r1 = expand_factor, r2 = expand_factor)

        sim = softclamp(sim, value = self.softclamp_value)

        if exists(attn_mask):
            if is_shape(attn_mask, 'i j'):
                attn_mask = rearrange(attn_mask, 'i j -> 1 1 1 i j')
            elif is_shape(attn_mask, 'b i j'):
                attn_mask = rearrange(attn_mask, 'b i j -> b 1 1 i j')
            elif is_shape(attn_mask, 'b h i j'):
                attn_mask = rearrange(attn_mask, 'b (g h) i j -> b g h i j', g = self.groups)

            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einx.dot('b g h i j, b h j d -> b g h i d', attn, v)

        return self.to_out(rearrange(out, 'b g h n d -> b n (g h d)'))

class SingleToPairwise(Module):
    def __init__(
        self,
        dim,
        pool_size = 16,
        dim_pairwise = 128,
        heads = 32,
        dim_head = None,
        max_positions = 8192,
        dropout = 0.
    ):
        super().__init__()
        dim_head = default(dim_head, dim_pairwise)

        self.pool_size = pool_size
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.to_qk = LinearNoBias(dim, heads * dim_head * 2)
        self.qk_to_pairwise = Linear(heads, dim_pairwise)
        self.to_outer_abs = LinearNoBias(dim, dim_pairwise)
        self.rope = RotaryEmbedding(dim_head, max_positions = max_positions)
        self.norm_out = RMSNorm(dim_pairwise)
        self.dropout = nn.Dropout(dropout)

    def forward(self, single):
        single = self.norm(reduce(single, 'b (n pool) d -> b n d', 'mean', pool = self.pool_size))

        q, k = rearrange(self.to_qk(single), 'b n (two h d) -> two b h n d', two = 2, h = self.heads).unbind(0)
        rotary_emb = self.rope(size(single, 'b [n] d'))
        q, k = apply_rotary_pos_emb(rotary_emb, q), apply_rotary_pos_emb(rotary_emb, k)

        sim = einx.dot('b h i d, b h j d -> b i j h', q, k)
        outer_abs = einx.subtract('b i d, b j d -> b i j d', single, single).abs()

        return self.dropout(self.norm_out(self.qk_to_pairwise(sim) + self.to_outer_abs(outer_abs)))

class PairwiseRowAttention(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_qk = LinearNoBias(dim, dim * 2)
        self.to_v = Linear(dim, dim)

    def forward(self, x):
        q, k = self.to_qk(x).chunk(2, dim = -1)
        v = self.to_v(x)

        sim = einx.dot('b n i d, b n j d -> b n i j', q, k) * self.scale
        return einx.dot('b n i j, b n j d -> b n i d', sim.softmax(dim = -1), v)

def FeedForward(dim, *, expansion_factor = 2.):
    dim_inner = int(dim * expansion_factor)

    return Sequential(
        Linear(dim, dim_inner),
        nn.ReLU(),
        Linear(dim_inner, dim)
    )

# multi-head attention residuals
# https://arxiv.org/abs/2603.15031 (Attention Residuals, Kimi Team)
# https://arxiv.org/abs/2607.27230 (Multi-Head Attention Residuals, Luo et al.)
# https://arxiv.org/abs/2604.03297 (XAttnRes, Liu et al.)

def resize_1d(x, length, pooling = 'max'):
    src_len = size(x, 'b [n] d')

    if src_len == length:
        return x

    assert pooling in ('max', 'avg', 'mean'), f'unsupported pooling mode {pooling}'

    if src_len > length:
        if divisible_by(src_len, length):
            return reduce(x, 'b (n pool) d -> b n d', 'max' if pooling == 'max' else 'mean', pool = src_len // length)

        pool_fn = F.adaptive_max_pool1d if pooling == 'max' else F.adaptive_avg_pool1d
        return rearrange(pool_fn(rearrange(x, 'b n d -> b d n'), length), 'b d n -> b n d')

    if divisible_by(length, src_len):
        return repeat(x, 'b n d -> b (n upsample) d', upsample = length // src_len)

    return rearrange(F.interpolate(rearrange(x, 'b n d -> b d n'), size = length, mode = 'linear', align_corners = False), 'b d n -> b n d')

def LoRA(dim, dim_out, dim_lora = 8):
    down = LinearNoBias(dim, dim_lora)
    up = LinearNoBias(dim_lora, dim_out)
    nn.init.zeros_(up.weight)
    return Sequential(RMSNorm(dim), down, up)

class MultiHeadAttentionResidual(Module):
    def __init__(
        self,
        dim,
        *,
        dim_ins: tuple[int, ...] | None = None,
        heads = 8,
        pooling = 'max',
        dim_lora: int | None = 8,
        channel_first = False
    ):
        super().__init__()
        assert divisible_by(dim, heads), f'dim {dim} must be divisible by heads {heads}'
        assert pooling in ('max', 'avg', 'mean'), f'unsupported pooling mode {pooling}'

        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.pooling = pooling
        self.channel_first = channel_first

        has_lora = exists(dim_lora) and dim_lora > 0

        self.projs = ModuleList([LinearNoBias(d, dim) if d != dim else nn.Identity() for d in dim_ins]) if exists(dim_ins) else None
        self.loras = ModuleList([LoRA(d, dim, dim_lora) for d in dim_ins]) if (exists(dim_ins) and has_lora) else None
        self.lora = LoRA(dim, dim, dim_lora) if (not exists(dim_ins) and has_lora) else None

        self.norm = RMSNorm(dim)
        self.pseudo_query = nn.Parameter(torch.zeros(heads, dim_head))

    def forward(self, x, history = None, return_attn = False):
        if not exists(history) or (exists(self.projs) and len(history) == 0):
            return (x, None) if return_attn else x

        if self.channel_first:
            x = rearrange(x, 'b d n -> b n d')
            history = tree_map_tensor(lambda t: rearrange(t, 'b d n -> b n d'), history)

        length = size(x, 'b [n] d')

        if exists(self.projs):
            assert len(history) == len(self.projs), f'history must have {len(self.projs)} entries, got {len(history)}'
            values, keys = [], []

            for h, proj, maybe_lora in zip_longest(history, self.projs, default(self.loras, ())):
                h = resize_1d(h, length, pooling = self.pooling)
                value = proj(h)
                key = value + maybe_lora(h) if exists(maybe_lora) else value

                values.append(value)
                keys.append(self.norm(key))

            values = torch.stack([*values, x])
            keys = torch.stack([*keys, self.norm(x)])
        else:
            values = torch.stack([*(resize_1d(h, length, pooling = self.pooling) for h in history), x])
            keys = self.norm(values + self.lora(values) if exists(self.lora) else values)

        values, keys = map(lambda t: rearrange(t, 'l b n (h d) -> b l h n d', h = self.heads), (values, keys))

        logits = einx.dot('b l h n d, h d -> b l h n', keys, self.pseudo_query) * self.scale
        attn = logits.softmax(dim = 1)

        out = einx.dot('b l h n, b l h n d -> b n (h d)', attn, values)

        if self.channel_first:
            out = rearrange(out, 'b n d -> b d n')

        return (out, attn) if return_attn else out

# transformer

class NormWrapper(Module):
    def __init__(
        self,
        dim,
        block,
        dropout = 0.,
        cond_dim = None,
        layer_norm = False
    ):
        super().__init__()
        self.norm = FiLMNorm(dim, cond_dim, layer_norm = layer_norm)
        self.block = block
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, cond = None, **kwargs):
        return self.dropout(self.block(self.norm(x, cond), *args, **kwargs))

class TransformerTower(Module):
    def __init__(
        self,
        dim,
        *,
        depth = 11,
        heads = 8,
        kv_heads = 1,
        dim_head_qk = 128,
        dim_head_v = 192,
        dropout = 0.,
        pairwise_dropout = 0.,
        ff_expansion_factor = 2.,
        polar_pos_emb = False,
        max_positions = 8192,
        dim_pairwise = 128,
        pairwise_every_num_single_blocks = 2,
        single_to_pairwise_heads = 32,
        single_to_pairwise_dim_head = None,
        pool_size = 16,
        cond_dim = None,
        use_attn_res = False,
        attn_res_kwargs: dict | None = None
    ):
        super().__init__()
        attn_res_kwargs = default(attn_res_kwargs, dict())
        self.pairwise_every = pairwise_every_num_single_blocks
        self.polar_pos_emb = polar_pos_emb
        self.rotary_emb, self.polar_emb = None, None

        if polar_pos_emb:
            inv_freqs = 1. / (torch.arange(dim_head_qk).float() + geomspace(1, max_positions - dim_head_qk + 1, dim_head_qk))
            self.polar_emb = PoPE(dim_head_qk, heads = 1, inv_freqs = inv_freqs)
        else:
            self.rotary_emb = RotaryEmbedding(dim_head_qk, max_positions = max_positions)

        layers = []
        attn_res_layers = []

        for layer_index in range(depth):
            attn = NormWrapper(
                dim,
                Attention(dim = dim, heads = heads, kv_heads = kv_heads, dim_head_qk = dim_head_qk, dim_head_v = dim_head_v, dim_pairwise = dim_pairwise),
                dropout = dropout,
                cond_dim = cond_dim,
                layer_norm = True
            )
            ff = NormWrapper(
                dim,
                FeedForward(dim, expansion_factor = ff_expansion_factor),
                dropout = dropout,
                cond_dim = cond_dim,
                layer_norm = True
            )

            single_to_pairwise, pairwise_attn, pairwise_ff = None, None, None

            if divisible_by(layer_index, self.pairwise_every):
                single_to_pairwise = SingleToPairwise(
                    dim = dim,
                    dim_pairwise = dim_pairwise,
                    heads = single_to_pairwise_heads,
                    dim_head = single_to_pairwise_dim_head,
                    pool_size = pool_size,
                    max_positions = max_positions,
                    dropout = dropout
                )
                pairwise_attn = NormWrapper(dim_pairwise, PairwiseRowAttention(dim_pairwise), dropout = pairwise_dropout)
                pairwise_ff = NormWrapper(dim_pairwise, FeedForward(dim_pairwise, expansion_factor = ff_expansion_factor), dropout = pairwise_dropout)

            layers.append(ModuleList([attn, ff, single_to_pairwise, pairwise_attn, pairwise_ff]))

            if use_attn_res:
                attn_res_layers.append(ModuleList([
                    MultiHeadAttentionResidual(dim, **attn_res_kwargs),
                    MultiHeadAttentionResidual(dim, **attn_res_kwargs)
                ]))

        self.layers = ModuleList(layers)
        self.attn_res_layers = ModuleList(attn_res_layers) if use_attn_res else None
        self.final_attn_res = MultiHeadAttentionResidual(dim, **attn_res_kwargs) if use_attn_res else None
        self.dim_pairwise = dim_pairwise

    def forward(self, single, cond = None, attn_mask = None, history: list[Tensor] | None = None):
        seq_len = size(single, 'b [n] d')
        pairwise = None

        pos_emb = dict(polar_emb = self.polar_emb(seq_len)) if self.polar_pos_emb else dict(rotary_emb = self.rotary_emb(seq_len))

        history = list(default(history, ()))

        for (attn, ff, maybe_single_to_pair, maybe_pairwise_attn, maybe_pairwise_ff), (maybe_attn_res, maybe_ff_res) in zip_longest(self.layers, default(self.attn_res_layers, ()), fillvalue = (None, None)):
            if exists(maybe_attn_res):
                single, prev = maybe_attn_res(single, history), single
                history.append(prev)

            if exists(maybe_single_to_pair):
                pairwise = maybe_single_to_pair(single) + default(pairwise, 0.)
                pairwise = maybe_pairwise_attn(pairwise) + pairwise
                pairwise = maybe_pairwise_ff(pairwise) + pairwise

            single = attn(single, pairwise = pairwise, cond = cond, attn_mask = attn_mask, **pos_emb) + single

            if exists(maybe_ff_res):
                single, prev = maybe_ff_res(single, history), single
                history.append(prev)

            single = ff(single, cond = cond) + single

        if exists(self.final_attn_res):
            single = self.final_attn_res(single, history)

        return single, pairwise

# transformer unet

class TransformerUnet(Module):
    def __init__(
        self,
        dims = (768, 896, 1024, 1152, 1280, 1408, 1536),
        basepairs = 4,
        dna_embed_width = 15,
        cond_dim = None,
        transformer_kwargs = None,
        use_attn_res = False,
        attn_res_kwargs: dict | None = None
    ):
        super().__init__()
        transformer_kwargs = dict(default(transformer_kwargs, dict()))
        attn_res_kwargs = dict(default(attn_res_kwargs, dict()))
        replace_skips = attn_res_kwargs.pop('replace_skips', False)

        assert is_odd(dna_embed_width)
        assert len(dims) >= 2
        assert not (replace_skips and not use_attn_res), 'replace_skips requires use_attn_res = True'

        self.replace_skips = replace_skips
        self.attn_res_pooling = attn_res_kwargs.get('pooling', 'max')

        first_dim, *_, last_dim = dims
        self.dna_embed = DNAEmbed(first_dim, dim_input = basepairs, width = dna_embed_width, cond_dim = cond_dim)

        downs = [
            DownresBlock(dim_in, channels_to_add = dim_out - dim_in, cond_dim = cond_dim)
            for dim_in, dim_out in zip(dims[:-1], dims[1:])
        ]

        dims_rev = list(reversed(dims))
        up_in_dims = [last_dim, *dims_rev[:-1]]
        ups = [
            UpresBlock(dim_in, channels_to_remove = dim_in - dim_out, has_skip = not replace_skips, cond_dim = cond_dim)
            for dim_in, dim_out in zip(up_in_dims, dims_rev)
        ]

        self.downs = ModuleList(downs)
        self.ups = ModuleList(ups)

        self.down_attn_res = None
        self.up_attn_res = None
        self.unpooled_to_tower = None

        if use_attn_res:
            unet_attn_res_kwargs = dict(channel_first = True, **attn_res_kwargs)

            down_attn_res, up_attn_res = [], []
            history_dims = [first_dim]

            for dim_out in dims[1:]:
                down_attn_res.append(MultiHeadAttentionResidual(dim_out, dim_ins = tuple(history_dims), **unet_attn_res_kwargs))
                history_dims.append(dim_out)

            self.unpooled_to_tower = ModuleList([
                LinearNoBias(d, last_dim) if d != last_dim else nn.Identity()
                for d in history_dims
            ])

            history_dims.append(last_dim)

            for dim_out in dims_rev:
                up_attn_res.append(MultiHeadAttentionResidual(dim_out, dim_ins = tuple(history_dims), **unet_attn_res_kwargs))
                history_dims.append(dim_out)

            self.down_attn_res = ModuleList(down_attn_res)
            self.up_attn_res = ModuleList(up_attn_res)

            transformer_kwargs = dict(
                use_attn_res = True,
                attn_res_kwargs = attn_res_kwargs,
                **transformer_kwargs
            )

        self.transformer = TransformerTower(dim = last_dim, cond_dim = cond_dim, **transformer_kwargs)

    def forward(self, seq, cond = None, attn_mask = None):
        x = self.dna_embed(seq, cond)
        skips = [x]
        history = [x]

        for down, maybe_attn_res in zip_longest(self.downs, default(self.down_attn_res, ())):
            x = down(x, cond)

            if exists(maybe_attn_res):
                x = maybe_attn_res(x, history)
                history.append(x)

            skips.append(x)

        x = reduce(x, 'b d (n pool) -> b d n', 'max', pool = 2)
        x = rearrange(x, 'b d n -> b n d')

        tower_history = None
        if exists(self.unpooled_to_tower):
            seq_len = size(x, 'b [n] d')
            tower_history = [
                proj(resize_1d(rearrange(h, 'b d n -> b n d'), seq_len, pooling = self.attn_res_pooling))
                for h, proj in zip(history, self.unpooled_to_tower)
            ]

        single, pairwise = self.transformer(x, cond = cond, attn_mask = attn_mask, history = tower_history)

        x = rearrange(single, 'b n d -> b d n')
        history.append(x)

        for up, maybe_attn_res in zip_longest(self.ups, default(self.up_attn_res, ())):
            skip = None if self.replace_skips else skips.pop()
            x = up(x, skip = skip, cond = cond)

            if exists(maybe_attn_res):
                x = maybe_attn_res(x, history)
                history.append(x)

        return TransformerUnetOutput(rearrange(x, 'b d n -> b n d'), single, pairwise)

# output embeddings

class OutputEmbedding(Module):
    def __init__(
        self,
        input_dim,
        skip_dim = None,
        cond_dim = None
    ):
        super().__init__()
        self.norm = FiLMNorm(input_dim, cond_dim = cond_dim)
        self.double_features = Linear(input_dim, 2 * input_dim)
        self.skip_proj = LinearNoBias(skip_dim, 2 * input_dim) if exists(skip_dim) else None
        self.activation = nn.GLU(dim = -1)

    def forward(self, x, skip = None, cond = None):
        x = self.double_features(self.norm(x, cond))

        if exists(skip):
            assert exists(self.skip_proj)
            repeat_factor = size(x, 'b [n] d') // size(skip, 'b [m] d')
            x = x + repeat(self.skip_proj(skip), 'b n d -> b (n repeat) d', repeat = repeat_factor)

        return self.activation(x)

class OutputPairEmbedding(Module):
    def __init__(self, pair_dim = 128):
        super().__init__()
        self.norm = RMSNorm(pair_dim)

    def forward(self, x):
        return gelu_1702(self.norm(symmetrize(x)))

# output heads (dropconnect on weights per Methods)

class WeightDropoutLinear(Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias = True,
        weight_dropout = 0.
    ):
        super().__init__(in_features, out_features, bias = bias)
        self.weight_dropout = weight_dropout

    def forward(self, x):
        weight = F.dropout(self.weight, p = self.weight_dropout, training = self.training) if self.weight_dropout > 0. else self.weight
        return F.linear(x, weight, self.bias)

class TracksScaledPrediction(Module):
    def __init__(
        self,
        dim,
        num_tracks,
        weight_dropout = 0.1
    ):
        super().__init__()
        self.to_pred = WeightDropoutLinear(dim, num_tracks, weight_dropout = weight_dropout)

    def forward(self, x):
        return self.to_pred(x).exp()

class GenomeTracksHead(Module):
    def __init__(
        self,
        dim_1bp,
        dim_128bp,
        num_tracks,
        resolutions = (1, 128),
        weight_dropout = 0.1
    ):
        super().__init__()
        dim_by_res = {1: dim_1bp, 128: dim_128bp}
        self.resolutions = ModuleDict({
            f'resolution_{res}': TracksScaledPrediction(dim_by_res[res], num_tracks, weight_dropout = weight_dropout)
            for res in resolutions
        })

    def forward(self, embeds_1bp = None, embeds_128bp = None):
        embeds_by_res = {1: embeds_1bp, 128: embeds_128bp}
        out = dict()

        for res, embed in embeds_by_res.items():
            key = f'resolution_{res}'
            if key in self.resolutions:
                assert exists(embed)
                out[f'scaled_predictions_{res}bp'] = self.resolutions[key](embed)

        return out

class ContactMapHead(Module):
    def __init__(
        self,
        input_dim,
        num_tracks = 1,
        weight_dropout = 0.2
    ):
        super().__init__()
        self.to_pred = WeightDropoutLinear(input_dim, num_tracks, weight_dropout = weight_dropout)

    def forward(self, embeds_pair):
        return F.softplus(self.to_pred(embeds_pair))

ContactMapsHead = ContactMapHead

class SpliceSiteClassifier(Module):
    def __init__(
        self,
        input_dim,
        weight_dropout = 0.2
    ):
        super().__init__()
        self.linear = WeightDropoutLinear(input_dim, 5, weight_dropout = weight_dropout)

    def forward(self, embeds_1bp):
        return self.linear(embeds_1bp)

class SpliceSiteUsage(Module):
    def __init__(
        self,
        input_dim,
        n_contexts = 6,
        weight_dropout = 0.2
    ):
        super().__init__()
        self.linear = WeightDropoutLinear(input_dim, n_contexts, weight_dropout = weight_dropout)

    def forward(self, embeds_1bp):
        return self.linear(embeds_1bp).sigmoid()

class SpliceJunctionHead(Module):
    def __init__(
        self,
        input_dim,
        hidden_dim = 256,
        n_contexts = 6,
        rope_max_position = 2 ** 20,
        dropout = 0.2
    ):
        super().__init__()
        self.project = Linear(input_dim, hidden_dim)
        self.donor_scale = nn.Parameter(torch.ones(n_contexts, hidden_dim))
        self.donor_offset = nn.Parameter(torch.zeros(n_contexts, hidden_dim))
        self.acceptor_scale = nn.Parameter(torch.ones(n_contexts, hidden_dim))
        self.acceptor_offset = nn.Parameter(torch.zeros(n_contexts, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(hidden_dim, max_positions = rope_max_position)

    def modulate_and_rotate(self, x_proj, indices, scale, offset):
        safe_indices = clamp(indices.long(), lo = 0)
        x = batched_index_select(x_proj, safe_indices)

        x = einx.multiply('b p h, c h -> b p c h', x, scale)
        x = einx.add('b p c h, c h -> b p c h', x, offset)
        x = self.dropout(x)

        rotary_emb = rearrange(self.rope(safe_indices), 'b p h -> b p 1 h')
        return apply_rotary_pos_emb(rotary_emb, x)

    def forward(
        self,
        embeds_1bp,
        splice_donor_idx,
        splice_acceptor_idx
    ):
        x_proj = self.project(embeds_1bp)
        donor = self.modulate_and_rotate(x_proj, splice_donor_idx, self.donor_scale, self.donor_offset)
        acceptor = self.modulate_and_rotate(x_proj, splice_acceptor_idx, self.acceptor_scale, self.acceptor_offset)

        scores = F.softplus(einx.dot('b d c h, b a c h -> b d a c', donor, acceptor))
        valid_mask = einx.logical_and('b d, b a -> b d a 1', splice_donor_idx >= 0, splice_acceptor_idx >= 0)
        return scores.masked_fill(~valid_mask, 0.)

class SpliceSitesJunctionHead(Module):
    def __init__(
        self,
        input_dim,
        hidden_dim = 256,
        num_assays = 3,
        rope_max_position = 2 ** 20,
        dropout = 0.2
    ):
        super().__init__()
        self.project = Linear(input_dim, hidden_dim)
        self.num_assays = num_assays
        self.dropout = nn.Dropout(dropout)

        # (2 strands, 2 roles [donor, acceptor], num_assays, hidden_dim)

        self.scale = nn.Parameter(torch.ones(2, 2, num_assays, hidden_dim))
        self.offset = nn.Parameter(torch.zeros(2, 2, num_assays, hidden_dim))
        self.rope = RotaryEmbedding(hidden_dim, max_positions = rope_max_position)

    def forward(self, embeds_1bp, splice_site_positions = None):
        if not exists(splice_site_positions):
            return None

        orig_positions = splice_site_positions
        positions = splice_site_positions.to(device = embeds_1bp.device, dtype = torch.long)

        if is_shape(positions, 'b (s r) p', s = 2, r = 2):
            positions = rearrange(positions, 'b (s r) p -> b s r p', s = 2, r = 2)

        valid_mask = positions >= 0
        safe_positions = clamp(positions, lo = 0)

        site_embeds = batched_index_select(self.project(embeds_1bp), safe_positions)
        site_embeds = einx.multiply('b s r p h, s r t h -> b s r p t h', site_embeds, self.scale)
        site_embeds = einx.add('b s r p t h, s r t h -> b s r p t h', site_embeds, self.offset)
        site_embeds = self.dropout(site_embeds)

        rotary_emb = rearrange(self.rope(safe_positions), 'b s r p h -> b s r p 1 h')
        site_embeds = apply_rotary_pos_emb(rotary_emb, site_embeds)

        donors, acceptors = site_embeds.unbind(dim = 2)
        donor_valid, acceptor_valid = valid_mask.unbind(dim = 2)

        logits = einx.dot('b s d t h, b s a t h -> b s d a t', donors, acceptors)
        pred_counts = rearrange(F.softplus(logits), 'b s d a t -> b d a (s t)')
        pair_mask = einx.logical_and('b s d, b s a -> b s d a', donor_valid, acceptor_valid)
        splice_junction_mask = repeat(pair_mask, 'b s d a -> b d a (s t)', t = self.num_assays)

        flat_positions = orig_positions if is_shape(orig_positions, 'b sr p') else rearrange(positions, 'b s r p -> b (s r) p')

        return dict(
            predictions = pred_counts.masked_fill(~splice_junction_mask, 0.),
            splice_site_positions = flat_positions,
            splice_junction_mask = splice_junction_mask
        )

class MethylationHead(Module):
    def __init__(
        self,
        input_dim,
        num_channels = 2,
        dim_128bp = None,
        weight_dropout = 0.1
    ):
        super().__init__()
        self.to_pred = WeightDropoutLinear(input_dim, num_channels, weight_dropout = weight_dropout)
        self.to_pred_128bp = WeightDropoutLinear(dim_128bp, num_channels, weight_dropout = weight_dropout) if exists(dim_128bp) else None

    def forward(self, embeds_1bp, embeds_128bp = None):
        pred_1bp = self.to_pred(embeds_1bp).sigmoid()

        if not exists(self.to_pred_128bp) or not exists(embeds_128bp):
            return pred_1bp

        return dict(
            predictions_1bp = pred_1bp,
            predictions_128bp = self.to_pred_128bp(embeds_128bp).sigmoid()
        )

class ExonAttentionPool(Module):
    def __init__(
        self,
        input_dim,
        num_channels,
        dim_attn = 128,
        weight_dropout = 0.1,
        eps = 1e-7
    ):
        super().__init__()
        self.scale = dim_attn ** -0.5
        self.eps = eps
        self.to_q = WeightDropoutLinear(input_dim, dim_attn, bias = False, weight_dropout = weight_dropout)
        self.to_k = WeightDropoutLinear(input_dim, dim_attn, bias = False, weight_dropout = weight_dropout)
        self.to_pred = WeightDropoutLinear(input_dim, num_channels, weight_dropout = weight_dropout)

    def forward(self, embeds, gene_mask):
        gene_mask = gene_mask.to(embeds.dtype)
        gene_mean = einx.dot('b n g, b n d -> b g d', gene_mask, embeds) / (reduce(gene_mask, 'b n g -> b g 1', 'sum') + self.eps)

        q, k = self.to_q(gene_mean) * self.scale, self.to_k(embeds)
        attn = F.softplus(einx.dot('b g d, b n d -> b n g', q, k)) * gene_mask
        pooled = einx.dot('b n g, b n d -> b g d', attn, embeds) / (reduce(attn, 'b n g -> b g 1', 'sum') + self.eps)

        return F.softplus(self.to_pred(pooled))

class GeneExpressionHead(Module):
    def __init__(
        self,
        input_dim,
        num_channels,
        dim_128bp = None,
        dim_attn = 128,
        weight_dropout = 0.1,
        channel_strand_mask = None
    ):
        super().__init__()
        self.pool_1bp = ExonAttentionPool(input_dim, num_channels, dim_attn = dim_attn, weight_dropout = weight_dropout)
        self.pool_128bp = ExonAttentionPool(dim_128bp, num_channels, dim_attn = dim_attn, weight_dropout = weight_dropout) if exists(dim_128bp) else None

        if exists(channel_strand_mask):
            self.register_buffer('channel_strand_mask', channel_strand_mask.float())
        else:
            self.channel_strand_mask = None

    def forward(
        self,
        embeds_1bp,
        gene_mask = None,
        embeds_128bp = None,
        channel_strand_mask = None
    ):
        if not exists(gene_mask):
            return None

        channel_strand_mask = default(channel_strand_mask, self.channel_strand_mask)
        is_stranded = is_shape(gene_mask, 'b n s g')
        unstranded_mask = reduce(gene_mask, 'b n s g -> b n g', 'max') if is_stranded else gene_mask

        if is_stranded and exists(channel_strand_mask):
            gene_strand_mask = reduce(gene_mask.to(embeds_1bp.dtype), 'b n s g -> b s g', 'max')
            compat_mask = einx.dot('b s g, s c -> b g c', gene_strand_mask, channel_strand_mask.to(embeds_1bp.dtype)) > 0
        else:
            compat_mask = reduce(unstranded_mask, 'b n g -> b g 1', 'max') > 0

        pred_1bp = self.pool_1bp(embeds_1bp, unstranded_mask).masked_fill(~compat_mask, 0.)

        if not exists(self.pool_128bp) or not exists(embeds_128bp):
            return pred_1bp

        pool_factor = size(unstranded_mask, 'b [n] g') // size(embeds_128bp, 'b [m] d')
        mask_128bp = reduce(unstranded_mask, 'b (n pool) g -> b n g', 'max', pool = pool_factor)
        pred_128bp = self.pool_128bp(embeds_128bp, mask_128bp).masked_fill(~compat_mask, 0.)

        return dict(
            predictions_1bp = pred_1bp,
            predictions_128bp = pred_128bp
        )

# model

class AlphaGenome2(Module):
    def __init__(
        self,
        dims = (768, 896, 1024, 1152, 1280, 1408, 1536),
        basepairs = 4,
        dna_embed_width = 15,
        num_organisms = 2,
        num_conditions = 2048,
        dim_cond = 256,
        dim_cell_embed = 256,
        dim_adapter_hidden = 8192,
        output_embed_film = False,
        add_reference_heads = False,
        include_denoised_sc_gene_expr = False,
        transformer_kwargs = None,
        use_attn_res = False,
        attn_res_kwargs: dict | None = None
    ):
        super().__init__()
        transformer_kwargs = default(transformer_kwargs, dict())

        self.transformer_unet = TransformerUnet(
            dims = dims,
            basepairs = basepairs,
            dna_embed_width = dna_embed_width,
            cond_dim = dim_cond,
            transformer_kwargs = transformer_kwargs,
            use_attn_res = use_attn_res,
            attn_res_kwargs = attn_res_kwargs
        )

        first_dim, *_, last_dim = dims
        out_cond_dim = dim_cond if output_embed_film else None

        self.conditioning_embedder = ConditioningEmbedder(
            dim = dim_cond,
            num_taxonomies = num_organisms,
            num_conditions = num_conditions,
            dim_cell_embed = dim_cell_embed,
            dim_adapter_hidden = dim_adapter_hidden
        )

        self.outembed_128bp = OutputEmbedding(last_dim, cond_dim = out_cond_dim)
        self.outembed_1bp = OutputEmbedding(first_dim, skip_dim = last_dim, cond_dim = out_cond_dim)
        self.outembed_pair = OutputPairEmbedding(self.transformer_unet.transformer.dim_pairwise)

        self.num_organisms = num_organisms
        self.num_conditions = num_conditions
        self.dim_cond = dim_cond
        self.dim_1bp = first_dim
        self.dim_128bp = last_dim
        self.dim_contacts = self.transformer_unet.transformer.dim_pairwise

        self.shared_heads = ModuleDict()
        self.heads = ModuleDict()
        self.head_forward_arg_names = defaultdict(dict)
        self.head_forward_arg_maps = defaultdict(dict)

        if add_reference_heads:
            self.add_reference_heads(include_denoised_sc_gene_expr = include_denoised_sc_gene_expr)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def has_heads(self):
        return bool(self.shared_heads) or bool(self.heads)

    @property
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def add_head(
        self,
        organism_or_head_name,
        head_name_or_head = None,
        head = None,
        head_input_kwarg_names = None
    ):
        if isinstance(head_name_or_head, Module):
            organism = None
            head_name = organism_or_head_name
            head_input_kwarg_names = head
            head = head_name_or_head
        else:
            organism = organism_or_head_name
            head_name = head_name_or_head

        if isinstance(head_input_kwarg_names, str):
            head_input_kwarg_names = (head_input_kwarg_names,)

        scope_key = default(organism, '_shared')

        if exists(organism):
            if organism not in self.heads:
                self.heads[organism] = ModuleDict()
            self.heads[organism][head_name] = head
        else:
            self.shared_heads[head_name] = head

        head_forward_arg_names = list(signature(head.forward).parameters)
        self.head_forward_arg_names[scope_key][head_name] = default(head_input_kwarg_names, head_forward_arg_names)
        self.head_forward_arg_maps[scope_key][head_name] = dict(zip(head_input_kwarg_names, head_forward_arg_names)) if exists(head_input_kwarg_names) else dict()

    def add_heads(
        self,
        organism = None,
        num_tracks_1bp = None,
        num_tracks_128bp = None,
        num_tracks_contacts = None,
        num_splicing_contexts = None,
        hidden_dim_splice_juncs = 256,
        num_methylation_channels = 2
    ):
        assert all(exists(v) for v in (num_tracks_1bp, num_tracks_128bp, num_tracks_contacts, num_splicing_contexts))

        heads = [
            ('1bp_tracks', TracksScaledPrediction(self.dim_1bp, num_tracks_1bp), ('embeds_1bp',)),
            ('128bp_tracks', TracksScaledPrediction(self.dim_128bp, num_tracks_128bp), ('embeds_128bp',)),
            ('contact_head', ContactMapHead(self.dim_contacts, num_tracks_contacts)),
            ('splice_logits', SpliceSiteClassifier(self.dim_1bp)),
            ('splice_usage', SpliceSiteUsage(self.dim_1bp, num_splicing_contexts)),
            ('splice_juncs', SpliceJunctionHead(self.dim_1bp, hidden_dim_splice_juncs, num_splicing_contexts)),
        ]

        if exists(num_methylation_channels) and num_methylation_channels > 0:
            heads.append(('methylation', MethylationHead(self.dim_1bp, num_methylation_channels)))

        for head_name, head_mod, *maybe_arg_names in heads:
            self.add_head(organism, head_name, head_mod, *(maybe_arg_names or (None,)))

    def add_reference_heads(
        self,
        organism = None,
        genome_track_heads = None,
        num_tracks_contacts = ag2_contact_map_channels,
        num_methylation_channels = ag2_methylation_channels,
        num_splice_usage_tracks = ag2_splice_usage_channels,
        splice_junction_hidden_dim = ag2_splice_junction_hidden_dim,
        splice_junction_num_assays = ag2_splice_junction_num_assays,
        num_bulk_gene_expr_channels = ag2_bulk_gene_expr_channels,
        num_sc_gene_expr_channels = ag2_sc_gene_expr_channels,
        bulk_channel_strand_mask = None,
        sc_channel_strand_mask = None,
        include_denoised_sc_gene_expr = False,
        aux_128bp_heads = True
    ):
        genome_track_heads = default(genome_track_heads, ag2_genome_track_heads)
        aux_dim_128bp = self.dim_128bp if aux_128bp_heads else None

        for head_name, (num_tracks, resolutions) in genome_track_heads.items():
            self.add_head(organism, head_name, GenomeTracksHead(
                dim_1bp = self.dim_1bp,
                dim_128bp = self.dim_128bp,
                num_tracks = num_tracks,
                resolutions = resolutions
            ))

        self.add_head(organism, 'contact_maps', ContactMapHead(self.dim_contacts, num_tracks_contacts))
        self.add_head(organism, 'methylation', MethylationHead(self.dim_1bp, num_methylation_channels, dim_128bp = aux_dim_128bp))
        self.add_head(organism, 'splice_sites_classification', SpliceSiteClassifier(self.dim_1bp))
        self.add_head(organism, 'splice_sites_usage', SpliceSiteUsage(self.dim_1bp, num_splice_usage_tracks))
        self.add_head(organism, 'splice_sites_junction', SpliceSitesJunctionHead(
            self.dim_1bp,
            splice_junction_hidden_dim,
            splice_junction_num_assays
        ))
        self.add_gene_expression_head(organism, 'gene_expression', num_bulk_gene_expr_channels, aux_128bp = aux_128bp_heads, channel_strand_mask = bulk_channel_strand_mask)
        self.add_gene_expression_head(organism, 'single_cell_gene_expression', num_sc_gene_expr_channels, aux_128bp = aux_128bp_heads, channel_strand_mask = sc_channel_strand_mask)

        if include_denoised_sc_gene_expr:
            self.add_denoised_sc_gene_expression_head(organism, aux_128bp = aux_128bp_heads)

    def add_gene_expression_head(
        self,
        organism = None,
        head_name = 'gene_expression',
        num_channels = 14,
        aux_128bp = True,
        channel_strand_mask = None
    ):
        dim_128bp = self.dim_128bp if aux_128bp else None
        self.add_head(organism, head_name, GeneExpressionHead(
            self.dim_1bp,
            num_channels,
            dim_128bp = dim_128bp,
            channel_strand_mask = channel_strand_mask
        ))

    def add_gene_expression_heads(self, organism = None, aux_128bp = True, bulk_channel_strand_mask = None, sc_channel_strand_mask = None):
        self.add_gene_expression_head(organism, 'gene_expression', ag2_bulk_gene_expr_channels, aux_128bp = aux_128bp, channel_strand_mask = bulk_channel_strand_mask)
        self.add_gene_expression_head(organism, 'single_cell_gene_expression', ag2_sc_gene_expr_channels, aux_128bp = aux_128bp, channel_strand_mask = sc_channel_strand_mask)

    def add_denoised_sc_gene_expression_head(
        self,
        organism = None,
        head_name = 'denoised_single_cell_gene_expression',
        aux_128bp = True,
        channel_strand_mask = None
    ):
        channel_strand_mask = default(channel_strand_mask, torch.eye(ag2_denoised_sc_gene_expr_channels))
        self.add_gene_expression_head(organism, head_name, ag2_denoised_sc_gene_expr_channels, aux_128bp = aux_128bp, channel_strand_mask = channel_strand_mask)

    def freeze_for_condition_refinement_(self):
        self.requires_grad_(False)
        self.conditioning_embedder.taxonomy_emb.requires_grad_(True)
        self.conditioning_embedder.condition_emb.requires_grad_(True)

        for mod in self.modules():
            if isinstance(mod, FiLM):
                mod.requires_grad_(True)

    def freeze_for_expression_post_training_(self, denoised_head_name = 'denoised_single_cell_gene_expression'):
        self.requires_grad_(False)
        self.conditioning_embedder.cell_adapter.requires_grad_(True)

        if denoised_head_name in self.shared_heads:
            self.shared_heads[denoised_head_name].requires_grad_(True)

        for heads in self.heads.values():
            if denoised_head_name in heads:
                heads[denoised_head_name].requires_grad_(True)

    def get_embeds(
        self,
        seq,
        organism_index = 0,
        condition_index = None,
        cell_embed = None,
        attn_mask = None,
        return_unet_transformer_out = False,
        flow_cond: Tensor | None = None
    ):
        seq = parse_dna_seq(seq, device = self.device)
        batch = size(seq, '[b] ...')

        if isinstance(organism_index, int):
            organism_index = torch.full((batch,), organism_index, device = seq.device, dtype = torch.long)

        if isinstance(condition_index, int):
            condition_index = torch.full((batch,), condition_index, device = seq.device, dtype = torch.long)

        cond = self.conditioning_embedder(organism_index, condition_index = condition_index, cell_embed = cell_embed)
        cond = cond + flow_cond.type_as(cond) if exists(flow_cond) else cond
        unet_transformer_out = self.transformer_unet(seq, cond = cond, attn_mask = attn_mask)
        unet_out, single, pairwise = unet_transformer_out

        embeds_128bp = self.outembed_128bp(single, cond = cond)
        embeds_1bp = self.outembed_1bp(unet_out, skip = embeds_128bp, cond = cond)
        embeds_pair = self.outembed_pair(pairwise)
        embeds = Embeds(embeds_1bp, embeds_128bp, embeds_pair)

        if return_unet_transformer_out:
            return embeds, unet_transformer_out

        return embeds

    def _apply_track_mask(self, head_output, mask):
        num_tracks = size(mask, '[c]')
        return tree_map_tensor(
            lambda t: t[..., mask] if is_shape(t, '_ ... c', c = num_tracks) else t,
            head_output
        )

    def _run_head_dict(self, scope_key, heads, head_inputs, requested_heads = None, track_masks = None):
        scope_out = dict()

        for head_name, head in heads.items():
            if exists(requested_heads) and head_name not in requested_heads:
                continue

            arg_names = self.head_forward_arg_names[scope_key][head_name]
            arg_map = self.head_forward_arg_maps[scope_key][head_name]
            kwargs = {arg_map.get(name, name): head_inputs[name] for name in arg_names if name in head_inputs}
            head_out = head(**kwargs)

            if exists(track_masks) and head_name in track_masks:
                head_out = self._apply_track_mask(head_out, track_masks[head_name])

            scope_out[head_name] = head_out

        return scope_out

    def run_heads(self, head_inputs, requested_heads = None, track_masks = None):
        if self.shared_heads and not self.heads:
            return self._run_head_dict('_shared', self.shared_heads, head_inputs, requested_heads, track_masks)

        out = dict()

        if self.shared_heads:
            shared_out = self._run_head_dict('_shared', self.shared_heads, head_inputs, requested_heads, track_masks)
            if shared_out:
                out['shared'] = shared_out

        for organism, heads in self.heads.items():
            organism_out = self._run_head_dict(organism, heads, head_inputs, requested_heads, track_masks)
            if organism_out:
                out[organism] = organism_out

        return out

    def _gather_head_inputs(self, embeds, organism_index, head_kwargs, unet_transformer_out = None):
        embeds_1bp, embeds_128bp, embeds_pair = embeds
        head_inputs = dict(
            embeds_1bp = embeds_1bp,
            embeds_128bp = embeds_128bp,
            embeds_pair = embeds_pair,
            organism_index = organism_index,
        )

        if exists(unet_transformer_out):
            head_inputs.update(
                unet_output = unet_transformer_out.unet_output,
                single_repr = unet_transformer_out.single_repr,
                pairwise_repr = unet_transformer_out.pairwise_repr,
            )

        assert set(head_inputs).isdisjoint(head_kwargs)
        head_inputs.update(head_kwargs)
        return head_inputs

    def forward(
        self,
        seq,
        organism_index = 0,
        return_embeds = False,
        condition_index = None,
        cell_embed = None,
        attn_mask = None,
        requested_heads = None,
        track_masks = None,
        **head_kwargs
    ):
        embeds, unet_transformer_out = self.get_embeds(
            seq,
            organism_index,
            condition_index = condition_index,
            cell_embed = cell_embed,
            attn_mask = attn_mask,
            return_unet_transformer_out = True
        )

        if return_embeds or not self.has_heads:
            return embeds

        head_inputs = self._gather_head_inputs(embeds, organism_index, head_kwargs, unet_transformer_out)
        return self.run_heads(head_inputs, requested_heads = requested_heads, track_masks = track_masks)

    def inference(
        self,
        seq,
        organism_index = 0,
        *,
        requested_heads = None,
        track_masks = None,
        embeds = None,
        condition_index = None,
        cell_embed = None,
        attn_mask = None,
        return_embeds = False,
        **head_kwargs
    ):
        if not exists(embeds):
            embeds = self.get_embeds(
                seq,
                organism_index,
                condition_index = condition_index,
                cell_embed = cell_embed,
                attn_mask = attn_mask
            )

        if not self.has_heads:
            return embeds if return_embeds else dict()

        head_inputs = self._gather_head_inputs(embeds, organism_index, head_kwargs)
        out = self.run_heads(head_inputs, requested_heads = requested_heads, track_masks = track_masks)
        return (out, embeds) if return_embeds else out

    def compile(self, **compile_kwargs):
        return torch.compile(self, **compile_kwargs)

    @torch.no_grad()
    def score_variant(
        self,
        seq_ref,
        seq_alt,
        organism_index = 0,
        scorer = None,
        settings = None,
        variant = None,
        interval = None,
        track_metadata = None,
        **kwargs
    ):
        from alphagenome_pytorch.scoring.variant_scoring import score_variant_with_model

        seq_ref = parse_dna_seq(seq_ref, device = self.device)
        seq_alt = parse_dna_seq(seq_alt, device = self.device)

        return score_variant_with_model(
            self,
            seq_ref,
            seq_alt,
            organism_index,
            scorer,
            settings,
            variant,
            interval,
            track_metadata,
            **kwargs
        )

class FlowAlphaGenome2(AlphaGenome2):
    # rectified flow (Liu et al., 2022) + XM best-of-k (Gladstone et al., 2026)
    def __init__(
        self,
        *args,
        channels: int = 12,
        target_scaler: TargetScaler | None = None,
        head: Module | None = None,
        loss_fn: Module | None = None,
        best_of_k: int = 2,
        noise_std: float = 0.5,
        noise_mean: float = 0.,
        time_fourier_scale: float = 16.,
        time_to_loss_weight_fn: Callable[[Tensor], Tensor] | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.noise_std, self.noise_mean, self.time_to_loss_weight_fn = noise_std, noise_mean, time_to_loss_weight_fn
        self.target_scaler = target_scaler
        self.head = default(head, TracksScaledPrediction(self.dim_1bp, channels))
        self.loss_fn = default(loss_fn, MultinomialLoss())
        self.best_of_k = best_of_k
        self.to_flow = Linear(channels, self.dim_1bp)
        self.register_buffer('time_freqs', torch.randn(self.dim_cond // 2) * time_fourier_scale)

    def _predict(self, seq, x, t, organism_index = 0, **kwargs):
        emb = einx.multiply('..., d -> ... d', t, self.time_freqs)
        flow_cond = torch.cat((emb.sin(), emb.cos()), dim = -1)

        embeds = self.get_embeds(seq, organism_index, flow_cond = flow_cond, **kwargs)
        embeds = embeds._replace(embeds_1bp = embeds.embeds_1bp + self.to_flow(x))

        return self.head(embeds.embeds_1bp)

    def _scale(self, x):
        return self.target_scaler(x) if exists(self.target_scaler) else x

    def _inverse(self, x):
        return self.target_scaler.inverse(x) if exists(self.target_scaler) else x

    def _forward_loss(self, seq, target, t, organism_index = 0, **kwargs):
        noise = torch.randn_like(target)
        noise = noise * self.noise_std + self.noise_mean

        noised = noise.lerp(target, pad_right_ndim_to(t, target.ndim))
        pred_target = self._predict(seq, noised, t, organism_index = organism_index, **kwargs)

        return self.loss_fn(pred_target, target)

    def forward(self, seq, organism_index = 0, target = None, **kwargs):
        if not exists(target):
            return super().forward(seq, organism_index, **kwargs)

        target = self._scale(target)
        t = torch.rand(target.shape[0], device = target.device)

        losses = torch.stack([
            self._forward_loss(seq, target, t, organism_index = organism_index, **kwargs)
            for _ in range(self.best_of_k)
        ])
        loss = losses.min()

        loss = loss * self.time_to_loss_weight_fn(t) if exists(self.time_to_loss_weight_fn) else loss

        return loss

    @torch.no_grad()
    def sample(self, seq, steps = 10, organism_index = 0, **kwargs):
        seq = parse_dna_seq(seq, device = self.device)

        shape = (*seq.shape[:2], self.to_flow.in_features)
        noise = torch.randn(shape, device = self.device)
        x = noise * self.noise_std + self.noise_mean

        for t in torch.linspace(0., 1., steps + 1, device = x.device)[:-1]:
            pred_target = self._predict(seq, x, t, organism_index = organism_index, **kwargs)
            pred_flow = (pred_target - x) / (1. - t)
            x = x + pred_flow / steps

        return self._inverse(x)
