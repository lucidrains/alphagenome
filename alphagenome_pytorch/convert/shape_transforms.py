from __future__ import annotations

import torch
from torch import Tensor


def transpose_linear_weight(w: Tensor) -> Tensor:
    """Haiku Linear uses (in, out); PyTorch Linear uses (out, in)."""
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(w.shape)}")
    return w.t()


def transpose_conv1d_weight(w: Tensor) -> Tensor:
    """Haiku/JAX conv kernels are (width, in, out); PyTorch Conv1d is (out, in, width)."""
    if w.ndim != 3:
        raise ValueError(f"expected 3D conv weight, got shape {tuple(w.shape)}")
    return w.permute(2, 1, 0).contiguous()

def transpose_linear_weight_to_conv1d(w: Tensor) -> Tensor:
    """Haiku Linear uses (in, out); PyTorch Conv1d (width=1) uses (out, in, 1)."""
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape {tuple(w.shape)}")
    return transpose_linear_weight(w).unsqueeze(-1).contiguous()


def concat_linear_weights(*ws: Tensor, dim: int = 0) -> Tensor:
    """Concat multiple Haiku Linear weights into one PyTorch weight.

    Typical use: combine Q/K/V projections into a single `to_qkv.weight`.

    Each input is expected to be Haiku-shaped (in, out_i). We transpose each to
    (out_i, in) and concatenate along `dim` (default: output/features dimension).
    """
    if len(ws) == 0:
        raise ValueError("expected at least one weight to concatenate")

    transposed = [transpose_linear_weight(w) for w in ws]

    # Sanity check: shared input dim after transpose.
    in_dims = {t.shape[1] for t in transposed}
    if len(in_dims) != 1:
        shapes = ", ".join(str(tuple(t.shape)) for t in transposed)
        raise ValueError(f"all weights must share the same input dim; got {shapes}")

    return torch.cat(transposed, dim=dim).contiguous()


def squeeze_to_1d(w: Tensor) -> Tensor:
    """Squeeze BatchRMSNorm params from [1, 1, dim] or [1, 1, 1, dim] to [dim]."""
    return w.squeeze()


def bake_standardized_conv1d_weight(
    *,
    w: Tensor,
    scale: Tensor,
    bias: Tensor,
    fan_in: int | None = None,
    min_scale: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Bake AlphaGenome StandardizedConv1D params into a plain PyTorch Conv1d kernel.

    Matches `alphagenome_research.model.convolutions.StandardizedConv1D`:
      - w: (width, in, out)
      - scale: (1, 1, out) (broadcastable)
      - bias: (out,)

    Returns:
      (weight_pt, bias_pt) where:
        - weight_pt is PyTorch Conv1d shape (out, in, width)
        - bias_pt is (out,)
    """
    if w.ndim != 3:
        raise ValueError(f"expected w with shape (width, in, out), got {tuple(w.shape)}")
    width, in_ch, out_ch = w.shape

    if bias.ndim != 1 or bias.shape[0] != out_ch:
        raise ValueError(f"expected bias shape ({out_ch},), got {tuple(bias.shape)}")

    # Allow passing scale as (out,) or (1,1,out) or any broadcastable variant.
    scale = scale.reshape((-1,)) if scale.ndim == 1 else scale

    if fan_in is None:
        fan_in = width * in_ch

    # Center per output channel: mean/var over (width, in_ch).
    w_centered = w - w.mean(dim=(0, 1), keepdim=True)

    # JAX `jnp.var` uses population variance (ddof=0).
    var = w_centered.var(dim=(0, 1), keepdim=True, unbiased=False)

    # `scale` is a learned per-out-channel parameter in the reference code.
    # Reference: `scale * rsqrt(max(fan_in * var, 1e-4))`
    denom = torch.maximum((fan_in * var).to(w_centered.dtype), torch.tensor(min_scale, dtype=w_centered.dtype, device=w_centered.device))
    w_standardized = w_centered * scale * torch.rsqrt(denom)

    return transpose_conv1d_weight(w_standardized), bias.contiguous()
