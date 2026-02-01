import torch

from alphagenome_pytorch.convert.shape_transforms import (
    bake_standardized_conv1d_weight,
    concat_linear_weights,
    transpose_conv1d_weight,
    transpose_linear_weight,
)


def test_transpose_linear_weight():
    w = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (in=2, out=3)
    wt = transpose_linear_weight(w)
    assert wt.shape == (3, 2)
    assert torch.allclose(wt, torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))


def test_transpose_conv1d_weight():
    # Haiku: (width, in, out)
    w = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    wt = transpose_conv1d_weight(w)
    assert wt.shape == (3, 2, 5)

    # spot-check a couple indices
    # Haiku w[width, in, out] -> Torch wt[out, in, width]
    assert wt[0, 0, 0].item() == w[0, 0, 0].item()
    assert wt[2, 1, 4].item() == w[4, 1, 2].item()


def test_concat_linear_weights_ordering():
    # Haiku Linear weights are (in, out)
    q = torch.tensor([[1.0], [2.0]])  # (2, 1)
    k = torch.tensor([[3.0], [4.0]])  # (2, 1)
    v = torch.tensor([[5.0], [6.0]])  # (2, 1)

    combined = concat_linear_weights(q, k, v)  # -> (out_total=3, in=2)
    assert combined.shape == (3, 2)

    expected = torch.tensor(
        [
            [1.0, 2.0],  # q.T
            [3.0, 4.0],  # k.T
            [5.0, 6.0],  # v.T
        ]
    )
    assert torch.allclose(combined, expected)


def test_bake_standardized_conv1d_weight_matches_reference_math():
    torch.manual_seed(0)

    width, in_ch, out_ch = 5, 4, 7
    w = torch.randn(width, in_ch, out_ch, dtype=torch.float32)
    scale = torch.randn(1, 1, out_ch, dtype=torch.float32)
    bias = torch.randn(out_ch, dtype=torch.float32)

    baked_w, baked_b = bake_standardized_conv1d_weight(
        w=w, scale=scale, bias=bias, fan_in=width * in_ch
    )

    # Manual reference (mirrors alphagenome_research.model.convolutions.StandardizedConv1D)
    w_centered = w - w.mean(dim=(0, 1), keepdim=True)
    var = w_centered.var(dim=(0, 1), keepdim=True, unbiased=False)
    denom = torch.maximum(width * in_ch * var, torch.tensor(1e-4, dtype=w.dtype))
    w_std = w_centered * scale * torch.rsqrt(denom)
    expected_w = w_std.permute(2, 1, 0).contiguous()  # (out, in, width)

    assert baked_w.shape == expected_w.shape == (out_ch, in_ch, width)
    assert torch.allclose(baked_w, expected_w, rtol=0, atol=0)
    assert torch.allclose(baked_b, bias, rtol=0, atol=0)

