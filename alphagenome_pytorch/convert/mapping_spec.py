"""JAX/Haiku -> PyTorch key mapping specification.

This module defines the explicit mapping between JAX/Haiku parameter keys and
PyTorch state_dict keys, including rules for:
- Linear weight transposition (in, out) -> (out, in)
- Conv1d weight transposition (width, in, out) -> (out, in, width)
- QKV weight concatenation
- StandardizedConv1D baking

The mapping uses EXACT keys from the JAX checkpoint - no transformations needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

import torch
from torch import Tensor

from .shape_transforms import (
    bake_standardized_conv1d_weight,
    concat_linear_weights,
    squeeze_to_1d,
    transpose_conv1d_weight,
    transpose_linear_weight,
    transpose_linear_weight_to_conv1d,
)


class TransformType(Enum):
    """Types of weight transformations."""
    IDENTITY = auto()           # No transformation (e.g., bias, scalar)
    TRANSPOSE_LINEAR = auto()   # Haiku (in, out) -> Torch (out, in)
    TRANSPOSE_CONV1D = auto()   # Haiku (w, in, out) -> Torch (out, in, w)
    TRANSPOSE_LINEAR_TO_CONV1D = auto()  # Haiku (in, out) -> Torch Conv1d (out, in, 1)
    CONCAT_QKV = auto()         # Concat Q, K, V weights into one
    CONCAT_QK = auto()          # Concat Q, K weights into one
    STACK_QK_BIAS = auto()      # Stack Q/K relative bias
    BAKE_STANDARDIZED_CONV = auto()  # Bake StandardizedConv1D
    SQUEEZE_TO_1D = auto()      # Squeeze [1,1,dim] or [1,1,1,dim] to [dim]
    UNSQUEEZE_SCALAR = auto()   # Unsqueeze scalar [] to [1]


@dataclass
class ParamMapping:
    """Mapping for a single parameter or group of parameters."""
    jax_keys: list[str]  # One or more JAX keys to combine
    torch_key: str       # Target PyTorch state_dict key
    transform: TransformType = TransformType.IDENTITY
    extra_jax_keys: dict[str, str] = field(default_factory=dict)  # For baking, etc.
    optional: bool = False  # If True, skip if JAX keys not found
    select_index: int | None = None  # If set, select this index along dim 0 before transform


def apply_transform(
    transform: TransformType,
    tensors: list[Tensor],
    extra_tensors: dict[str, Tensor] | None = None,
    select_index: int | None = None,
) -> Tensor | tuple[Tensor, ...]:
    """Apply the specified transformation to the input tensor(s)."""
    extra_tensors = extra_tensors or {}

    if select_index is not None:
        tensors = [t[select_index] for t in tensors]
        extra_tensors = {k: v[select_index] for k, v in extra_tensors.items()}

    if transform == TransformType.IDENTITY:
        assert len(tensors) == 1, f"IDENTITY expects 1 tensor, got {len(tensors)}"
        return tensors[0]

    elif transform == TransformType.TRANSPOSE_LINEAR:
        assert len(tensors) == 1, f"TRANSPOSE_LINEAR expects 1 tensor, got {len(tensors)}"
        return transpose_linear_weight(tensors[0])

    elif transform == TransformType.TRANSPOSE_CONV1D:
        assert len(tensors) == 1, f"TRANSPOSE_CONV1D expects 1 tensor, got {len(tensors)}"
        return transpose_conv1d_weight(tensors[0])

    elif transform == TransformType.TRANSPOSE_LINEAR_TO_CONV1D:
        assert len(tensors) == 1, f"TRANSPOSE_LINEAR_TO_CONV1D expects 1 tensor, got {len(tensors)}"
        return transpose_linear_weight_to_conv1d(tensors[0])

    elif transform == TransformType.CONCAT_QKV:
        assert len(tensors) == 3, f"CONCAT_QKV expects 3 tensors (Q, K, V), got {len(tensors)}"
        return concat_linear_weights(*tensors)

    elif transform == TransformType.CONCAT_QK:
        assert len(tensors) == 2, f"CONCAT_QK expects 2 tensors (Q, K), got {len(tensors)}"
        return concat_linear_weights(*tensors)

    elif transform == TransformType.STACK_QK_BIAS:
        assert len(tensors) == 2, f"STACK_QK_BIAS expects 2 tensors (Q, K), got {len(tensors)}"
        return torch.stack(tensors, dim=0)

    elif transform == TransformType.BAKE_STANDARDIZED_CONV:
        assert len(tensors) == 1, f"BAKE_STANDARDIZED_CONV expects 1 tensor (w), got {len(tensors)}"
        assert "scale" in extra_tensors and "bias" in extra_tensors, \
            "BAKE_STANDARDIZED_CONV requires 'scale' and 'bias' in extra_tensors"
        return bake_standardized_conv1d_weight(
            w=tensors[0],
            scale=extra_tensors["scale"],
            bias=extra_tensors["bias"],
        )

    elif transform == TransformType.SQUEEZE_TO_1D:
        assert len(tensors) == 1, f"SQUEEZE_TO_1D expects 1 tensor, got {len(tensors)}"
        return squeeze_to_1d(tensors[0])

    elif transform == TransformType.UNSQUEEZE_SCALAR:
        assert len(tensors) == 1, f"UNSQUEEZE_SCALAR expects 1 tensor, got {len(tensors)}"
        return tensors[0].unsqueeze(0)

    else:
        raise ValueError(f"Unknown transform type: {transform}")


# ============================================================================
# HELPER FUNCTIONS FOR JAX KEY CONSTRUCTION
# ============================================================================
# These helpers construct the exact JAX key paths from the checkpoint.

def _jax_block_suffix(idx: int) -> str:
    """Get JAX block suffix: '' for idx=0, '_{idx}' for idx>0."""
    return "" if idx == 0 else f"_{idx}"


def build_trunk_mapping() -> list[ParamMapping]:
    """Build the mapping for the trunk (non-head) parameters.

    This is the main body of the AlphaGenome model including:
    - DNA embedder
    - UNet encoder/decoder
    - Transformer tower
    - Output embeddings

    Returns:
        List of ParamMapping objects defining the JAX -> PyTorch conversion.
    """
    mappings = []

    # ========================================================================
    # DNA Embedder
    # ========================================================================
    # JAX: alphagenome/sequence_encoder/dna_embedder/conv1_d/...
    # JAX: alphagenome/sequence_encoder/dna_embedder/conv_block/...
    jax_dna = "alphagenome/sequence_encoder/dna_embedder"
    mappings.extend([
        # Main conv (width=15)
        ParamMapping(
            jax_keys=[f"{jax_dna}/conv1_d/w"],
            torch_key="transformer_unet.dna_embed.conv.weight",
            transform=TransformType.TRANSPOSE_CONV1D,
        ),
        ParamMapping(
            jax_keys=[f"{jax_dna}/conv1_d/b"],
            torch_key="transformer_unet.dna_embed.conv.bias",
        ),
        # Conv block (width=5)
        ParamMapping(
            jax_keys=[f"{jax_dna}/conv_block/standardized_conv1_d/w"],
            torch_key="transformer_unet.dna_embed.pointwise.net.2.weight",
            transform=TransformType.BAKE_STANDARDIZED_CONV,
            extra_jax_keys={
                "scale": f"{jax_dna}/conv_block/standardized_conv1_d/scale",
                "bias": f"{jax_dna}/conv_block/standardized_conv1_d/bias",
            },
        ),
        ParamMapping(
            jax_keys=[f"{jax_dna}/conv_block/rms_batch_norm/scale"],
            torch_key="transformer_unet.dna_embed.pointwise.net.0.gamma",
            transform=TransformType.SQUEEZE_TO_1D,
        ),
        ParamMapping(
            jax_keys=[f"{jax_dna}/conv_block/rms_batch_norm/offset"],
            torch_key="transformer_unet.dna_embed.pointwise.net.0.beta",
            transform=TransformType.SQUEEZE_TO_1D,
        ),
    ])

    # ========================================================================
    # UNet Down Blocks (sequence_encoder)
    # ========================================================================
    # JAX: alphagenome/sequence_encoder/downres_block_{i}/...
    for i in range(6):  # 6 down blocks
        jax_prefix = f"alphagenome/sequence_encoder/downres_block_{i}"
        torch_prefix = f"transformer_unet.downs.{i}"

        # First conv block
        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_block/standardized_conv1_d/w"],
                torch_key=f"{torch_prefix}.conv.net.2.weight",
                transform=TransformType.BAKE_STANDARDIZED_CONV,
                extra_jax_keys={
                    "scale": f"{jax_prefix}/conv_block/standardized_conv1_d/scale",
                    "bias": f"{jax_prefix}/conv_block/standardized_conv1_d/bias",
                },
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_block/rms_batch_norm/scale"],
                torch_key=f"{torch_prefix}.conv.net.0.gamma",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_block/rms_batch_norm/offset"],
                torch_key=f"{torch_prefix}.conv.net.0.beta",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

        # Second conv block (conv_out)
        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_block_1/standardized_conv1_d/w"],
                torch_key=f"{torch_prefix}.conv_out.net.2.weight",
                transform=TransformType.BAKE_STANDARDIZED_CONV,
                extra_jax_keys={
                    "scale": f"{jax_prefix}/conv_block_1/standardized_conv1_d/scale",
                    "bias": f"{jax_prefix}/conv_block_1/standardized_conv1_d/bias",
                },
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_block_1/rms_batch_norm/scale"],
                torch_key=f"{torch_prefix}.conv_out.net.0.gamma",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_block_1/rms_batch_norm/offset"],
                torch_key=f"{torch_prefix}.conv_out.net.0.beta",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

    # ========================================================================
    # Transformer Tower
    # ========================================================================
    # The JAX model has a flattened structure with separate blocks:
    # - mha_block, mha_block_1, ..., mha_block_8
    # - mlp_block, mlp_block_1, ..., mlp_block_8
    # - attention_bias_block, attention_bias_block_1, ..., attention_bias_block_8
    # - pair_update_block, pair_update_block_1, ..., pair_update_block_4

    jax_tower = "alphagenome/transformer_tower"

    for layer_idx in range(9):
        suffix = _jax_block_suffix(layer_idx)
        torch_layer_prefix = f"transformer_unet.transformer.layers.{layer_idx}"

        # --- MHA Block ---
        jax_mha = f"{jax_tower}/mha_block{suffix}"

        # QKV projection (concatenated in PyTorch)
        mappings.append(
            ParamMapping(
                jax_keys=[
                    f"{jax_mha}/q_layer/w",
                    f"{jax_mha}/k_layer/w",
                    f"{jax_mha}/v_layer/w",
                ],
                torch_key=f"{torch_layer_prefix}.0.block.to_qkv.weight",
                transform=TransformType.CONCAT_QKV,
            )
        )

        # Q/K/V LayerNorms
        for norm_name in ["q", "k", "v"]:
            mappings.extend([
                ParamMapping(
                    jax_keys=[f"{jax_mha}/norm_{norm_name}/scale"],
                    torch_key=f"{torch_layer_prefix}.0.block.{norm_name}_norm.weight",
                ),
                ParamMapping(
                    jax_keys=[f"{jax_mha}/norm_{norm_name}/offset"],
                    torch_key=f"{torch_layer_prefix}.0.block.{norm_name}_norm.bias",
                ),
            ])

        # Output projection
        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_mha}/linear_embedding/w"],
                torch_key=f"{torch_layer_prefix}.0.block.to_out.weight",
                transform=TransformType.TRANSPOSE_LINEAR,
            ),
            ParamMapping(
                jax_keys=[f"{jax_mha}/linear_embedding/b"],
                torch_key=f"{torch_layer_prefix}.0.block.to_out.bias",
            ),
        ])

        # Pre/post RMSNorm for attention wrapper
        mappings.extend([
            # Pre-norm
            ParamMapping(
                jax_keys=[f"{jax_mha}/rms_batch_norm/scale"],
                torch_key=f"{torch_layer_prefix}.0.pre_rmsnorm.gamma",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_mha}/rms_batch_norm/offset"],
                torch_key=f"{torch_layer_prefix}.0.pre_rmsnorm.beta",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            # Post-norm
            ParamMapping(
                jax_keys=[f"{jax_mha}/rms_batch_norm_1/scale"],
                torch_key=f"{torch_layer_prefix}.0.post_rmsnorm.gamma",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_mha}/rms_batch_norm_1/offset"],
                torch_key=f"{torch_layer_prefix}.0.post_rmsnorm.beta",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

        # --- Attention Bias Block ---
        jax_attn_bias = f"{jax_tower}/attention_bias_block{suffix}"

        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_attn_bias}/linear/w"],
                torch_key=f"{torch_layer_prefix}.0.block.to_attn_bias.2.weight",
                transform=TransformType.TRANSPOSE_LINEAR,
            ),
            ParamMapping(
                jax_keys=[f"{jax_attn_bias}/rms_batch_norm/scale"],
                torch_key=f"{torch_layer_prefix}.0.block.to_attn_bias.0.gamma",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_attn_bias}/rms_batch_norm/offset"],
                torch_key=f"{torch_layer_prefix}.0.block.to_attn_bias.0.beta",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

        # --- MLP Block ---
        jax_mlp = f"{jax_tower}/mlp_block{suffix}"

        # FF in projection
        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_mlp}/linear/w"],
                torch_key=f"{torch_layer_prefix}.1.block.0.weight",
                transform=TransformType.TRANSPOSE_LINEAR,
            ),
            ParamMapping(
                jax_keys=[f"{jax_mlp}/linear/b"],
                torch_key=f"{torch_layer_prefix}.1.block.0.bias",
            ),
        ])

        # FF out projection
        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_mlp}/linear_1/w"],
                torch_key=f"{torch_layer_prefix}.1.block.3.weight",
                transform=TransformType.TRANSPOSE_LINEAR,
            ),
            ParamMapping(
                jax_keys=[f"{jax_mlp}/linear_1/b"],
                torch_key=f"{torch_layer_prefix}.1.block.3.bias",
            ),
        ])

        # Pre/post RMSNorm for FF wrapper
        mappings.extend([
            # Pre-norm
            ParamMapping(
                jax_keys=[f"{jax_mlp}/rms_batch_norm/scale"],
                torch_key=f"{torch_layer_prefix}.1.pre_rmsnorm.gamma",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_mlp}/rms_batch_norm/offset"],
                torch_key=f"{torch_layer_prefix}.1.pre_rmsnorm.beta",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            # Post-norm
            ParamMapping(
                jax_keys=[f"{jax_mlp}/rms_batch_norm_1/scale"],
                torch_key=f"{torch_layer_prefix}.1.post_rmsnorm.gamma",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_mlp}/rms_batch_norm_1/offset"],
                torch_key=f"{torch_layer_prefix}.1.post_rmsnorm.beta",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

        # --- Pairwise blocks (every 2 layers: 0, 2, 4, 6, 8) ---
        if layer_idx % 2 == 0:
            pairwise_idx = layer_idx // 2
            pairwise_suffix = _jax_block_suffix(pairwise_idx)
            jax_pair = f"{jax_tower}/pair_update_block{pairwise_suffix}"

            # Sequence-to-pair block
            jax_seq2pair = f"{jax_pair}/sequence_to_pair_block"

            # QK projection (concatenated)
            mappings.append(
                ParamMapping(
                    jax_keys=[
                        f"{jax_seq2pair}/linear_q/w",
                        f"{jax_seq2pair}/linear_k/w",
                    ],
                    torch_key=f"{torch_layer_prefix}.2.to_qk.weight",
                    transform=TransformType.CONCAT_QK,
                )
            )

            # QK to pairwise projection
            mappings.extend([
                ParamMapping(
                    jax_keys=[f"{jax_seq2pair}/linear_pair/w"],
                    torch_key=f"{torch_layer_prefix}.2.qk_to_pairwise.weight",
                    transform=TransformType.TRANSPOSE_LINEAR,
                ),
                ParamMapping(
                    jax_keys=[f"{jax_seq2pair}/linear_pair/b"],
                    torch_key=f"{torch_layer_prefix}.2.qk_to_pairwise.bias",
                ),
            ])

            # Outer sum projection (Y_Q + Y_K concatenated)
            mappings.append(
                ParamMapping(
                    jax_keys=[
                        f"{jax_seq2pair}/linear_y_q/w",
                        f"{jax_seq2pair}/linear_y_k/w",
                    ],
                    torch_key=f"{torch_layer_prefix}.2.to_outer_sum.1.weight",
                    transform=TransformType.CONCAT_QK,
                )
            )

            # Relative position encoding projection
            mappings.extend([
                ParamMapping(
                    jax_keys=[f"{jax_seq2pair}/linear_pos_features/w"],
                    torch_key=f"{torch_layer_prefix}.2.to_rel_pos_encoding.weight",
                    transform=TransformType.TRANSPOSE_LINEAR,
                ),
                ParamMapping(
                    jax_keys=[f"{jax_seq2pair}/linear_pos_features/b"],
                    torch_key=f"{torch_layer_prefix}.2.to_rel_pos_encoding.bias",
                ),
            ])

            # QK relative position bias
            mappings.append(
                ParamMapping(
                    jax_keys=[
                        f"{jax_seq2pair}/q_r_bias",
                        f"{jax_seq2pair}/k_r_bias",
                    ],
                    torch_key=f"{torch_layer_prefix}.2.qk_rel_pos_bias",
                    transform=TransformType.STACK_QK_BIAS,
                )
            )

            # Seq2pair pre-norm (RMSNorm, not BatchRMSNorm)
            mappings.extend([
                ParamMapping(
                    jax_keys=[f"{jax_seq2pair}/norm_seq2pair/scale"],
                    torch_key=f"{torch_layer_prefix}.2.norm.weight",
                ),
                ParamMapping(
                    jax_keys=[f"{jax_seq2pair}/norm_seq2pair/offset"],
                    torch_key=f"{torch_layer_prefix}.2.norm.bias",
                ),
            ])

            # --- Pairwise row attention ---
            jax_row_attn = f"{jax_pair}/row_attention_block"

            # QK projection (concatenated)
            mappings.append(
                ParamMapping(
                    jax_keys=[
                        f"{jax_row_attn}/linear_q/w",
                        f"{jax_row_attn}/linear_k/w",
                    ],
                    torch_key=f"{torch_layer_prefix}.3.block.to_qk.weight",
                    transform=TransformType.CONCAT_QK,
                )
            )

            # V projection
            mappings.extend([
                ParamMapping(
                    jax_keys=[f"{jax_row_attn}/linear_v/w"],
                    torch_key=f"{torch_layer_prefix}.3.block.to_v.weight",
                    transform=TransformType.TRANSPOSE_LINEAR,
                ),
                ParamMapping(
                    jax_keys=[f"{jax_row_attn}/linear_v/b"],
                    torch_key=f"{torch_layer_prefix}.3.block.to_v.bias",
                ),
            ])

            # Row attention pre-norm (LayerNorm)
            mappings.extend([
                ParamMapping(
                    jax_keys=[f"{jax_row_attn}/layer_norm/scale"],
                    torch_key=f"{torch_layer_prefix}.3.pre_rmsnorm.weight",
                ),
                ParamMapping(
                    jax_keys=[f"{jax_row_attn}/layer_norm/offset"],
                    torch_key=f"{torch_layer_prefix}.3.pre_rmsnorm.bias",
                ),
            ])

            # --- Pairwise MLP ---
            jax_pair_mlp = f"{jax_pair}/pair_mlp_block"

            mappings.extend([
                ParamMapping(
                    jax_keys=[f"{jax_pair_mlp}/linear/w"],
                    torch_key=f"{torch_layer_prefix}.4.block.0.weight",
                    transform=TransformType.TRANSPOSE_LINEAR,
                ),
                ParamMapping(
                    jax_keys=[f"{jax_pair_mlp}/linear/b"],
                    torch_key=f"{torch_layer_prefix}.4.block.0.bias",
                ),
                ParamMapping(
                    jax_keys=[f"{jax_pair_mlp}/linear_1/w"],
                    torch_key=f"{torch_layer_prefix}.4.block.3.weight",
                    transform=TransformType.TRANSPOSE_LINEAR,
                ),
                ParamMapping(
                    jax_keys=[f"{jax_pair_mlp}/linear_1/b"],
                    torch_key=f"{torch_layer_prefix}.4.block.3.bias",
                ),
            ])

            # Pair MLP pre-norm (LayerNorm)
            mappings.extend([
                ParamMapping(
                    jax_keys=[f"{jax_pair_mlp}/layer_norm/scale"],
                    torch_key=f"{torch_layer_prefix}.4.pre_rmsnorm.weight",
                ),
                ParamMapping(
                    jax_keys=[f"{jax_pair_mlp}/layer_norm/offset"],
                    torch_key=f"{torch_layer_prefix}.4.pre_rmsnorm.bias",
                ),
            ])

    # ========================================================================
    # UNet Up Blocks (sequence_decoder)
    # ========================================================================
    # JAX: alphagenome/sequence_decoder/up_res_block/... (first block, 1536->1536)
    # JAX: alphagenome/sequence_decoder/up_res_block_{i}/... (i >= 1)
    # Note: JAX uses conv_in/conv_out, not conv_block/conv_block_1
    # JAX has 7 blocks: idx 0-6 (bin_size 64 -> 1)
    # PyTorch has 7 blocks: ups.0-6 (1536 -> 768)
    # Mapping: JAX idx 0-6 -> PyTorch idx 0-6

    for jax_idx in range(0, 7):  # JAX blocks 0-6
        # JAX suffix: empty for first, _1 to _6 for rest
        jax_suffix = _jax_block_suffix(jax_idx)
        jax_prefix = f"alphagenome/sequence_decoder/up_res_block{jax_suffix}"

        # Map JAX idx 0 -> PyTorch idx 0, etc.
        torch_idx = jax_idx
        torch_prefix = f"transformer_unet.ups.{torch_idx}"

        # First conv block (conv_in in JAX)
        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_in/standardized_conv1_d/w"],
                torch_key=f"{torch_prefix}.conv.net.2.weight",
                transform=TransformType.BAKE_STANDARDIZED_CONV,
                extra_jax_keys={
                    "scale": f"{jax_prefix}/conv_in/standardized_conv1_d/scale",
                    "bias": f"{jax_prefix}/conv_in/standardized_conv1_d/bias",
                },
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_in/rms_batch_norm/scale"],
                torch_key=f"{torch_prefix}.conv.net.0.gamma",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_in/rms_batch_norm/offset"],
                torch_key=f"{torch_prefix}.conv.net.0.beta",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

        # Residual scale (JAX scalar [] -> PyTorch [1])
        mappings.append(
            ParamMapping(
                jax_keys=[f"{jax_prefix}/residual_scale"],
                torch_key=f"{torch_prefix}.residual_scale",
                transform=TransformType.UNSQUEEZE_SCALAR,
            )
        )

        # Skip connection conv (pointwise_conv_unet_skip in JAX)
        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_prefix}/pointwise_conv_unet_skip/linear/w"],
                torch_key=f"{torch_prefix}.unet_conv.net.2.weight",
                transform=TransformType.TRANSPOSE_LINEAR_TO_CONV1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/pointwise_conv_unet_skip/linear/b"],
                torch_key=f"{torch_prefix}.unet_conv.net.2.bias",
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/pointwise_conv_unet_skip/rms_batch_norm/scale"],
                torch_key=f"{torch_prefix}.unet_conv.net.0.gamma",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/pointwise_conv_unet_skip/rms_batch_norm/offset"],
                torch_key=f"{torch_prefix}.unet_conv.net.0.beta",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

        # Output conv block (conv_out in JAX)
        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_out/standardized_conv1_d/w"],
                torch_key=f"{torch_prefix}.conv_out.net.2.weight",
                transform=TransformType.BAKE_STANDARDIZED_CONV,
                extra_jax_keys={
                    "scale": f"{jax_prefix}/conv_out/standardized_conv1_d/scale",
                    "bias": f"{jax_prefix}/conv_out/standardized_conv1_d/bias",
                },
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_out/rms_batch_norm/scale"],
                torch_key=f"{torch_prefix}.conv_out.net.0.gamma",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_out/rms_batch_norm/offset"],
                torch_key=f"{torch_prefix}.conv_out.net.0.beta",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

    # ========================================================================
    # Organism Embedding
    # ========================================================================
    mappings.append(
        ParamMapping(
            jax_keys=["alphagenome/embed/embeddings"],
            torch_key="organism_embed.embed.weight",
        )
    )

    # ========================================================================
    # Output Embeddings
    # ========================================================================
    # 128bp output embedding (output_embedder in JAX)
    jax_out_128 = "alphagenome/output_embedder"
    mappings.extend([
        ParamMapping(
            jax_keys=[f"{jax_out_128}/linear/w"],
            torch_key="outembed_128bp.double_features.weight",
            transform=TransformType.TRANSPOSE_LINEAR,
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_128}/linear/b"],
            torch_key="outembed_128bp.double_features.bias",
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_128}/rms_batch_norm/scale"],
            torch_key="outembed_128bp.norm.gamma",
            transform=TransformType.SQUEEZE_TO_1D,
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_128}/rms_batch_norm/offset"],
            torch_key="outembed_128bp.norm.beta",
            transform=TransformType.SQUEEZE_TO_1D,
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_128}/embed/embeddings"],
            torch_key="outembed_128bp.embed.weight",
        ),
    ])

    # 1bp output embedding (output_embedder_1 in JAX)
    jax_out_1 = "alphagenome/output_embedder_1"
    mappings.extend([
        ParamMapping(
            jax_keys=[f"{jax_out_1}/linear/w"],
            torch_key="outembed_1bp.double_features.weight",
            transform=TransformType.TRANSPOSE_LINEAR,
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_1}/linear/b"],
            torch_key="outembed_1bp.double_features.bias",
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_1}/linear_1/w"],
            torch_key="outembed_1bp.skip_proj.weight",
            transform=TransformType.TRANSPOSE_LINEAR,
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_1}/rms_batch_norm/scale"],
            torch_key="outembed_1bp.norm.gamma",
            transform=TransformType.SQUEEZE_TO_1D,
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_1}/rms_batch_norm/offset"],
            torch_key="outembed_1bp.norm.beta",
            transform=TransformType.SQUEEZE_TO_1D,
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_1}/embed/embeddings"],
            torch_key="outembed_1bp.embed.weight",
        ),
    ])

    # Pair output embedding (output_pair in JAX)
    jax_out_pair = "alphagenome/output_pair"
    mappings.extend([
        ParamMapping(
            jax_keys=[f"{jax_out_pair}/layer_norm/scale"],
            torch_key="outembed_pair.norm.weight",
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_pair}/layer_norm/offset"],
            torch_key="outembed_pair.norm.bias",
        ),
        ParamMapping(
            jax_keys=[f"{jax_out_pair}/embed/embeddings"],
            torch_key="outembed_pair.embed.weight",
        ),
    ])

    # ========================================================================
    # Output Heads (JAX head/*)
    # ========================================================================
    organism_indices = {
        "human": 0,
        "mouse": 1,
    }

    genome_track_heads = {
        "rna_seq": (1, 128),
        "cage": (1, 128),
        "dnase": (1, 128),
        "procap": (1, 128),
        "atac": (1, 128),
        "chip_tf": (128,),
        "chip_histone": (128,),
    }

    for head_name, resolutions in genome_track_heads.items():
        for res in resolutions:
            jax_prefix = f"alphagenome/head/{head_name}/resolution_{res}"
            for organism, idx in organism_indices.items():
                torch_prefix = f"heads.{organism}.{head_name}.resolution_{res}"
                mappings.extend([
                    ParamMapping(
                        jax_keys=[f"{jax_prefix}/multi_organism_linear/w"],
                        torch_key=f"{torch_prefix}.to_pred.weight",
                        transform=TransformType.TRANSPOSE_LINEAR,
                        select_index=idx,
                    ),
                    ParamMapping(
                        jax_keys=[f"{jax_prefix}/multi_organism_linear/b"],
                        torch_key=f"{torch_prefix}.to_pred.bias",
                        select_index=idx,
                    ),
                    ParamMapping(
                        jax_keys=[f"{jax_prefix}/learnt_scale"],
                        torch_key=f"{torch_prefix}.scale",
                        select_index=idx,
                    ),
                ])

    # Contact maps head
    for organism, idx in organism_indices.items():
        mappings.extend([
            ParamMapping(
                jax_keys=["alphagenome/head/contact_maps/multi_organism_linear/w"],
                torch_key=f"heads.{organism}.contact_maps.to_pred.weight",
                transform=TransformType.TRANSPOSE_LINEAR,
                select_index=idx,
            ),
            ParamMapping(
                jax_keys=["alphagenome/head/contact_maps/multi_organism_linear/b"],
                torch_key=f"heads.{organism}.contact_maps.to_pred.bias",
                select_index=idx,
            ),
        ])

    # Splice site heads
    for organism, idx in organism_indices.items():
        mappings.extend([
            ParamMapping(
                jax_keys=["alphagenome/head/splice_sites_classification/multi_organism_linear/w"],
                torch_key=f"heads.{organism}.splice_sites_classification.linear.weight",
                transform=TransformType.TRANSPOSE_LINEAR,
                select_index=idx,
            ),
            ParamMapping(
                jax_keys=["alphagenome/head/splice_sites_classification/multi_organism_linear/b"],
                torch_key=f"heads.{organism}.splice_sites_classification.linear.bias",
                select_index=idx,
            ),
            ParamMapping(
                jax_keys=["alphagenome/head/splice_sites_usage/multi_organism_linear/w"],
                torch_key=f"heads.{organism}.splice_sites_usage.linear.weight",
                transform=TransformType.TRANSPOSE_LINEAR,
                select_index=idx,
            ),
            ParamMapping(
                jax_keys=["alphagenome/head/splice_sites_usage/multi_organism_linear/b"],
                torch_key=f"heads.{organism}.splice_sites_usage.linear.bias",
                select_index=idx,
            ),
            ParamMapping(
                jax_keys=["alphagenome/head/splice_sites_junction/multi_organism_linear/w"],
                torch_key=f"heads.{organism}.splice_sites_junction.project.weight",
                transform=TransformType.TRANSPOSE_LINEAR,
                select_index=idx,
            ),
            ParamMapping(
                jax_keys=["alphagenome/head/splice_sites_junction/multi_organism_linear/b"],
                torch_key=f"heads.{organism}.splice_sites_junction.project.bias",
                select_index=idx,
            ),
            ParamMapping(
                jax_keys=["alphagenome/head/splice_sites_junction/pos_donor_logits/embeddings"],
                torch_key=f"heads.{organism}.splice_sites_junction.pos_donor_embeddings",
                select_index=idx,
            ),
            ParamMapping(
                jax_keys=["alphagenome/head/splice_sites_junction/pos_acceptor_logits/embeddings"],
                torch_key=f"heads.{organism}.splice_sites_junction.pos_acceptor_embeddings",
                select_index=idx,
            ),
            ParamMapping(
                jax_keys=["alphagenome/head/splice_sites_junction/neg_donor_logits/embeddings"],
                torch_key=f"heads.{organism}.splice_sites_junction.neg_donor_embeddings",
                select_index=idx,
            ),
            ParamMapping(
                jax_keys=["alphagenome/head/splice_sites_junction/neg_acceptor_logits/embeddings"],
                torch_key=f"heads.{organism}.splice_sites_junction.neg_acceptor_embeddings",
                select_index=idx,
            ),
        ])

    return mappings


def build_state_mapping() -> list[ParamMapping]:
    """Build the mapping for state (e.g., BatchRMSNorm running variance).

    Returns:
        List of ParamMapping objects for state variables.
    """
    mappings = []

    # DNA embedder BatchRMSNorm running variance
    mappings.append(
        ParamMapping(
            jax_keys=["alphagenome/sequence_encoder/dna_embedder/conv_block/rms_batch_norm/var_ema"],
            torch_key="transformer_unet.dna_embed.pointwise.net.0.running_var",
            transform=TransformType.SQUEEZE_TO_1D,
        )
    )

    # UNet down blocks BatchRMSNorm running variance
    for i in range(6):
        jax_prefix = f"alphagenome/sequence_encoder/downres_block_{i}"
        torch_prefix = f"transformer_unet.downs.{i}"

        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_block/rms_batch_norm/var_ema"],
                torch_key=f"{torch_prefix}.conv.net.0.running_var",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_block_1/rms_batch_norm/var_ema"],
                torch_key=f"{torch_prefix}.conv_out.net.0.running_var",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

    # UNet up blocks BatchRMSNorm running variance
    # JAX idx 0-6 -> PyTorch idx 0-6
    for jax_idx in range(0, 7):
        jax_suffix = _jax_block_suffix(jax_idx)
        jax_prefix = f"alphagenome/sequence_decoder/up_res_block{jax_suffix}"
        torch_idx = jax_idx
        torch_prefix = f"transformer_unet.ups.{torch_idx}"

        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_in/rms_batch_norm/var_ema"],
                torch_key=f"{torch_prefix}.conv.net.0.running_var",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/conv_out/rms_batch_norm/var_ema"],
                torch_key=f"{torch_prefix}.conv_out.net.0.running_var",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_prefix}/pointwise_conv_unet_skip/rms_batch_norm/var_ema"],
                torch_key=f"{torch_prefix}.unet_conv.net.0.running_var",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

    # Transformer tower BatchRMSNorm running variance
    jax_tower = "alphagenome/transformer_tower"

    for layer_idx in range(9):
        suffix = _jax_block_suffix(layer_idx)
        torch_prefix = f"transformer_unet.transformer.layers.{layer_idx}"

        # Attention bias block
        mappings.append(
            ParamMapping(
                jax_keys=[f"{jax_tower}/attention_bias_block{suffix}/rms_batch_norm/var_ema"],
                torch_key=f"{torch_prefix}.0.block.to_attn_bias.0.running_var",
                transform=TransformType.SQUEEZE_TO_1D,
            )
        )

        # MHA block pre/post norms
        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_tower}/mha_block{suffix}/rms_batch_norm/var_ema"],
                torch_key=f"{torch_prefix}.0.pre_rmsnorm.running_var",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_tower}/mha_block{suffix}/rms_batch_norm_1/var_ema"],
                torch_key=f"{torch_prefix}.0.post_rmsnorm.running_var",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

        # MLP block pre/post norms
        mappings.extend([
            ParamMapping(
                jax_keys=[f"{jax_tower}/mlp_block{suffix}/rms_batch_norm/var_ema"],
                torch_key=f"{torch_prefix}.1.pre_rmsnorm.running_var",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
            ParamMapping(
                jax_keys=[f"{jax_tower}/mlp_block{suffix}/rms_batch_norm_1/var_ema"],
                torch_key=f"{torch_prefix}.1.post_rmsnorm.running_var",
                transform=TransformType.SQUEEZE_TO_1D,
            ),
        ])

    # Output embedding BatchRMSNorm running variance
    mappings.extend([
        ParamMapping(
            jax_keys=["alphagenome/output_embedder/rms_batch_norm/var_ema"],
            torch_key="outembed_128bp.norm.running_var",
            transform=TransformType.SQUEEZE_TO_1D,
        ),
        ParamMapping(
            jax_keys=["alphagenome/output_embedder_1/rms_batch_norm/var_ema"],
            torch_key="outembed_1bp.norm.running_var",
            transform=TransformType.SQUEEZE_TO_1D,
        ),
    ])

    return mappings


def get_full_mapping() -> tuple[list[ParamMapping], list[ParamMapping]]:
    """Get the complete mapping specification.

    Returns:
        Tuple of (param_mappings, state_mappings)
    """
    return build_trunk_mapping(), build_state_mapping()
