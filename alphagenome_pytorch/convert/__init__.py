"""JAX/Haiku -> PyTorch checkpoint conversion utilities.

This package is intentionally lightweight: heavy deps (jax/orbax/haiku) should be
imported lazily by the conversion entrypoints, not at import-time.

Usage:
    # Shape transforms (no JAX deps required)
    from alphagenome_pytorch.convert import transpose_linear_weight

    # Full conversion (requires [convert] deps)
    python -m alphagenome_pytorch.convert.convert_checkpoint --output model.pt

    # Inspect JAX checkpoint (requires [convert] deps)
    python -m alphagenome_pytorch.convert.inspect_jax

    # Inspect PyTorch model
    python -m alphagenome_pytorch.convert.inspect_torch
"""

from .shape_transforms import (
    transpose_linear_weight,
    transpose_conv1d_weight,
    concat_linear_weights,
    bake_standardized_conv1d_weight,
)

__all__ = [
    # Shape transforms (always available)
    "transpose_linear_weight",
    "transpose_conv1d_weight",
    "concat_linear_weights",
    "bake_standardized_conv1d_weight",
    # Submodules (import lazily to avoid JAX deps)
    # - mapping_spec: key mapping definitions
    # - convert_checkpoint: main conversion script
    # - inspect_jax: inspect JAX checkpoint
    # - inspect_torch: inspect PyTorch model
]

