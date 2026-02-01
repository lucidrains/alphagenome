#!/usr/bin/env python3
"""Convert JAX/Haiku AlphaGenome checkpoint to PyTorch state_dict.

Usage:
    python -m alphagenome_pytorch.convert.convert_checkpoint \
        --output converted_checkpoint.pt

Requires the [convert] optional dependencies:
    pip install -e ".[convert]"

Environment variables:
    HF_TOKEN: HuggingFace access token for gated model access
    HF_HOME: HuggingFace cache directory (default: ~/.cache/huggingface)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def flatten_nested_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    """Flatten a nested dict into a flat dict with path-like keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def jax_to_torch_tensor(arr) -> Tensor:
    """Convert a JAX/numpy array to a PyTorch tensor."""
    import numpy as np
    # Convert to numpy first (handles JAX arrays and numpy arrays)
    if hasattr(arr, 'numpy'):
        np_arr = arr.numpy()
    else:
        np_arr = np.asarray(arr)
    return torch.from_numpy(np_arr.copy())


def load_jax_checkpoint(model_version: str = "all_folds", use_gpu: bool = True):
    """Load JAX checkpoint using official alphagenome_research loader.

    Args:
        model_version: Model version to load (e.g., "all_folds", "fold_0", etc.)
        use_gpu: Whether to use GPU if available (default: True)

    Returns:
        Tuple of (flat_params, flat_state) where each is a flat dict of JAX arrays.
    """
    try:
        import jax
        from alphagenome_research.model.dna_model import create, create_from_huggingface
    except ImportError as e:
        print(f"Error: Missing required dependencies. Install with: pip install -e '.[convert]'")
        print(f"Details: {e}")
        sys.exit(1)

    # Select device
    if use_gpu:
        try:
            device = jax.devices("gpu")[0]
            print(f"Using GPU device: {device}")
        except RuntimeError:
            device = jax.devices("cpu")[0]
            print(f"No GPU available, using CPU device: {device}")
    else:
        device = jax.devices("cpu")[0]
        print(f"Using CPU device: {device}")

    local_model_path = os.environ.get("ALPHAGENOME_MODEL_PATH")
    if local_model_path and Path(local_model_path).exists():
        print(f"Loading checkpoint from local path: {local_model_path}")
        model = create(
            checkpoint_path=local_model_path,
            device=device,
        )
    else:
        print(f"Loading checkpoint (model_version={model_version})...")
        model = create_from_huggingface(
            model_version=model_version,
            device=device,
        )

    # Extract and flatten params and state (note: underscore prefix)
    flat_params = flatten_nested_dict(model._params)
    flat_state = flatten_nested_dict(model._state)

    print(f"  Loaded {len(flat_params)} parameter arrays")
    print(f"  Loaded {len(flat_state)} state arrays")

    return flat_params, flat_state


def convert_checkpoint(
    flat_params: dict[str, Any],
    flat_state: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Tensor]:
    """Convert JAX checkpoint to PyTorch state_dict.

    Args:
        flat_params: Flattened JAX params dict (key -> JAX array)
        flat_state: Flattened JAX state dict (key -> JAX array)
        verbose: Whether to print progress

    Returns:
        PyTorch state_dict ready for model.load_state_dict()
    """
    from .mapping_spec import (
        ParamMapping,
        TransformType,
        apply_transform,
        build_trunk_mapping,
        build_state_mapping,
    )

    state_dict = {}
    used_jax_keys = set()

    # Get all mappings
    param_mappings = build_trunk_mapping()
    state_mappings = build_state_mapping()

    if verbose:
        print(f"\nConverting {len(param_mappings)} parameter mappings...")

    # Process parameter mappings
    for mapping in param_mappings:
        try:
            # Get primary tensors
            tensors = []
            for jax_key in mapping.jax_keys:
                if jax_key not in flat_params:
                    if mapping.optional:
                        if verbose:
                            print(f"  Skipping optional mapping (key not found): {jax_key}")
                        tensors = None
                        break
                    raise KeyError(f"JAX key not found: {jax_key}")
                tensors.append(jax_to_torch_tensor(flat_params[jax_key]))
                used_jax_keys.add(jax_key)

            # Skip if optional mapping was not found
            if tensors is None:
                continue

            # Get extra tensors if needed
            extra_tensors = {}
            for name, jax_key in mapping.extra_jax_keys.items():
                if jax_key not in flat_params:
                    raise KeyError(f"Extra JAX key not found: {jax_key}")
                extra_tensors[name] = jax_to_torch_tensor(flat_params[jax_key])
                used_jax_keys.add(jax_key)

            # Apply transform
            result = apply_transform(
                mapping.transform,
                tensors,
                extra_tensors,
                select_index=mapping.select_index,
            )

            # Handle multiple outputs (e.g., baked conv returns weight and bias)
            if isinstance(result, tuple):
                # For BAKE_STANDARDIZED_CONV, we get (weight, bias)
                # The torch_key should end with .weight, and we add .bias sibling
                assert mapping.torch_key.endswith(".weight"), \
                    f"Expected .weight suffix for multi-output transform, got: {mapping.torch_key}"
                base_key = mapping.torch_key[:-7]  # Remove ".weight"
                state_dict[mapping.torch_key] = result[0]
                state_dict[f"{base_key}.bias"] = result[1]
            else:
                state_dict[mapping.torch_key] = result

        except Exception as e:
            print(f"Error processing mapping {mapping.torch_key}: {e}")
            raise

    if verbose:
        print(f"Converting {len(state_mappings)} state mappings...")

    # Process state mappings
    for mapping in state_mappings:
        try:
            tensors = []
            for jax_key in mapping.jax_keys:
                if jax_key not in flat_state:
                    if verbose:
                        print(f"  Warning: State key not found: {jax_key}")
                    continue
                tensors.append(jax_to_torch_tensor(flat_state[jax_key]))
                used_jax_keys.add(jax_key)

            if tensors:
                result = apply_transform(
                    mapping.transform,
                    tensors,
                    select_index=mapping.select_index,
                )
                state_dict[mapping.torch_key] = result

        except Exception as e:
            print(f"Error processing state mapping {mapping.torch_key}: {e}")
            raise

    if verbose:
        # Report unused JAX keys
        all_jax_keys = set(flat_params.keys()) | set(flat_state.keys())
        unused_keys = all_jax_keys - used_jax_keys
        if unused_keys:
            print(f"\nWarning: {len(unused_keys)} JAX keys not mapped:")
            for key in sorted(unused_keys)[:20]:
                print(f"  - {key}")
            if len(unused_keys) > 20:
                print(f"  ... and {len(unused_keys) - 20} more")

        print(f"\nConverted {len(state_dict)} tensors to PyTorch state_dict")

    return state_dict


def verify_state_dict(
    state_dict: dict[str, Tensor],
    strict: bool = True,
    verbose: bool = True,
) -> bool:
    """Verify that the state_dict can be loaded into a PyTorch model.

    Args:
        state_dict: Converted state_dict
        strict: Whether to use strict=True when loading
        verbose: Whether to print progress

    Returns:
        True if verification passes
    """
    from alphagenome_pytorch import AlphaGenome

    if verbose:
        print("\nVerifying state_dict compatibility...")

    # Create model with default config
    model = AlphaGenome()

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if verbose:
            if missing:
                print(f"  Missing keys ({len(missing)}):")
                for key in missing[:20]:
                    print(f"    - {key}")
                if len(missing) > 20:
                    print(f"    ... and {len(missing) - 20} more")

            if unexpected:
                print(f"  Unexpected keys ({len(unexpected)}):")
                for key in unexpected[:20]:
                    print(f"    - {key}")
                if len(unexpected) > 20:
                    print(f"    ... and {len(unexpected) - 20} more")

        if strict and (missing or unexpected):
            print("\nVerification FAILED: strict loading would fail")
            return False

        if verbose:
            print("\nVerification PASSED")

        return True

    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert JAX/Haiku AlphaGenome checkpoint to PyTorch"
    )
    parser.add_argument(
        "--model-version",
        default="all_folds",
        help="Model version to load (e.g., 'all_folds', 'fold_0')",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU device (default: use GPU if available)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("alphagenome_pytorch.pt"),
        help="Output path for converted checkpoint",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification step",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Reduce output verbosity",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    # Load JAX checkpoint
    flat_params, flat_state = load_jax_checkpoint(args.model_version, use_gpu=not args.cpu)

    # Convert to PyTorch state_dict
    state_dict = convert_checkpoint(flat_params, flat_state, verbose=verbose)

    # Verify the conversion
    if not args.skip_verify:
        if not verify_state_dict(state_dict, strict=True, verbose=verbose):
            print("\nConversion completed but verification failed.")
            print("The state_dict may still be usable with strict=False.")

    # Save the state_dict
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output)
    print(f"\nSaved converted checkpoint to: {args.output}")

    # Print usage instructions
    print("\nUsage:")
    print(f"  import torch")
    print(f"  from alphagenome_pytorch import AlphaGenome")
    print(f"  ")
    print(f"  model = AlphaGenome()")
    print(f"  state_dict = torch.load('{args.output}')")
    print(f"  model.load_state_dict(state_dict)")


if __name__ == "__main__":
    main()
