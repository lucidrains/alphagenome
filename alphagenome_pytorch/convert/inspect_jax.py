#!/usr/bin/env python3
"""Inspect JAX/Haiku checkpoint: load and print all param/state keys and shapes.

Usage:
    python -m alphagenome_pytorch.convert.inspect_jax [--output keys.json]

Requires the [convert] optional dependencies:
    pip install -e ".[convert]"

Environment variables:
    HF_TOKEN: HuggingFace access token for gated model access
    HF_HOME: HuggingFace cache directory (default: ~/.cache/huggingface)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    """Flatten a nested dict into a flat dict with path-like keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_shape_and_dtype(arr) -> dict[str, Any]:
    """Extract shape and dtype from a JAX/numpy array."""
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def load_jax_checkpoint(model_version: str = "all_folds", use_gpu: bool = True):
    """Load JAX checkpoint using official alphagenome_research loader.

    Args:
        model_version: Model version to load (e.g., "all_folds", "fold_0", etc.)
        use_gpu: Whether to use GPU if available (default: True)

    Returns:
        Tuple of (params, state) where each is a nested dict of JAX arrays.
    """
    # Lazy import to avoid requiring JAX deps at module load time
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

    # Extract params and state from the model (note: underscore prefix)
    params = model._params
    state = model._state

    return params, state


def inspect_checkpoint(params, state) -> dict[str, Any]:
    """Generate a mapping of all parameter and state keys with their shapes."""
    result = {
        "params": {},
        "state": {},
    }

    # Flatten and extract shapes for params
    flat_params = flatten_dict(params)
    for key, arr in flat_params.items():
        result["params"][key] = get_shape_and_dtype(arr)

    # Flatten and extract shapes for state
    flat_state = flatten_dict(state)
    for key, arr in flat_state.items():
        result["state"][key] = get_shape_and_dtype(arr)

    return result


def print_summary(mapping: dict[str, Any]) -> None:
    """Print a human-readable summary of the checkpoint."""
    print("\n" + "=" * 80)
    print("PARAMETERS")
    print("=" * 80)
    for key, info in sorted(mapping["params"].items()):
        shape_str = "x".join(str(s) for s in info["shape"])
        print(f"  {key}: [{shape_str}] ({info['dtype']})")

    print("\n" + "=" * 80)
    print("STATE")
    print("=" * 80)
    for key, info in sorted(mapping["state"].items()):
        shape_str = "x".join(str(s) for s in info["shape"])
        print(f"  {key}: [{shape_str}] ({info['dtype']})")

    # Summary stats
    n_params = len(mapping["params"])
    n_state = len(mapping["state"])
    print("\n" + "=" * 80)
    print(f"Total: {n_params} parameter arrays, {n_state} state arrays")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect JAX/Haiku checkpoint and print all keys/shapes"
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
        help="Output JSON file for key mapping (optional)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output JSON, skip human-readable summary",
    )
    args = parser.parse_args()

    # Load checkpoint
    params, state = load_jax_checkpoint(args.model_version, use_gpu=not args.cpu)

    # Inspect and generate mapping
    mapping = inspect_checkpoint(params, state)

    # Print summary
    if not args.quiet:
        print_summary(mapping)

    # Save to JSON if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(mapping, f, indent=2)
        print(f"\nKey mapping saved to: {args.output}")


if __name__ == "__main__":
    main()
