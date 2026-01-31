#!/usr/bin/env python3
"""Inspect PyTorch AlphaGenome model: print all state_dict keys and shapes.

Usage:
    python -m alphagenome_pytorch.convert.inspect_torch [--output keys.json]

This creates a default AlphaGenome model and prints its state_dict structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from alphagenome_pytorch import AlphaGenome


def get_shape_and_dtype(tensor: torch.Tensor) -> dict[str, Any]:
    """Extract shape and dtype from a PyTorch tensor."""
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def inspect_model(model: torch.nn.Module) -> dict[str, Any]:
    """Generate a mapping of all state_dict keys with their shapes."""
    result = {
        "parameters": {},
        "buffers": {},
    }

    state_dict = model.state_dict()

    # Separate parameters and buffers
    param_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}

    for key, tensor in state_dict.items():
        info = get_shape_and_dtype(tensor)
        if key in param_names:
            result["parameters"][key] = info
        elif key in buffer_names:
            result["buffers"][key] = info
        else:
            # Some state_dict keys may not be directly in named_parameters/buffers
            # (e.g., parametrized weights)
            result["parameters"][key] = info

    return result


def print_summary(mapping: dict[str, Any]) -> None:
    """Print a human-readable summary of the model state."""
    print("\n" + "=" * 80)
    print("PARAMETERS")
    print("=" * 80)
    for key, info in sorted(mapping["parameters"].items()):
        shape_str = "x".join(str(s) for s in info["shape"])
        print(f"  {key}: [{shape_str}] ({info['dtype']})")

    print("\n" + "=" * 80)
    print("BUFFERS")
    print("=" * 80)
    for key, info in sorted(mapping["buffers"].items()):
        shape_str = "x".join(str(s) for s in info["shape"])
        print(f"  {key}: [{shape_str}] ({info['dtype']})")

    # Summary stats
    n_params = len(mapping["parameters"])
    n_buffers = len(mapping["buffers"])

    # Calculate total parameter count
    total_params = sum(
        int(torch.prod(torch.tensor(info["shape"])).item())
        for info in mapping["parameters"].values()
    )

    print("\n" + "=" * 80)
    print(f"Total: {n_params} parameter tensors, {n_buffers} buffer tensors")
    print(f"Total parameter count: {total_params:,}")
    print("=" * 80)


def create_default_model() -> AlphaGenome:
    """Create a default AlphaGenome model with standard configuration."""
    return AlphaGenome()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PyTorch AlphaGenome model and print all state_dict keys/shapes"
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
    parser.add_argument(
        "--with-heads",
        action="store_true",
        help="Include prediction heads in the model (human config)",
    )
    args = parser.parse_args()

    # Create model
    print("Creating AlphaGenome model...")
    model = create_default_model()

    if args.with_heads:
        # Add human heads using publication config
        from alphagenome_pytorch import publication_heads_config
        human_config = publication_heads_config["human"]
        model.add_heads(
            organism="human",
            num_tracks_1bp=human_config["num_tracks_1bp"],
            num_tracks_128bp=human_config["num_tracks_128bp"],
            num_tracks_contacts=human_config["num_tracks_contacts"],
            num_splicing_contexts=human_config["num_splicing_contexts"],
            hidden_dim_splice_juncs=human_config["hidden_dim_splice_juncs"],
        )

    # Inspect state_dict
    mapping = inspect_model(model)

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
