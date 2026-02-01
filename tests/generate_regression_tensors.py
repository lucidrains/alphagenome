"""Generate input/output tensor pairs for regression testing.

This script runs the JAX AlphaGenome model and saves input/output pairs
that can be used to verify PyTorch implementation correctness.

Usage:
    python tests/generate_regression_tensors.py

Requirements:
    - alphagenome_research package installed
    - HuggingFace access to google/alphagenome-all-folds (or cached model)

Environment variables (can be set in .env file):
    - HF_TOKEN: HuggingFace API token
    - HF_HOME: Path to HuggingFace cache directory (optional)

Generated Files:
    - input_sequence.npy: Random DNA sequence indices (seed=42)
    - input_organism.npy: Organism index (0=human)
    - output_embeds_*.npy: Embedding outputs from trunk
    - output_rna_seq_*.npy: RNA-seq track predictions
    - output_contact_maps.npy: Contact map predictions
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

# =============================================================================
# Constants
# =============================================================================

RANDOM_SEED = 42
SEQUENCE_LENGTH = 2048  # Minimum sequence length for the model
NUM_NUCLEOTIDES = 4     # A=0, C=1, G=2, T=3
ORGANISM_HUMAN = 0
DEFAULT_OUTPUT_DIR = "tests/regression_data"


# =============================================================================
# Environment Setup
# =============================================================================

def load_dotenv() -> None:
    """Load environment variables from .env file if it exists."""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if not env_file.exists():
        return

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


load_dotenv()


# =============================================================================
# JAX Model Utilities
# =============================================================================

def build_jax_apply_fn(
    jax_model: Any,
    output_type: Literal["embeddings", "predictions"],
) -> Callable:
    """Build a JAX apply function for the AlphaGenome model.

    Args:
        jax_model: Loaded JAX AlphaGenome model instance.
        output_type: What to return - "embeddings" or "predictions".

    Returns:
        Callable: apply_fn(params, state, sequence, organism) -> output
    """
    import haiku as hk
    import jmp
    from alphagenome_research.model import dna_model as dna_model_lib
    from alphagenome_research.model import model as model_lib

    metadata = jax_model._metadata
    model_settings = dna_model_lib.ModelSettings()
    jmp_policy = jmp.get_policy("params=float32,compute=float32,output=float32")

    @hk.transform_with_state
    def _forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(model_lib.AlphaGenome, jmp_policy):
            return model_lib.AlphaGenome(
                metadata,
                num_splice_sites=model_settings.num_splice_sites,
                splice_site_threshold=model_settings.splice_site_threshold,
            )(dna_sequence, organism_index)

    def _apply_fn(params, state, dna_sequence, organism_index):
        (preds, embeddings), _ = _forward.apply(
            params, state, None, dna_sequence, organism_index
        )
        return embeddings if output_type == "embeddings" else preds

    return _apply_fn


# =============================================================================
# Validation & Saving
# =============================================================================

def validate_array(arr: np.ndarray, name: str) -> None:
    """Validate that an array contains finite values (no NaN/Inf)."""
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        raise ValueError(
            f"Generated {name} contains invalid values: {nan_count} NaN, {inf_count} Inf"
        )


def save_array(arr: np.ndarray, path: Path, name: str) -> None:
    """Validate and save a NumPy array."""
    validate_array(arr, name)
    np.save(path, arr)
    print(f"  {path.name}: shape {arr.shape}")


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_regression_tensors(output_dir: str = DEFAULT_OUTPUT_DIR) -> None:
    """Generate and save regression test tensors.

    Args:
        output_dir: Directory to save the tensor files.
    """
    import jax.numpy as jnp
    from alphagenome_research.model.dna_model import create_from_huggingface

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Generate deterministic input
    # -------------------------------------------------------------------------
    print("Generating input data...")
    rng = np.random.default_rng(RANDOM_SEED)
    sequence = rng.integers(0, NUM_NUCLEOTIDES, size=(1, SEQUENCE_LENGTH))
    organism_idx = ORGANISM_HUMAN

    # Save inputs
    print(f"Saving inputs to {output_path}/")
    np.save(output_path / "input_sequence.npy", sequence)
    np.save(output_path / "input_organism.npy", np.array([organism_idx]))
    print(f"  input_sequence.npy: shape {sequence.shape}")
    print(f"  input_organism.npy: value {organism_idx}")

    # -------------------------------------------------------------------------
    # Load JAX model
    # -------------------------------------------------------------------------
    print("\nLoading JAX model from HuggingFace...")
    jax_model = create_from_huggingface("all_folds")

    # Build apply functions (consolidated into single factory)
    embed_apply_fn = build_jax_apply_fn(jax_model, output_type="embeddings")
    heads_apply_fn = build_jax_apply_fn(jax_model, output_type="predictions")

    # Prepare inputs - JAX expects one-hot encoded sequence
    seq_onehot = jnp.eye(NUM_NUCLEOTIDES, dtype=jnp.float32)[sequence]
    organism_jax = jnp.array([organism_idx], dtype=jnp.int32)

    # -------------------------------------------------------------------------
    # Generate embedding outputs
    # -------------------------------------------------------------------------
    print("\nRunning JAX model for embeddings...")
    jax_embeds = embed_apply_fn(
        jax_model._params,
        jax_model._state,
        seq_onehot,
        organism_jax,
    )

    print(f"\nSaving embeddings to {output_path}/")
    save_array(np.asarray(jax_embeds.embeddings_1bp), output_path / "output_embeds_1bp.npy", "embeds_1bp")
    save_array(np.asarray(jax_embeds.embeddings_128bp), output_path / "output_embeds_128bp.npy", "embeds_128bp")
    save_array(np.asarray(jax_embeds.embeddings_pair), output_path / "output_embeds_pair.npy", "embeds_pair")

    # -------------------------------------------------------------------------
    # Generate track prediction outputs
    # -------------------------------------------------------------------------
    print("\nRunning JAX model for track predictions...")
    jax_preds = heads_apply_fn(
        jax_model._params,
        jax_model._state,
        seq_onehot,
        organism_jax,
    )

    print(f"\nSaving track predictions to {output_path}/")
    save_array(np.asarray(jax_preds["rna_seq"]["scaled_predictions_1bp"]), output_path / "output_rna_seq_1bp.npy", "rna_seq_1bp")
    save_array(np.asarray(jax_preds["rna_seq"]["scaled_predictions_128bp"]), output_path / "output_rna_seq_128bp.npy", "rna_seq_128bp")
    save_array(np.asarray(jax_preds["contact_maps"]["predictions"]), output_path / "output_contact_maps.npy", "contact_maps")

    print(f"\nDone! Generated {len(list(output_path.glob('*.npy')))} tensor files.")


if __name__ == "__main__":
    generate_regression_tensors()
