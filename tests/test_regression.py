"""Regression test using saved JAX reference tensors.

This test compares PyTorch model outputs against pre-computed JAX reference outputs.
The reference tensors are stored in tests/regression_data/ and were generated using
the official JAX AlphaGenome model.

Usage:
    ALPHAGENOME_RUN_INTEGRATION_TESTS=1 pytest tests/test_regression.py -v

To regenerate reference tensors:
    python tests/generate_regression_tensors.py
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Tolerance for numerical comparisons (accounts for mixed-precision differences)
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-2
MIN_PERCENT_WITHIN_TOLERANCE = 95.0

# Required regression data files
REQUIRED_FILES = [
    "input_sequence.npy",
    "input_organism.npy",
    "output_embeds_1bp.npy",
    "output_embeds_128bp.npy",
    "output_embeds_pair.npy",
    "output_rna_seq_1bp.npy",
    "output_rna_seq_128bp.npy",
    "output_contact_maps.npy",
]

# Skip unless integration tests are enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("ALPHAGENOME_RUN_INTEGRATION_TESTS", "0") != "1",
    reason="Integration tests disabled. Set ALPHAGENOME_RUN_INTEGRATION_TESTS=1 to enable.",
)


# =============================================================================
# Environment Setup
# =============================================================================

def _load_dotenv() -> None:
    """Load .env file if it exists."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


# =============================================================================
# Tolerance Comparison Utilities
# =============================================================================

@dataclass
class ToleranceResult:
    """Result of a tolerance comparison."""

    percent_within: float
    max_diff: float
    mean_diff: float
    passed: bool

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{status}: {self.percent_within:.2f}% within tolerance, "
            f"max_diff={self.max_diff:.6f}, mean_diff={self.mean_diff:.6f}"
        )


def compute_tolerance_stats(
    actual: np.ndarray,
    expected: np.ndarray,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    min_percent: float = MIN_PERCENT_WITHIN_TOLERANCE,
) -> ToleranceResult:
    """Compare arrays and compute tolerance statistics.

    Uses numpy allclose formula: |actual - expected| <= atol + rtol * |expected|
    """
    diff = np.abs(actual - expected)
    within_tol = diff <= (atol + rtol * np.abs(expected))
    pct_within = np.mean(within_tol) * 100

    return ToleranceResult(
        percent_within=pct_within,
        max_diff=float(diff.max()),
        mean_diff=float(diff.mean()),
        passed=pct_within >= min_percent,
    )


def assert_within_tolerance(
    actual: np.ndarray,
    expected: np.ndarray,
    output_name: str,
) -> None:
    """Assert that actual and expected arrays match within tolerance."""
    result = compute_tolerance_stats(actual, expected)

    print(f"\n{output_name}: {result}")

    assert result.passed, (
        f"{output_name}: Only {result.percent_within:.2f}% within tolerance "
        f"(expected >= {MIN_PERCENT_WITHIN_TOLERANCE}%). "
        f"Max diff: {result.max_diff:.6f}, Mean diff: {result.mean_diff:.6f}"
    )


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def regression_data() -> dict[str, Any]:
    """Load pre-computed regression test data."""
    data_dir = Path(__file__).parent / "regression_data"

    if not data_dir.exists():
        pytest.skip(
            f"Regression data not found at {data_dir}. "
            "Run 'python tests/generate_regression_tensors.py' to generate."
        )

    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        pytest.skip(
            f"Missing regression files: {missing}. "
            "Run 'python tests/generate_regression_tensors.py' to generate."
        )

    def load(name: str) -> np.ndarray:
        return np.load(data_dir / name)

    return {
        # Inputs
        "sequence": load("input_sequence.npy"),
        "organism": load("input_organism.npy")[0],
        # Embeddings (JAX reference)
        "embeds_1bp": load("output_embeds_1bp.npy"),
        "embeds_128bp": load("output_embeds_128bp.npy"),
        "embeds_pair": load("output_embeds_pair.npy"),
        # Track predictions (JAX reference)
        "rna_seq_1bp": load("output_rna_seq_1bp.npy"),
        "rna_seq_128bp": load("output_rna_seq_128bp.npy"),
        "contact_maps": load("output_contact_maps.npy"),
    }


@pytest.fixture(scope="module")
def torch_model():
    """Load PyTorch model with pretrained weights and reference heads."""
    import jax
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.alphagenome import set_update_running_var
    from alphagenome_pytorch.convert.convert_checkpoint import (
        convert_checkpoint,
        flatten_nested_dict,
    )
    from alphagenome_research.model.dna_model import create_from_huggingface

    print("\nLoading JAX model and converting weights...")
    
    # Try to get GPU, fallback to CPU
    try:
        device = jax.devices("gpu")[0]
    except (RuntimeError, ValueError):
        try:
            device = jax.devices("tpu")[0]
        except (RuntimeError, ValueError):
            device = jax.devices("cpu")[0]
    
    print(f"JAX using device: {device}")
    
    jax_model = create_from_huggingface("all_folds", device=device)
    flat_params = flatten_nested_dict(jax_model._params)
    flat_state = flatten_nested_dict(jax_model._state)
    state_dict = convert_checkpoint(flat_params, flat_state, verbose=False)

    model = AlphaGenome()
    model.add_reference_heads("human")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    set_update_running_var(model, False)

    return model


@pytest.fixture(scope="module")
def model_outputs(regression_data: dict, torch_model) -> dict[str, np.ndarray]:
    """Run model once and cache all outputs for reuse across tests."""
    seq = torch.tensor(regression_data["sequence"])
    organism_index = torch.tensor([regression_data["organism"]], dtype=torch.long)

    with torch.no_grad():
        # Get embeddings
        embeds = torch_model(seq, organism_index, return_embeds=True)

        # Get track predictions
        tracks = torch_model(seq, organism_index)

    return {
        # Embeddings
        "embeds_1bp": embeds.embeds_1bp.numpy(),
        "embeds_128bp": embeds.embeds_128bp.numpy(),
        "embeds_pair": embeds.embeds_pair.numpy(),
        # Track predictions
        "rna_seq_1bp": tracks["human"]["rna_seq"]["scaled_predictions_1bp"].numpy(),
        "rna_seq_128bp": tracks["human"]["rna_seq"]["scaled_predictions_128bp"].numpy(),
        "contact_maps": tracks["human"]["contact_maps"].numpy(),
    }


# =============================================================================
# Test Cases
# =============================================================================

# Define test cases: (output_key, description)
EMBEDDING_TESTS = [
    ("embeds_1bp", "1bp resolution embeddings"),
    ("embeds_128bp", "128bp resolution embeddings"),
    ("embeds_pair", "pairwise embeddings"),
]

TRACK_PREDICTION_TESTS = [
    ("rna_seq_1bp", "RNA-seq 1bp predictions"),
    ("rna_seq_128bp", "RNA-seq 128bp predictions"),
    ("contact_maps", "contact map predictions"),
]


# =============================================================================
# Tests
# =============================================================================

class TestEmbeddingRegression:
    """Test embedding outputs match JAX reference within tolerance."""

    @pytest.mark.parametrize("output_key,description", EMBEDDING_TESTS)
    def test_embedding(
        self,
        regression_data: dict,
        model_outputs: dict,
        output_key: str,
        description: str,
    ):
        """Test {description} matches JAX reference."""
        assert_within_tolerance(
            actual=model_outputs[output_key],
            expected=regression_data[output_key],
            output_name=output_key,
        )


class TestTrackPredictionRegression:
    """Test track prediction outputs match JAX reference within tolerance."""

    @pytest.mark.parametrize("output_key,description", TRACK_PREDICTION_TESTS)
    def test_track_prediction(
        self,
        regression_data: dict,
        model_outputs: dict,
        output_key: str,
        description: str,
    ):
        """Test {description} matches JAX reference."""
        assert_within_tolerance(
            actual=model_outputs[output_key],
            expected=regression_data[output_key],
            output_name=output_key,
        )
