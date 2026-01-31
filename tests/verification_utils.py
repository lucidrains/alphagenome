"""
Verification utilities for JAX to PyTorch conversion testing.

This module provides threshold constants, test level configurations,
test case generators, and statistical comparison functions for validating
the equivalence of JAX and PyTorch model outputs.
"""

from enum import IntEnum
from typing import Any, Dict, List, Tuple, Union

import numpy as np

# Conditional imports for scipy
try:
    from scipy.stats import pearsonr, ttest_rel
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    pearsonr = None
    ttest_rel = None


# =============================================================================
# Threshold Constants (at seq_len=16384)
# =============================================================================

THRESHOLDS = {
    'embeds_1bp': {
        'max_abs_error': 5e-2,
        'mean_abs_error': 5e-3,
        'min_correlation': 0.9999,
        'outlier_fraction': 0.001,
    },
    'embeds_128bp': {
        'max_abs_error': 5e-2,
        'mean_abs_error': 5e-3,
        'min_correlation': 0.9999,
        'outlier_fraction': 0.001,
    },
    'embeds_pair': {
        'max_abs_error': 1e-1,
        'mean_abs_error': 1e-2,
        'min_correlation': 0.999,
        'outlier_fraction': 0.005,
    },
}


# =============================================================================
# Test Level Enum and Configurations
# =============================================================================

class VerificationLevel(IntEnum):
    """Test strictness levels for conversion verification."""
    SANITY = 1   # Shapes match, no NaN/Inf
    LOOSE = 2    # 99% elements within atol=1e-2, rtol=1e-2
    MEDIUM = 3   # 95% elements within atol=1e-3, rtol=1e-3
    STRICT = 4   # 90% elements within atol=1e-4, rtol=1e-4


LEVEL_CONFIGS = {
    VerificationLevel.SANITY: {
        'description': 'Shapes match, no NaN/Inf',
        'fraction_required': None,  # N/A for sanity check
        'atol': None,
        'rtol': None,
    },
    VerificationLevel.LOOSE: {
        'description': '99% elements within tolerance',
        'fraction_required': 0.99,
        'atol': 1e-2,
        'rtol': 1e-2,
    },
    VerificationLevel.MEDIUM: {
        'description': '95% elements within tolerance',
        'fraction_required': 0.95,
        'atol': 1e-3,
        'rtol': 1e-3,
    },
    VerificationLevel.STRICT: {
        'description': '90% elements within tolerance',
        'fraction_required': 0.90,
        'atol': 1e-4,
        'rtol': 1e-4,
    },
}


# =============================================================================
# Test Case Generators
# =============================================================================

# Nucleotide encoding: A=0, C=1, G=2, T=3
NUCLEOTIDE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
IDX_TO_NUCLEOTIDE = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}


def generate_random_sequence(seq_len: int = 16384, seed: int = 42) -> np.ndarray:
    """Generate a random nucleotide sequence as indices (0-3)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=seq_len, dtype=np.int64)


def generate_homopolymer_sequence(seq_len: int = 16384, nucleotide: str = 'A') -> np.ndarray:
    """Generate a sequence of all the same nucleotide."""
    idx = NUCLEOTIDE_TO_IDX[nucleotide.upper()]
    return np.full(seq_len, idx, dtype=np.int64)


def generate_alternating_sequence(
    seq_len: int = 16384,
    pattern: str = 'AT',
) -> np.ndarray:
    """Generate an alternating sequence based on the given pattern."""
    pattern_indices = [NUCLEOTIDE_TO_IDX[n.upper()] for n in pattern]
    pattern_len = len(pattern_indices)
    repeats = (seq_len + pattern_len - 1) // pattern_len
    full_pattern = np.tile(pattern_indices, repeats)
    return full_pattern[:seq_len].astype(np.int64)


def generate_gc_rich_sequence(seq_len: int = 16384, gc_fraction: float = 0.8, seed: int = 42) -> np.ndarray:
    """Generate a GC-rich sequence."""
    rng = np.random.default_rng(seed)
    result = np.zeros(seq_len, dtype=np.int64)
    for i in range(seq_len):
        if rng.random() < gc_fraction:
            # G or C
            result[i] = rng.choice([1, 2])  # C=1, G=2
        else:
            # A or T
            result[i] = rng.choice([0, 3])  # A=0, T=3
    return result


def generate_at_rich_sequence(seq_len: int = 16384, at_fraction: float = 0.8, seed: int = 42) -> np.ndarray:
    """Generate an AT-rich sequence."""
    rng = np.random.default_rng(seed)
    result = np.zeros(seq_len, dtype=np.int64)
    for i in range(seq_len):
        if rng.random() < at_fraction:
            # A or T
            result[i] = rng.choice([0, 3])  # A=0, T=3
        else:
            # G or C
            result[i] = rng.choice([1, 2])  # C=1, G=2
    return result


def generate_half_and_half_sequence(seq_len: int = 16384) -> np.ndarray:
    """Generate a sequence that is half A's and half T's."""
    half = seq_len // 2
    first_half = np.zeros(half, dtype=np.int64)  # A=0
    second_half = np.full(seq_len - half, 3, dtype=np.int64)  # T=3
    return np.concatenate([first_half, second_half])


def get_test_cases(seq_len: int = 16384) -> List[Tuple[str, np.ndarray]]:
    """
    Generate all 13 test cases for verification.

    Returns a list of (name, sequence) tuples.
    """
    test_cases = []

    # Random seeds: 42, 123, 456 (3 cases)
    for seed in [42, 123, 456]:
        test_cases.append((f'random_seed_{seed}', generate_random_sequence(seq_len, seed)))

    # All same nucleotide: A, C, G, T (4 cases)
    for nucleotide in ['A', 'C', 'G', 'T']:
        test_cases.append((f'homopolymer_{nucleotide}', generate_homopolymer_sequence(seq_len, nucleotide)))

    # Alternating: A-T, G-C, ACGT repeat (3 cases)
    test_cases.append(('alternating_AT', generate_alternating_sequence(seq_len, 'AT')))
    test_cases.append(('alternating_GC', generate_alternating_sequence(seq_len, 'GC')))
    test_cases.append(('alternating_ACGT', generate_alternating_sequence(seq_len, 'ACGT')))

    # Special: GC-rich, AT-rich, half-and-half (3 cases)
    test_cases.append(('gc_rich', generate_gc_rich_sequence(seq_len, gc_fraction=0.8, seed=42)))
    test_cases.append(('at_rich', generate_at_rich_sequence(seq_len, at_fraction=0.8, seed=42)))
    test_cases.append(('half_and_half', generate_half_and_half_sequence(seq_len)))

    return test_cases


# =============================================================================
# Statistical Comparison Functions
# =============================================================================

def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert tensor to numpy array, handling JAX, PyTorch, and numpy inputs."""
    if isinstance(tensor, np.ndarray):
        return tensor
    # Check for PyTorch tensor
    if hasattr(tensor, 'detach') and hasattr(tensor, 'cpu'):
        return tensor.detach().cpu().numpy()
    # Check for JAX array
    if hasattr(tensor, 'device') or str(type(tensor).__module__).startswith('jax'):
        return np.asarray(tensor)
    # Fallback
    return np.asarray(tensor)


def compute_statistics(
    jax_output: Any,
    torch_output: Any,
    sigma_threshold: float = 5.0,
) -> Dict[str, float]:
    """
    Compute comparison statistics between JAX and PyTorch outputs.

    Args:
        jax_output: Output tensor from JAX model.
        torch_output: Output tensor from PyTorch model.
        sigma_threshold: Number of standard deviations to consider as outlier.

    Returns:
        Dictionary containing:
            - max_abs_error: Maximum absolute difference
            - mean_abs_error: Mean absolute difference
            - pearson_correlation: Pearson correlation coefficient
            - cosine_similarity: Cosine similarity
            - outlier_fraction: Fraction of elements > sigma_threshold std devs
    """
    jax_arr = _to_numpy(jax_output).flatten().astype(np.float64)
    torch_arr = _to_numpy(torch_output).flatten().astype(np.float64)

    if jax_arr.shape != torch_arr.shape:
        raise ValueError(
            f"Shape mismatch: JAX output has shape {jax_arr.shape}, "
            f"PyTorch output has shape {torch_arr.shape}"
        )

    # Absolute errors
    abs_diff = np.abs(jax_arr - torch_arr)
    max_abs_error = float(np.max(abs_diff))
    mean_abs_error = float(np.mean(abs_diff))

    # Pearson correlation
    if SCIPY_AVAILABLE:
        # Handle constant arrays (correlation undefined)
        if np.std(jax_arr) < 1e-12 or np.std(torch_arr) < 1e-12:
            # If both are constant and equal, correlation is 1
            if np.allclose(jax_arr, torch_arr):
                pearson_corr = 1.0
            else:
                pearson_corr = 0.0
        else:
            pearson_corr, _ = pearsonr(jax_arr, torch_arr)
            pearson_corr = float(pearson_corr)
    else:
        # Fallback using numpy
        if np.std(jax_arr) < 1e-12 or np.std(torch_arr) < 1e-12:
            if np.allclose(jax_arr, torch_arr):
                pearson_corr = 1.0
            else:
                pearson_corr = 0.0
        else:
            pearson_corr = float(np.corrcoef(jax_arr, torch_arr)[0, 1])

    # Cosine similarity
    jax_norm = np.linalg.norm(jax_arr)
    torch_norm = np.linalg.norm(torch_arr)
    if jax_norm < 1e-12 or torch_norm < 1e-12:
        if jax_norm < 1e-12 and torch_norm < 1e-12:
            cosine_sim = 1.0
        else:
            cosine_sim = 0.0
    else:
        cosine_sim = float(np.dot(jax_arr, torch_arr) / (jax_norm * torch_norm))

    # Outlier fraction (elements with |error| > sigma_threshold * std)
    std_diff = np.std(abs_diff)
    if std_diff < 1e-12:
        outlier_fraction = 0.0
    else:
        outlier_mask = abs_diff > (sigma_threshold * std_diff)
        outlier_fraction = float(np.mean(outlier_mask))

    return {
        'max_abs_error': max_abs_error,
        'mean_abs_error': mean_abs_error,
        'pearson_correlation': pearson_corr,
        'cosine_similarity': cosine_sim,
        'outlier_fraction': outlier_fraction,
    }


def check_level(
    stats: Dict[str, float],
    level: VerificationLevel,
    jax_output: Any = None,
    torch_output: Any = None,
) -> bool:
    """
    Check if statistics pass the given test level.

    Args:
        stats: Statistics dictionary from compute_statistics().
        level: Test level to check against.
        jax_output: JAX output tensor (required for levels > SANITY).
        torch_output: PyTorch output tensor (required for levels > SANITY).

    Returns:
        True if stats pass the given level, False otherwise.
    """
    config = LEVEL_CONFIGS[level]

    if level == VerificationLevel.SANITY:
        # For sanity level, we only check shapes and NaN/Inf
        # This function assumes shapes already match if stats were computed
        return True

    if jax_output is None or torch_output is None:
        raise ValueError("jax_output and torch_output are required for levels > SANITY")

    jax_arr = _to_numpy(jax_output).flatten().astype(np.float64)
    torch_arr = _to_numpy(torch_output).flatten().astype(np.float64)

    atol = config['atol']
    rtol = config['rtol']
    fraction_required = config['fraction_required']

    # Check element-wise tolerance
    within_tolerance = np.abs(jax_arr - torch_arr) <= (atol + rtol * np.abs(jax_arr))
    fraction_within = float(np.mean(within_tolerance))

    return fraction_within >= fraction_required


def check_thresholds(stats: Dict[str, float], output_name: str) -> bool:
    """
    Check if statistics pass the per-output thresholds.

    Args:
        stats: Statistics dictionary from compute_statistics().
        output_name: Name of the output ('embeds_1bp', 'embeds_128bp', 'embeds_pair').

    Returns:
        True if stats pass all thresholds for the output, False otherwise.
    """
    if output_name not in THRESHOLDS:
        raise ValueError(f"Unknown output name: {output_name}. Valid names: {list(THRESHOLDS.keys())}")

    thresholds = THRESHOLDS[output_name]

    if stats['max_abs_error'] > thresholds['max_abs_error']:
        return False
    if stats['mean_abs_error'] > thresholds['mean_abs_error']:
        return False
    if stats['pearson_correlation'] < thresholds['min_correlation']:
        return False
    if stats['outlier_fraction'] > thresholds['outlier_fraction']:
        return False

    return True


def check_systematic_bias(
    jax_output: Any,
    torch_output: Any,
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """
    Check for systematic bias between JAX and PyTorch outputs using paired t-test.

    Args:
        jax_output: Output tensor from JAX model.
        torch_output: Output tensor from PyTorch model.
        alpha: Significance level for the test.

    Returns:
        Tuple of (has_bias, p_value):
            - has_bias: True if systematic bias is detected (p_value < alpha)
            - p_value: P-value from the paired t-test

    Raises:
        ImportError: If scipy is not available.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for check_systematic_bias(). Install with: pip install scipy")

    jax_arr = _to_numpy(jax_output).flatten().astype(np.float64)
    torch_arr = _to_numpy(torch_output).flatten().astype(np.float64)

    # For very large arrays, sample to avoid memory issues
    max_samples = 1_000_000
    if len(jax_arr) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(jax_arr), size=max_samples, replace=False)
        jax_arr = jax_arr[indices]
        torch_arr = torch_arr[indices]

    # Paired t-test
    _, p_value = ttest_rel(jax_arr, torch_arr)
    p_value = float(p_value)

    has_bias = p_value < alpha

    return has_bias, p_value


def check_no_nan_inf(tensor: Any) -> bool:
    """
    Check if a tensor contains no NaN or Inf values.

    Args:
        tensor: Input tensor (JAX, PyTorch, or numpy).

    Returns:
        True if tensor contains no NaN or Inf values, False otherwise.
    """
    arr = _to_numpy(tensor)
    return bool(np.all(np.isfinite(arr)))


# =============================================================================
# Expected Output Shapes Helper
# =============================================================================

def get_expected_shapes(seq_len: int = 16384, batch_size: int = 1) -> Dict[str, Tuple[int, ...]]:
    """
    Get expected output shapes for the model outputs.

    Args:
        seq_len: Sequence length (default 16384).
        batch_size: Batch size (default 1).

    Returns:
        Dictionary mapping output names to expected shapes.
    """
    return {
        'embeds_1bp': (batch_size, seq_len, 1536),
        'embeds_128bp': (batch_size, seq_len // 128, 3072),
        'embeds_pair': (batch_size, seq_len // 2048, seq_len // 2048, 128),
    }


def check_shapes(
    outputs: Dict[str, Any],
    seq_len: int = 16384,
    batch_size: int = 1,
) -> Dict[str, bool]:
    """
    Check if output shapes match expected shapes.

    Args:
        outputs: Dictionary of output tensors.
        seq_len: Sequence length.
        batch_size: Batch size.

    Returns:
        Dictionary mapping output names to whether shapes match.
    """
    expected = get_expected_shapes(seq_len, batch_size)
    results = {}

    for name, expected_shape in expected.items():
        if name not in outputs:
            results[name] = False
            continue

        output = outputs[name]
        if hasattr(output, 'shape'):
            actual_shape = tuple(output.shape)
        else:
            actual_shape = tuple(np.asarray(output).shape)

        results[name] = actual_shape == expected_shape

    return results
