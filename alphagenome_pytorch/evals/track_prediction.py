# Ported from alphagenome_research reference implementation (reference/src/alphagenome_research/evals/)
"""Evaluation utilities for genomic track prediction.

This module provides high-level evaluation functions for assessing model
performance on track prediction tasks across different genomic data bundles.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from alphagenome_pytorch.evals.regression_metrics import (
    RegressionState,
    crop_sequence_length,
    finalize_regression_metrics,
    initialize_regression_metrics,
    reduce_regression_metrics,
    update_regression_metrics,
)

if TYPE_CHECKING:
    from alphagenome_pytorch.data import BundleName, DataBatch


logger = logging.getLogger(__name__)


# Default bundles to evaluate (matching JAX reference)
DEFAULT_EVAL_BUNDLES = [
    'ATAC',
    'CAGE',
    'CHIP_HISTONE',
    'CHIP_TF',
    'DNASE',
    'PROCAP',
    'RNA_SEQ',
]

# Mapping from bundle name to output head key
_BUNDLE_TO_OUTPUT_KEY = {
    'ATAC': 'atac',
    'CAGE': 'cage',
    'CHIP_HISTONE': 'chip_histone',
    'CHIP_TF': 'chip_tf',
    'DNASE': 'dnase',
    'PROCAP': 'procap',
    'RNA_SEQ': 'rna_seq',
}


def _get_bundle_predictions(
    predictions: dict,
    bundle_name: str,
    organism: str = 'human',
) -> Tensor | None:
    """Extract predictions for a specific bundle from model output.

    Handles both the reference JAX-style output format and the custom
    PyTorch head output format.

    Args:
        predictions: Model output dictionary.
        bundle_name: Name of the bundle (e.g., 'ATAC', 'CAGE').
        organism: Organism key in predictions (default 'human').

    Returns:
        Tensor of predictions for the bundle, or None if not found.
    """
    output_key = _BUNDLE_TO_OUTPUT_KEY.get(bundle_name, bundle_name.lower())

    # Try organism-keyed output first (PyTorch format)
    if organism in predictions:
        organism_preds = predictions[organism]
        if output_key in organism_preds:
            head_output = organism_preds[output_key]
            # Handle GenomeTracksHead output format (dict with resolution keys)
            if isinstance(head_output, dict):
                # Prefer 1bp resolution for 1bp bundles, 128bp for others
                if 'scaled_predictions_1bp' in head_output:
                    return head_output['scaled_predictions_1bp']
                elif 'scaled_predictions_128bp' in head_output:
                    return head_output['scaled_predictions_128bp']
            return head_output

    # Try direct key access (JAX format)
    if output_key in predictions:
        return predictions[output_key]

    return None


def _get_bundle_targets(
    batch: 'DataBatch',
    bundle_name: str,
) -> tuple[Tensor, Tensor | None]:
    """Extract targets and mask for a bundle from the batch.

    Args:
        batch: DataBatch containing genomic data.
        bundle_name: Name of the bundle.

    Returns:
        Tuple of (targets, mask) where mask may be None.
    """
    # Use the batch's get_genome_tracks method if available
    if hasattr(batch, 'get_genome_tracks'):
        from alphagenome_pytorch.data import BundleName
        try:
            bundle = BundleName[bundle_name]
            return batch.get_genome_tracks(bundle)
        except (KeyError, AttributeError):
            pass

    # Fallback: try direct attribute access
    data_key = bundle_name.lower()
    mask_key = f'{data_key}_mask'

    targets = getattr(batch, data_key, None)
    mask = getattr(batch, mask_key, None)

    if targets is None:
        raise ValueError(f'Bundle {bundle_name} not found in batch')

    return targets, mask


def evaluate(
    model: nn.Module,
    dataloader: DataLoader | Iterator,
    device: str | torch.device = 'cuda',
    bundles: list[str] | None = None,
    organism: str = 'human',
    log_frequency: int = 5,
    max_batches: int | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate model on dataset, returning per-bundle metrics.

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader or iterator yielding batches.
        device: Device to run evaluation on.
        bundles: List of bundle names to evaluate. Defaults to all standard bundles.
        organism: Organism key for predictions (default 'human').
        log_frequency: How often to log progress (every N batches).
        max_batches: Optional limit on number of batches to process.

    Returns:
        Dictionary mapping bundle names to their metric dictionaries.
        Each metric dict contains 'pearsonr', 'pearsonr_log1p', 'mse', 'mae'.
    """
    if bundles is None:
        bundles = DEFAULT_EVAL_BUNDLES

    model.eval()
    model.to(device)

    # Initialize metrics for each bundle
    metrics: dict[str, RegressionState] = {
        bundle: initialize_regression_metrics(device=device)
        for bundle in bundles
    }

    num_elements = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            # Handle different batch formats
            if isinstance(batch, tuple):
                batch = batch[0]  # (batch, metadata) format

            # Move batch to device
            if hasattr(batch, 'to'):
                batch = batch.to(device)
            elif hasattr(batch, 'dna_sequence'):
                # DataBatch or similar - move individual tensors
                dna_sequence = batch.dna_sequence.to(device)
                organism_index = batch.organism_index.to(device)
            else:
                dna_sequence = batch['dna_sequence'].to(device)
                organism_index = batch['organism_index'].to(device)

            # Get model predictions
            if hasattr(batch, 'dna_sequence'):
                predictions = model(batch.dna_sequence.to(device), batch.organism_index.to(device))
            else:
                predictions = model(dna_sequence, organism_index)

            # Track batch size
            if hasattr(batch, 'dna_sequence'):
                num_elements += batch.dna_sequence.shape[0]
            else:
                num_elements += dna_sequence.shape[0]

            # Update metrics for each bundle
            for bundle in bundles:
                try:
                    targets_true, mask = _get_bundle_targets(batch, bundle)
                    targets_pred = _get_bundle_predictions(predictions, bundle, organism)

                    if targets_pred is None:
                        logger.warning(f'Predictions not found for bundle {bundle}')
                        continue

                    # Move targets to device if needed
                    targets_true = targets_true.to(device)
                    if mask is not None:
                        mask = mask.to(device)

                    # Crop predictions to match target length
                    target_length = targets_true.shape[-2]
                    targets_pred = crop_sequence_length(
                        targets_pred, target_length=target_length
                    )

                    # Update metrics
                    step_metrics = update_regression_metrics(
                        targets_true, targets_pred, mask
                    )
                    metrics[bundle] = reduce_regression_metrics(
                        metrics[bundle], step_metrics
                    )

                except Exception as e:
                    logger.warning(f'Error processing bundle {bundle}: {e}')
                    continue

            # Log progress
            if i % log_frequency == 1 and i > 0:
                current_metrics = {
                    bundle: finalize_regression_metrics(state)
                    for bundle, state in metrics.items()
                }
                logger.info(f'Step {i}: {current_metrics}')

    logger.info(f'Total elements processed: {num_elements}')

    # Finalize and return metrics
    final_metrics = {}
    for bundle, state in metrics.items():
        bundle_metrics = finalize_regression_metrics(state)
        # Convert tensors to Python floats for easier consumption
        final_metrics[bundle] = {
            key: value.item() if hasattr(value, 'item') else float(value)
            for key, value in bundle_metrics.items()
        }

    return final_metrics


def evaluate_bundle(
    model: nn.Module,
    dataloader: DataLoader | Iterator,
    bundle_name: str,
    device: str | torch.device = 'cuda',
    organism: str = 'human',
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate model on a single bundle.

    Convenience function for evaluating a single bundle.

    Args:
        model: PyTorch model to evaluate.
        dataloader: DataLoader or iterator yielding batches.
        bundle_name: Name of the bundle to evaluate.
        device: Device to run evaluation on.
        organism: Organism key for predictions.
        max_batches: Optional limit on number of batches.

    Returns:
        Dictionary with 'pearsonr', 'pearsonr_log1p', 'mse', 'mae'.
    """
    results = evaluate(
        model=model,
        dataloader=dataloader,
        device=device,
        bundles=[bundle_name],
        organism=organism,
        max_batches=max_batches,
    )
    return results.get(bundle_name, {})


def flatten_metrics(
    metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Flatten per-bundle metrics into a single dictionary.

    Args:
        metrics: Nested dict mapping bundle -> metric_name -> value.

    Returns:
        Flat dict with keys like 'ATAC_pearsonr', 'CAGE_mse', etc.
    """
    flattened = {}
    for bundle, bundle_metrics in metrics.items():
        for metric_name, value in bundle_metrics.items():
            flattened[f'{bundle}_{metric_name}'] = value
    return flattened
