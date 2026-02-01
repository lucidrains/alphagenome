# Ported from alphagenome_research reference implementation (reference/src/alphagenome_research/evals/)
"""Regression metrics for evaluating genomic track predictions.

This module provides stateful metrics for computing Pearson correlation,
MSE, and MAE across batches, with support for masked inputs and reduction
across distributed setups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass
class PearsonRState:
    """State for computing Pearson correlation coefficient incrementally.

    Accumulates sufficient statistics (sums, products, counts) to compute
    Pearson correlation across multiple batches without storing all data.
    """

    xy_sum: Tensor
    x_sum: Tensor
    xx_sum: Tensor
    y_sum: Tensor
    yy_sum: Tensor
    count: Tensor

    def __add__(self, other: 'PearsonRState') -> 'PearsonRState':
        """Combine two PearsonRState objects by summing their statistics."""
        return PearsonRState(
            xy_sum=self.xy_sum + other.xy_sum,
            x_sum=self.x_sum + other.x_sum,
            xx_sum=self.xx_sum + other.xx_sum,
            y_sum=self.y_sum + other.y_sum,
            yy_sum=self.yy_sum + other.yy_sum,
            count=self.count + other.count,
        )

    def to(self, device: torch.device | str) -> 'PearsonRState':
        """Move all tensors to the specified device."""
        return PearsonRState(
            xy_sum=self.xy_sum.to(device),
            x_sum=self.x_sum.to(device),
            xx_sum=self.xx_sum.to(device),
            y_sum=self.y_sum.to(device),
            yy_sum=self.yy_sum.to(device),
            count=self.count.to(device),
        )


def _pearsonr_initialize(device: torch.device | str | None = None) -> PearsonRState:
    """Initialize PearsonRState with zeros."""
    return PearsonRState(
        xy_sum=torch.zeros((), device=device),
        x_sum=torch.zeros((), device=device),
        xx_sum=torch.zeros((), device=device),
        y_sum=torch.zeros((), device=device),
        yy_sum=torch.zeros((), device=device),
        count=torch.zeros((), device=device),
    )


def _masked_sum(
    x: Tensor,
    axis: Sequence[int] | int | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute sum with optional masking, matching JAX jnp.sum where= semantics."""
    if mask is not None:
        x = x * mask.to(x.dtype)
    if axis is None:
        return x.sum().to(torch.float32)
    return x.sum(dim=axis).to(torch.float32)


def _pearsonr_update(
    x: Tensor,
    y: Tensor,
    axis: Sequence[int] | int | None = None,
    mask: Tensor | None = None,
) -> PearsonRState:
    """Compute PearsonRState from two arrays, optionally with masking."""
    if mask is not None:
        mask = mask.to(torch.bool)

    return PearsonRState(
        xy_sum=_masked_sum(x * y, axis=axis, mask=mask),
        x_sum=_masked_sum(x, axis=axis, mask=mask),
        xx_sum=_masked_sum(x.square(), axis=axis, mask=mask),
        y_sum=_masked_sum(y, axis=axis, mask=mask),
        yy_sum=_masked_sum(y.square(), axis=axis, mask=mask),
        count=_masked_sum(torch.ones_like(x), axis=axis, mask=mask),
    )


def _pearsonr_result(state: PearsonRState) -> Tensor:
    """Compute Pearson correlation coefficient from accumulated state."""
    x_mean = state.x_sum / state.count
    y_mean = state.y_sum / state.count

    covariance = state.xy_sum - state.count * x_mean * y_mean

    x_var = state.xx_sum - state.count * x_mean * x_mean
    y_var = state.yy_sum - state.count * y_mean * y_mean
    variance = x_var.sqrt() * y_var.sqrt()

    eps = torch.finfo(variance.dtype).eps
    return covariance / (variance + eps)


@dataclass
class RegressionState:
    """State for accumulating regression statistics across batches.

    Combines Pearson correlation (both raw and log1p transformed), MSE, and MAE.
    """

    pearsonr: PearsonRState
    pearsonr_log1p: PearsonRState
    sq_error: Tensor
    abs_error: Tensor
    count: Tensor

    def __add__(self, other: 'RegressionState') -> 'RegressionState':
        """Combine two RegressionState objects."""
        return RegressionState(
            pearsonr=self.pearsonr + other.pearsonr,
            pearsonr_log1p=self.pearsonr_log1p + other.pearsonr_log1p,
            sq_error=self.sq_error + other.sq_error,
            abs_error=self.abs_error + other.abs_error,
            count=self.count + other.count,
        )

    def to(self, device: torch.device | str) -> 'RegressionState':
        """Move all tensors to the specified device."""
        return RegressionState(
            pearsonr=self.pearsonr.to(device),
            pearsonr_log1p=self.pearsonr_log1p.to(device),
            sq_error=self.sq_error.to(device),
            abs_error=self.abs_error.to(device),
            count=self.count.to(device),
        )


def initialize_regression_metrics(
    device: torch.device | str | None = None,
) -> RegressionState:
    """Initialize a RegressionState with zeros."""
    return RegressionState(
        pearsonr=_pearsonr_initialize(device),
        pearsonr_log1p=_pearsonr_initialize(device),
        sq_error=torch.zeros((), device=device),
        abs_error=torch.zeros((), device=device),
        count=torch.zeros((), device=device),
    )


def update_regression_metrics(
    y_true: Tensor,
    y_pred: Tensor,
    mask: Tensor | None = None,
) -> RegressionState:
    """Compute regression metrics for a batch.

    Args:
        y_true: Ground truth tensor with shape (..., sequence, features).
        y_pred: Prediction tensor with same shape as y_true.
        mask: Optional boolean mask with same shape as y_true.

    Returns:
        RegressionState containing accumulated statistics for this batch.
    """
    y_true = y_true.to(torch.float32)
    y_pred = y_pred.to(torch.float32)

    # Reduce over sequence (-2) and batch (-3) dimensions, matching JAX axis=(-2, -3)
    reduction_axes = (-2, -3)

    return RegressionState(
        pearsonr=_pearsonr_update(y_true, y_pred, mask=mask, axis=reduction_axes),
        pearsonr_log1p=_pearsonr_update(
            torch.log1p(y_true), torch.log1p(y_pred), mask=mask, axis=reduction_axes
        ),
        sq_error=_masked_sum((y_true - y_pred).square(), axis=reduction_axes, mask=mask),
        abs_error=_masked_sum((y_true - y_pred).abs(), axis=reduction_axes, mask=mask),
        count=_masked_sum(torch.ones_like(y_true), axis=reduction_axes, mask=mask),
    )


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Compute mean of x where mask is True, matching JAX mean where= semantics."""
    valid_mask = mask > 0
    if not valid_mask.any():
        return torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
    return x[valid_mask].mean()


def finalize_regression_metrics(state: RegressionState) -> dict[str, Tensor]:
    """Compute final metrics from accumulated state.

    Args:
        state: RegressionState containing accumulated statistics.

    Returns:
        Dictionary with 'pearsonr', 'pearsonr_log1p', 'mse', 'mae' tensors.
    """
    pearsonr_values = _pearsonr_result(state.pearsonr)
    pearsonr_log1p_values = _pearsonr_result(state.pearsonr_log1p)

    pearsonr_count = state.pearsonr.count
    pearsonr_log1p_count = state.pearsonr_log1p.count

    return {
        'pearsonr': _masked_mean(pearsonr_values, pearsonr_count),
        'pearsonr_log1p': _masked_mean(pearsonr_log1p_values, pearsonr_log1p_count),
        'mse': _masked_mean(state.sq_error / state.count, state.count),
        'mae': _masked_mean(state.abs_error / state.count, state.count),
    }


def reduce_regression_metrics(
    previous_metrics: RegressionState,
    current_metrics: RegressionState,
) -> RegressionState:
    """Combine metrics from multiple batches or devices.

    Args:
        previous_metrics: Previously accumulated metrics.
        current_metrics: New metrics to add.

    Returns:
        Combined RegressionState.
    """
    return previous_metrics + current_metrics


def crop_sequence_length(x: Tensor, *, target_length: int) -> Tensor:
    """Crop a tensor to match the target length along the sequence dimension.

    Assumes sequence dimension is -2 (second to last). Centers the crop.

    Args:
        x: Input tensor with shape (..., S, D).
        target_length: Desired sequence length.

    Returns:
        Cropped tensor with shape (..., target_length, D).

    Raises:
        ValueError: If input length is shorter than target_length.
    """
    sequence_axis = -2
    input_length = x.shape[sequence_axis]

    if input_length < target_length:
        raise ValueError(
            f'Input length {input_length} is shorter than the requested '
            f'cropped length of {target_length}.'
        )
    elif input_length == target_length:
        return x

    ltrim = (input_length - target_length) // 2
    rtrim = input_length - target_length - ltrim

    slices = [slice(None)] * len(x.shape)
    slices[sequence_axis] = slice(ltrim, -rtrim if rtrim > 0 else None)
    return x[tuple(slices)]
