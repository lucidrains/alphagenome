# Ported from alphagenome_research reference implementation (reference/src/alphagenome_research/evals/)
"""Evaluation metrics for PyTorch AlphaGenome.

This module provides evaluation metrics and utilities for assessing model
performance on genomic track prediction tasks.
"""

from alphagenome_pytorch.evals.regression_metrics import (
    PearsonRState,
    RegressionState,
    initialize_regression_metrics,
    update_regression_metrics,
    finalize_regression_metrics,
    reduce_regression_metrics,
    crop_sequence_length,
)
from alphagenome_pytorch.evals.track_prediction import evaluate

__all__ = [
    'PearsonRState',
    'RegressionState',
    'initialize_regression_metrics',
    'update_regression_metrics',
    'finalize_regression_metrics',
    'reduce_regression_metrics',
    'crop_sequence_length',
    'evaluate',
]
