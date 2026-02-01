from __future__ import annotations

from importlib import import_module

from alphagenome_pytorch.config import AlphaGenomeConfig

# Data loading utilities
from alphagenome_pytorch.data import (
    DummyGenomeDataset,
    DummyTargetsDataset,
    AlphaGenomeTFRecordDataset,
    DataBatch,
    collate_batch,
    BundleName,
)

_DNA_MODEL_ATTRS = {
    'DNAModel',
    'ModelSettings',
    'create_from_jax_model',
}

_MODEL_ATTRS = {
    'AlphaGenome',
    'Attention',
    'PairwiseRowAttention',
    'RelativePosFeatures',
    'RotaryEmbedding',
    'FeedForward',
    'TransformerTower',
    'TransformerUnet',
    'UpresBlock',
    'DownresBlock',
    'BatchRMSNorm',
    'TargetScaler',
    'MultinomialLoss',
    'JunctionsLoss',
    'TracksScaledPrediction',
    'SoftClip',
    'PoissonLoss',
    'MultinomialCrossEntropy',
    'set_update_running_var',
    'publication_heads_config',
}


def __getattr__(name: str):
    """Lazy import for optional modules and heavy model components."""
    if name in _DNA_MODEL_ATTRS:
        module = import_module('alphagenome_pytorch.dna_model')
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _MODEL_ATTRS:
        module = import_module('alphagenome_pytorch.alphagenome')
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == 'scoring':
        module = import_module('alphagenome_pytorch.scoring')
        globals()[name] = module
        return module
    if name == 'evals':
        module = import_module('alphagenome_pytorch.evals')
        globals()[name] = module
        return module
    raise AttributeError(f"module 'alphagenome_pytorch' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        list(globals().keys())
        + list(_DNA_MODEL_ATTRS)
        + list(_MODEL_ATTRS)
        + ['scoring', 'evals']
    )

__all__ = [
    # Core model
    'DNAModel',
    'ModelSettings',
    'create_from_jax_model',
    'AlphaGenome',
    'AlphaGenomeConfig',
    'Attention',
    'PairwiseRowAttention',
    'RelativePosFeatures',
    'RotaryEmbedding',
    'FeedForward',
    'TransformerTower',
    'TransformerUnet',
    'UpresBlock',
    'DownresBlock',
    'BatchRMSNorm',
    'TargetScaler',
    'MultinomialLoss',
    'JunctionsLoss',
    'TracksScaledPrediction',
    'SoftClip',
    'PoissonLoss',
    'MultinomialCrossEntropy',
    'set_update_running_var',
    'publication_heads_config',
    # Data loading
    'DummyGenomeDataset',
    'DummyTargetsDataset',
    'AlphaGenomeTFRecordDataset',
    'DataBatch',
    'collate_batch',
    'BundleName',
    # Lazy-loaded modules
    'scoring',
    'evals',
]
