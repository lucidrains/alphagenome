from alphagenome_pytorch.alphagenome import (
    AlphaGenome,
    Attention,
    PairwiseRowAttention,
    RelativePosFeatures,
    RotaryEmbedding,
    FeedForward,
    TransformerTower,
    TransformerUnet,
    UpresBlock,
    DownresBlock,
    BatchRMSNorm,
    TargetScaler,
    MultinomialLoss,
    JunctionsLoss,
    TracksScaledPrediction,
    SoftClip,
    PoissonLoss,
    MultinomialCrossEntropy,
    set_update_running_var,
    publication_heads_config
)

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

# Scoring modules (lazy import to avoid dependency issues)
def __getattr__(name):
    """Lazy import for optional modules that require additional dependencies."""
    if name == 'scoring':
        from alphagenome_pytorch import scoring
        return scoring
    elif name == 'evals':
        from alphagenome_pytorch import evals
        return evals
    raise AttributeError(f"module 'alphagenome_pytorch' has no attribute {name!r}")

__all__ = [
    # Core model
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
