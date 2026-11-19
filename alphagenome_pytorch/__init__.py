from __future__ import annotations

from alphagenome_pytorch.alphagenome import (
    AlphaGenome,
    Attention,
    BatchRMSNorm,
    DownresBlock,
    FeedForward,
    JunctionsLoss,
    MultinomialCrossEntropy,
    MultinomialLoss,
    PairwiseRowAttention,
    PoissonLoss,
    RelativePosFeatures,
    RotaryEmbedding,
    SoftClip,
    TargetScaler,
    TracksScaledPrediction,
    TransformerTower,
    TransformerUnet,
    UpresBlock,
    publication_heads_config,
    set_update_running_var,
)
from alphagenome_pytorch.alphagenome2 import (
    AlphaGenome2,
    BinomialLoss,
    ConditioningEmbedder,
    FiLM,
    GeneExpressionHead,
    MethylationHead,
    MultiHeadAttentionResidual,
    SingleCellAutoencoder,
)
from alphagenome_pytorch.config import AlphaGenomeConfig
from alphagenome_pytorch.data import (
    AlphaGenomeTFRecordDataset,
    BundleName,
    DataBatch,
    DummyGenomeDataset,
    DummyTargetsDataset,
    collate_batch,
)
from alphagenome_pytorch.dna_model import DNAModel, ModelSettings, create_from_jax_model
from alphagenome_pytorch.scoring import DirectGeneExpressionScorer

from . import evals, scoring

__all__ = [
    # core model
    'DNAModel',
    'ModelSettings',
    'create_from_jax_model',
    'AlphaGenome',
    'AlphaGenome2',
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
    'ConditioningEmbedder',
    'SingleCellAutoencoder',
    'FiLM',
    'MethylationHead',
    'MultiHeadAttentionResidual',
    'GeneExpressionHead',
    'BinomialLoss',
    'DirectGeneExpressionScorer',
    # data loading
    'DummyGenomeDataset',
    'DummyTargetsDataset',
    'AlphaGenomeTFRecordDataset',
    'DataBatch',
    'collate_batch',
    'BundleName',
    # submodules
    'scoring',
    'evals',
]
