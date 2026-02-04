# Ported from alphagenome_research reference implementation.
"""Data loading utilities for PyTorch AlphaGenome.

This module provides data loading utilities for training and inference,
including TFRecord dataset loading and batch collation.
"""

from alphagenome_pytorch.data.batch import (
    DataBatch,
    collate_batch,
    dict_to_batch,
)
from alphagenome_pytorch.data.bundles import BundleName
from alphagenome_pytorch.data.dataset import (
    AlphaGenomeTFRecordDataset,
    AlphaGenomeTFRecordIterableDataset,
    get_tfrecords_df,
)
from alphagenome_pytorch.data.dummy import (
    DummyGenomeDataset,
    DummyTargetsDataset,
)

__all__ = [
    'AlphaGenomeTFRecordDataset',
    'AlphaGenomeTFRecordIterableDataset',
    'BundleName',
    'DataBatch',
    'DummyGenomeDataset',
    'DummyTargetsDataset',
    'collate_batch',
    'dict_to_batch',
    'get_tfrecords_df',
]
