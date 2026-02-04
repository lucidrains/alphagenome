# Ported from alphagenome_research reference implementation.
"""PyTorch Dataset for AlphaGenome TFRecord data.

This module provides a PyTorch Dataset that loads AlphaGenome data from
TFRecord files using TensorFlow for reading and converting to PyTorch tensors.
"""

import functools
import os
import re
from collections.abc import Sequence
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

try:
    import tensorflow as tf
    import pandas as pd
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from alphagenome_pytorch.data.bundles import BundleName


_DEFAULT_PATH = 'gs://alphagenome-datasets/v1/train/'

_DNA_SEQUENCE_DTYPE = 'float32'
_DNA_SEQUENCE_FEATURE_SPEC = {
    'dna_sequence': 'string'
}
_INTERVAL_FEATURE_SPEC = {
    'interval/chromosome': 'string',
    'interval/start': 'int64',
    'interval/end': 'int64',
}
_FILENAME_REGEX = re.compile(
    r'data_(?P<chr>.+)_(?P<shard>\d+)-(?P<num_shards>\d+)\.gz\.tfrecord'
)


def _check_tensorflow():
    """Check if TensorFlow is available."""
    if not HAS_TENSORFLOW:
        raise ImportError(
            "TensorFlow is required to load TFRecord files. "
            "Install it with: pip install tensorflow"
        )


def get_tfrecords_df(
    *,
    organism: Optional[str] = None,
    bundle: Optional[BundleName] = None,
    fold_split: Optional[str] = None,
    subset: Optional[str] = None,
    chromosome: Optional[str] = None,
    path: Optional[str] = None,
) -> 'pd.DataFrame':
    """Return a dataframe with metadata about the TFRecord files.

    Args:
        organism: The organism to load (e.g., 'HOMO_SAPIENS', 'MUS_MUSCULUS').
            If None, all organisms are loaded.
        bundle: The bundle to load. If None, all bundles are loaded.
        fold_split: The fold split to load (e.g., 'FOLD_0'). If None, all are loaded.
        subset: The subset to load (e.g., 'TRAIN', 'VALID', 'TEST').
            If None, all subsets are loaded.
        chromosome: The chromosome to load. If None, all chromosomes are loaded.
        path: The path to the TFRecord files. If None, the default path is used.

    Returns:
        DataFrame with columns: organism, bundle, fold_split, subset,
        chromosome, shard, num_shards, path
    """
    _check_tensorflow()
    from etils import epath

    organism_pattern = organism if organism is not None else '*'
    fold_split_pattern = fold_split if fold_split is not None else '*'
    subset_pattern = subset if subset is not None else '*'
    chromosome_pattern = chromosome if chromosome is not None else '*'
    bundle_pattern = bundle.value.upper() if bundle is not None else '*'

    glob_pattern = '/'.join([
        fold_split_pattern,
        organism_pattern,
        subset_pattern,
        bundle_pattern,
        f'data_{chromosome_pattern}_*-*.gz.tfrecord',
    ])
    tfrecord_paths = list(epath.Path(path or _DEFAULT_PATH).glob(glob_pattern))

    def _parse_path(tfrecord_path):
        base_name = tfrecord_path.name
        match_ = _FILENAME_REGEX.match(base_name)
        if not match_:
            raise ValueError(f'Could not parse metadata for file: {base_name}')

        parsed = match_.groupdict()
        metadata = {
            'organism': tfrecord_path.parts[-4],
            'bundle': tfrecord_path.parts[-2],
            'fold_split': tfrecord_path.parts[-5],
            'subset': tfrecord_path.parts[-3],
            'chromosome': parsed['chr'],
            'shard': int(parsed['shard']),
            'num_shards': int(parsed['num_shards']),
            'path': str(tfrecord_path),
        }
        return pd.DataFrame(metadata, index=[0])

    if not tfrecord_paths:
        return pd.DataFrame()

    return pd.concat(
        [_parse_path(p) for p in tfrecord_paths]
    ).reset_index(drop=True)


def _get_tf_dtype(bundle: BundleName, key: str) -> 'tf.DType':
    """Get TensorFlow dtype for a bundle key."""
    dtypes = bundle.get_dtypes()
    torch_dtype = dtypes[key]

    # Map torch dtypes to TF dtypes
    dtype_map = {
        torch.bfloat16: tf.bfloat16,
        torch.float16: tf.float16,
        torch.float32: tf.float32,
        torch.int32: tf.int32,
        torch.int64: tf.int64,
        torch.bool: tf.bool,
    }
    return dtype_map.get(torch_dtype, tf.float32)


def _get_parse_function(bundle: BundleName):
    """Get parse function for a given bundle type."""
    _check_tensorflow()

    feature_spec = {
        'dna_sequence': tf.io.FixedLenFeature([], tf.string),
        'interval/chromosome': tf.io.FixedLenFeature([], tf.string),
        'interval/start': tf.io.FixedLenFeature([], tf.int64),
        'interval/end': tf.io.FixedLenFeature([], tf.int64),
    }
    for key in bundle.get_dtypes().keys():
        feature_spec[key] = tf.io.FixedLenFeature([], tf.string)

    output_dtypes = {'dna_sequence': tf.float32}
    for key in bundle.get_dtypes().keys():
        output_dtypes[key] = _get_tf_dtype(bundle, key)

    def _parse(proto):
        example = tf.io.parse_single_example(proto, feature_spec)
        for key, dtype in output_dtypes.items():
            example[key] = tf.io.parse_tensor(example[key], dtype)
        return example

    return _parse


def _tf_to_torch_dtype(tf_dtype: 'tf.DType') -> torch.dtype:
    """Convert TensorFlow dtype to PyTorch dtype."""
    dtype_map = {
        tf.bfloat16: torch.bfloat16,
        tf.float16: torch.float16,
        tf.float32: torch.float32,
        tf.float64: torch.float64,
        tf.int8: torch.int8,
        tf.int16: torch.int16,
        tf.int32: torch.int32,
        tf.int64: torch.int64,
        tf.bool: torch.bool,
    }
    return dtype_map.get(tf_dtype, torch.float32)


def _numpy_to_torch(arr: np.ndarray, target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert numpy array to torch tensor with appropriate dtype handling."""
    # Handle bfloat16 - numpy doesn't support it natively
    if arr.dtype == np.float32 and target_dtype == torch.bfloat16:
        return torch.from_numpy(arr).to(torch.bfloat16)
    elif arr.dtype.name == 'bfloat16':
        # If somehow we get bfloat16 numpy array (unlikely)
        return torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    else:
        tensor = torch.from_numpy(arr)
        if target_dtype is not None and tensor.dtype != target_dtype:
            tensor = tensor.to(target_dtype)
        return tensor


class AlphaGenomeTFRecordDataset(Dataset):
    """PyTorch Dataset for AlphaGenome TFRecord files.

    This dataset loads TFRecord files using TensorFlow and converts
    the data to PyTorch tensors. It supports both lazy loading (reading
    from disk on each access) and in-memory caching.

    Example usage:
        >>> dataset = AlphaGenomeTFRecordDataset(
        ...     organism='HOMO_SAPIENS',
        ...     fold_split='FOLD_0',
        ...     subset='TRAIN',
        ...     bundles=[BundleName.ATAC, BundleName.CAGE],
        ... )
        >>> sample = dataset[0]
        >>> dna_seq = sample['dna_sequence']  # [S, 4] tensor
    """

    def __init__(
        self,
        *,
        organism: str = 'HOMO_SAPIENS',
        fold_split: str = 'FOLD_0',
        subset: str = 'TRAIN',
        bundles: Optional[Sequence[BundleName]] = None,
        path: Optional[str] = None,
        cache_in_memory: bool = False,
        organism_index: Optional[int] = None,
    ):
        """Initialize the dataset.

        Args:
            organism: The organism to load (e.g., 'HOMO_SAPIENS', 'MUS_MUSCULUS').
            fold_split: The fold split to load (e.g., 'FOLD_0').
            subset: The subset to load (e.g., 'TRAIN', 'VALID', 'TEST').
            bundles: The bundles to load. If None, all bundles are loaded.
            path: The path to the TFRecord files. If None, uses default path.
            cache_in_memory: If True, cache all data in memory after first load.
            organism_index: The organism index (0 for human, 1 for mouse).
                If None, inferred from organism name.
        """
        _check_tensorflow()

        self.organism = organism
        self.fold_split = fold_split
        self.subset = subset
        self.bundles = bundles or list(BundleName)
        self.path = path or _DEFAULT_PATH
        self.cache_in_memory = cache_in_memory

        # Infer organism index
        if organism_index is not None:
            self.organism_index = organism_index
        elif organism.upper() in ('HOMO_SAPIENS', 'HUMAN'):
            self.organism_index = 0
        elif organism.upper() in ('MUS_MUSCULUS', 'MOUSE'):
            self.organism_index = 1
        else:
            self.organism_index = 0

        # Load metadata and create dataset
        self._load_metadata()
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._tf_dataset: Optional['tf.data.Dataset'] = None
        self._data_list: Optional[List[Dict[str, np.ndarray]]] = None

    def _load_metadata(self):
        """Load metadata about available TFRecord files."""
        records = []
        for bundle in self.bundles:
            df = get_tfrecords_df(
                organism=self.organism,
                bundle=bundle,
                fold_split=self.fold_split,
                subset=self.subset,
                path=self.path,
            )
            records.append(df)

        if not records or all(df.empty for df in records):
            self._metadata_df = pd.DataFrame()
            self._num_samples = 0
            return

        self._metadata_df = pd.concat(records).reset_index(drop=True)

        # Count samples by loading dataset length
        # For efficiency, we just use one bundle to count
        bundle_df = self._metadata_df[
            self._metadata_df['bundle'] == self.bundles[0].value.upper()
        ]
        if bundle_df.empty:
            self._num_samples = 0
            return

        # Get number of samples from first bundle's files
        self._bundle_paths: Dict[BundleName, List[str]] = {}
        for bundle in self.bundles:
            bundle_name = bundle.value.upper()
            paths = self._metadata_df[
                self._metadata_df['bundle'] == bundle_name
            ].sort_values('shard')['path'].tolist()
            self._bundle_paths[bundle] = paths

        # Count samples from first bundle
        self._num_samples = self._count_samples(self.bundles[0])

    def _count_samples(self, bundle: BundleName) -> int:
        """Count the number of samples in the dataset for a given bundle."""
        paths = self._bundle_paths.get(bundle, [])
        if not paths:
            return 0

        count = 0
        for path in paths:
            ds = tf.data.TFRecordDataset(path, compression_type='GZIP')
            for _ in ds:
                count += 1
        return count

    def _load_all_data(self):
        """Load all data into memory."""
        if self._data_list is not None:
            return

        self._data_list = []

        # Create datasets for each bundle
        bundle_datasets = {}
        for bundle in self.bundles:
            paths = self._bundle_paths.get(bundle, [])
            if not paths:
                continue

            parser = _get_parse_function(bundle)
            ds = tf.data.TFRecordDataset(paths, compression_type='GZIP')
            ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
            bundle_datasets[bundle] = iter(ds)

        # Zip and iterate through all data
        if not bundle_datasets:
            return

        # Load all samples
        try:
            while True:
                sample = {}
                for bundle, ds_iter in bundle_datasets.items():
                    try:
                        element = next(ds_iter)
                        for key, value in element.items():
                            if key not in sample:
                                sample[key] = value.numpy()
                    except StopIteration:
                        raise StopIteration
                self._data_list.append(sample)
        except StopIteration:
            pass

        self._num_samples = len(self._data_list)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.cache_in_memory and self._data_list is not None:
            return len(self._data_list)
        return self._num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample by index.

        Args:
            idx: The sample index.

        Returns:
            Dictionary mapping field names to torch tensors.
        """
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Check cache first
        if idx in self._cache:
            return self._cache[idx]

        # Load all data if caching in memory
        if self.cache_in_memory:
            self._load_all_data()
            if self._data_list and idx < len(self._data_list):
                sample = self._convert_sample(self._data_list[idx])
                self._cache[idx] = sample
                return sample

        # Otherwise, load on demand (less efficient for random access)
        sample = self._load_sample(idx)
        if self.cache_in_memory:
            self._cache[idx] = sample
        return sample

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample from disk."""
        # This is inefficient for random access - consider using cache_in_memory
        self._load_all_data()
        if self._data_list and idx < len(self._data_list):
            return self._convert_sample(self._data_list[idx])
        raise IndexError(f"Index {idx} out of range")

    def _convert_sample(self, np_sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert a numpy sample to torch tensors."""
        result: Dict[str, torch.Tensor] = {}

        for key, value in np_sample.items():
            if key.startswith('interval/'):
                # Skip metadata fields
                continue

            # Convert to torch tensor
            if isinstance(value, bytes):
                # String values (like chromosome)
                continue
            elif isinstance(value, np.ndarray):
                result[key] = _numpy_to_torch(value)
            else:
                result[key] = torch.tensor(value)

        # Add organism index
        result['organism_index'] = torch.tensor(self.organism_index, dtype=torch.int32)

        return result


class AlphaGenomeTFRecordIterableDataset(IterableDataset):
    """Iterable PyTorch Dataset for streaming AlphaGenome TFRecord files.

    This is more efficient for large datasets where random access is not needed.
    It streams data directly from TFRecords without loading everything into memory.

    Example usage:
        >>> dataset = AlphaGenomeTFRecordIterableDataset(
        ...     organism='HOMO_SAPIENS',
        ...     fold_split='FOLD_0',
        ...     subset='TRAIN',
        ...     bundles=[BundleName.ATAC, BundleName.CAGE],
        ... )
        >>> for sample in dataset:
        ...     dna_seq = sample['dna_sequence']
    """

    def __init__(
        self,
        *,
        organism: str = 'HOMO_SAPIENS',
        fold_split: str = 'FOLD_0',
        subset: str = 'TRAIN',
        bundles: Optional[Sequence[BundleName]] = None,
        path: Optional[str] = None,
        organism_index: Optional[int] = None,
        shuffle: bool = False,
        shuffle_buffer_size: int = 1000,
    ):
        """Initialize the iterable dataset.

        Args:
            organism: The organism to load.
            fold_split: The fold split to load.
            subset: The subset to load.
            bundles: The bundles to load. If None, all bundles are loaded.
            path: The path to the TFRecord files.
            organism_index: The organism index. If None, inferred from organism.
            shuffle: Whether to shuffle the data.
            shuffle_buffer_size: Buffer size for shuffling.
        """
        _check_tensorflow()

        self.organism = organism
        self.fold_split = fold_split
        self.subset = subset
        self.bundles = bundles or list(BundleName)
        self.path = path or _DEFAULT_PATH
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        # Infer organism index
        if organism_index is not None:
            self.organism_index = organism_index
        elif organism.upper() in ('HOMO_SAPIENS', 'HUMAN'):
            self.organism_index = 0
        elif organism.upper() in ('MUS_MUSCULUS', 'MOUSE'):
            self.organism_index = 1
        else:
            self.organism_index = 0

        # Load metadata
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata about available TFRecord files."""
        self._bundle_paths: Dict[BundleName, List[str]] = {}

        for bundle in self.bundles:
            df = get_tfrecords_df(
                organism=self.organism,
                bundle=bundle,
                fold_split=self.fold_split,
                subset=self.subset,
                path=self.path,
            )
            if not df.empty:
                self._bundle_paths[bundle] = df.sort_values('shard')['path'].tolist()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset."""
        if not self._bundle_paths:
            return

        # Create datasets for each bundle
        bundle_datasets = {}
        for bundle in self.bundles:
            paths = self._bundle_paths.get(bundle, [])
            if not paths:
                continue

            parser = _get_parse_function(bundle)
            ds = tf.data.TFRecordDataset(paths, compression_type='GZIP')
            ds = ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)

            if self.shuffle:
                ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)

            bundle_datasets[bundle] = ds

        if not bundle_datasets:
            return

        # Zip all bundle datasets
        first_bundle = list(bundle_datasets.keys())[0]
        zipped_ds = bundle_datasets[first_bundle]

        for bundle in list(bundle_datasets.keys())[1:]:
            zipped_ds = tf.data.Dataset.zip((zipped_ds, bundle_datasets[bundle]))

        # Iterate and yield torch samples
        for element in zipped_ds:
            yield self._convert_element(element)

    def _convert_element(self, element) -> Dict[str, torch.Tensor]:
        """Convert a TF dataset element to torch tensors."""
        result: Dict[str, torch.Tensor] = {}

        # Handle zipped elements
        if isinstance(element, tuple):
            for sub_element in element:
                result.update(self._process_dict(sub_element))
        else:
            result.update(self._process_dict(element))

        # Add organism index
        result['organism_index'] = torch.tensor(self.organism_index, dtype=torch.int32)

        return result

    def _process_dict(self, element) -> Dict[str, torch.Tensor]:
        """Process a single element dictionary."""
        result = {}
        for key, value in element.items():
            if key.startswith('interval/'):
                continue

            # Convert TF tensor to numpy, then to torch
            if hasattr(value, 'numpy'):
                arr = value.numpy()
                result[key] = _numpy_to_torch(arr)
            elif isinstance(value, np.ndarray):
                result[key] = _numpy_to_torch(value)

        return result
