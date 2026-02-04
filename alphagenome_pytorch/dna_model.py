from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from alphagenome_pytorch.alphagenome import AlphaGenome

if TYPE_CHECKING:
    from alphagenome.data import genome
    from alphagenome.data import junction_data
    from alphagenome.data import track_data
    from alphagenome.models import dna_model as dna_model_types
    from alphagenome.models import dna_output
    from alphagenome_research.io import fasta
    from alphagenome_research.model.metadata import metadata as metadata_lib


_DNA_TO_INDEX = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
    'N': -1,
}

_INDEX_TO_DNA = np.array(['A', 'C', 'G', 'T'])
_PAD_VALUE = -1
_NUM_SPLICE_CLASSES = 4

_OUTPUT_NAME_TO_RESOLUTION = {
    'ATAC': 1,
    'CAGE': 1,
    'DNASE': 1,
    'RNA_SEQ': 1,
    'PROCAP': 1,
    'CHIP_HISTONE': 128,
    'CHIP_TF': 128,
    'SPLICE_SITES': 1,
    'SPLICE_SITE_USAGE': 1,
    'SPLICE_JUNCTIONS': 1,
    'CONTACT_MAPS': 2048,
}

_HEAD_TO_OUTPUT_NAME = {
    'atac': 'ATAC',
    'cage': 'CAGE',
    'dnase': 'DNASE',
    'rna_seq': 'RNA_SEQ',
    'procap': 'PROCAP',
    'chip_histone': 'CHIP_HISTONE',
    'chip_tf': 'CHIP_TF',
    'contact_maps': 'CONTACT_MAPS',
    'splice_sites_classification': 'SPLICE_SITES',
    'splice_sites_usage': 'SPLICE_SITE_USAGE',
    'splice_sites_junction': 'SPLICE_JUNCTIONS',
}


@dataclass
class ModelSettings:
    """Settings for convenience DNA model operations."""

    num_splice_sites: int = 512
    splice_site_threshold: float = 0.1


def create(
    *,
    model: AlphaGenome | None = None,
    model_kwargs: Mapping[str, object] | None = None,
    settings: ModelSettings | None = None,
    device: torch.device | str | None = None,
    add_reference_heads: bool = False,
    organisms: Sequence[str] = ('human', 'mouse'),
    checkpoint_path: str | None = None,
    strict: bool = False,
    compile: bool = False,
    compile_kwargs: Mapping[str, object] | None = None,
) -> 'DNAModel':
    """Create a DNAModel wrapper with optional heads and checkpoint loading.

    Args:
        model: Existing AlphaGenome model, or None to create a new one.
        model_kwargs: Arguments passed to AlphaGenome() if creating new model.
        settings: ModelSettings for convenience operations.
        device: Device to move model to ('cuda', 'cpu', etc.).
        add_reference_heads: If True, add JAX-aligned reference heads.
        organisms: Organisms to add heads for (default: human, mouse).
        checkpoint_path: Path to checkpoint file to load.
        strict: If True, require exact state_dict match when loading.
        compile: If True, compile the model with torch.compile for faster inference.
        compile_kwargs: Arguments passed to torch.compile (mode, backend, etc.).

    Returns:
        DNAModel wrapper around the AlphaGenome model.
    """
    if model is None:
        model = AlphaGenome(**(model_kwargs or {}))
    if add_reference_heads:
        for organism in organisms:
            model.add_reference_heads(organism)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=strict)
    if compile:
        model = torch.compile(model, **(compile_kwargs or {}))
    return DNAModel(model, settings=settings, device=device)


def _encode_sequence(seq: str) -> Tensor:
    seq = seq.upper()
    indices = [_DNA_TO_INDEX.get(base, -1) for base in seq]
    return torch.tensor(indices, dtype=torch.long)


def _as_token_tensor(sequences: str | Sequence[str] | Tensor | np.ndarray) -> Tensor:
    if isinstance(sequences, Tensor):
        if sequences.dtype.is_floating_point:
            if sequences.ndim >= 2 and sequences.shape[-1] == 4:
                return sequences.argmax(dim=-1).long()
            raise ValueError('Floating-point sequences must be one-hot with last dim=4.')
        if sequences.ndim == 1:
            return sequences.unsqueeze(0).long()
        return sequences.long()

    if isinstance(sequences, np.ndarray):
        return _as_token_tensor(torch.from_numpy(sequences))

    if isinstance(sequences, str):
        return _encode_sequence(sequences).unsqueeze(0)

    if isinstance(sequences, Sequence):
        if not sequences:
            raise ValueError('Empty sequence list.')
        encoded = [_encode_sequence(seq) for seq in sequences]
        lengths = {seq.shape[0] for seq in encoded}
        if len(lengths) != 1:
            raise ValueError('All sequences must have the same length.')
        return torch.stack(encoded, dim=0)

    raise TypeError(
        'Sequences must be a string, a list of strings, a numpy array, or a torch Tensor.'
    )


def _resolve_output_name(output_key) -> str | None:
    if output_key is None:
        return None
    if hasattr(output_key, 'name'):
        return output_key.name
    if isinstance(output_key, str):
        if output_key in _HEAD_TO_OUTPUT_NAME:
            return _HEAD_TO_OUTPUT_NAME[output_key]
        return output_key.upper()
    return None


def _select_head_output(head_output, output_name: str):
    resolution = _OUTPUT_NAME_TO_RESOLUTION.get(output_name)
    if isinstance(head_output, dict):
        if resolution == 1 and 'scaled_predictions_1bp' in head_output:
            return head_output['scaled_predictions_1bp']
        if resolution == 128 and 'scaled_predictions_128bp' in head_output:
            return head_output['scaled_predictions_128bp']
        if resolution == 2048 and 'scaled_predictions_2048bp' in head_output:
            return head_output['scaled_predictions_2048bp']
    return head_output


class DNAModel:
    """PyTorch DNA model wrapper for AlphaGenome.

    Provides convenience helpers for predictions, variant scoring, and ISM.
    Compatible with the JAX API for predict_interval, predict_variant, etc.
    """

    def __init__(
        self,
        model: AlphaGenome,
        *,
        settings: ModelSettings | None = None,
        device: torch.device | str | None = None,
        organism_map: Mapping[str, int] | None = None,
        fasta_extractors: Mapping['dna_model_types.Organism', 'fasta.FastaExtractor'] | None = None,
        splice_site_extractors: Mapping['dna_model_types.Organism', object] | None = None,
        metadata: Mapping['dna_model_types.Organism', 'metadata_lib.AlphaGenomeOutputMetadata'] | None = None,
        output_metadata_by_organism: Mapping['dna_model_types.Organism', 'dna_output.OutputMetadata'] | None = None,
    ) -> None:
        self.model = model
        self.settings = settings or ModelSettings()
        self.organism_map = dict(organism_map or {'human': 0, 'mouse': 1})
        self._fasta_extractors = fasta_extractors or {}
        self._splice_site_extractors = splice_site_extractors or {}
        self._metadata = metadata or {}
        self._output_metadata_by_organism = output_metadata_by_organism or {}
        if device is not None:
            self.model.to(device)

    @property
    def device(self) -> torch.device:
        params = list(self.model.parameters())
        return params[0].device if params else torch.device('cpu')

    def to(self, device: torch.device | str) -> 'DNAModel':
        self.model.to(device)
        return self

    def eval(self) -> 'DNAModel':
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> 'DNAModel':
        self.model.train(mode)
        return self

    def _resolve_organism_index(
        self,
        organism: str | None,
        organism_index: int | Tensor | None,
        batch_size: int,
    ) -> Tensor:
        if organism_index is None:
            if organism is None:
                organism_index = 0
            else:
                if organism not in self.organism_map:
                    raise KeyError(
                        f'Unknown organism {organism!r}. '
                        f'Available: {list(self.organism_map.keys())}'
                    )
                organism_index = self.organism_map[organism]

        if isinstance(organism_index, Tensor):
            return organism_index.to(device=self.device)
        return torch.full(
            (batch_size,),
            int(organism_index),
            dtype=torch.long,
            device=self.device,
        )

    def _pick_organism_key(
        self,
        predictions: Mapping[str, object],
        organism: str | None,
        return_all_organisms: bool,
    ) -> str | None:
        if return_all_organisms:
            return None
        if organism is not None:
            return organism
        if len(predictions) == 1:
            return next(iter(predictions.keys()))
        return None

    def _resolve_track_masks(
        self,
        organism: str | None,
        requested_outputs: Iterable[str] | None,
        ontology_terms: Iterable[str] | None,
    ) -> dict[str, Tensor] | None:
        """Convert ontology terms to per-head track masks.

        Args:
            organism: Organism name ('human', 'mouse').
            requested_outputs: Head names to filter (e.g., 'rna_seq', 'cage').
            ontology_terms: Cell type/tissue ontology CURIEs (e.g., 'EFO:0001187').

        Returns:
            Dict mapping head names to boolean tensors, or None if no filtering.
        """
        if ontology_terms is None:
            return None

        # Convert organism string to enum for metadata lookup
        _, organism_enum = self._convert_organism(organism)

        metadata = self._metadata.get(organism_enum)
        if metadata is None:
            return None

        ontology_curies = set(ontology_terms)

        # Determine which heads to filter
        organism_str = organism or 'human'
        heads_dict = self.model.heads[organism_str] if organism_str in self.model.heads else {}
        head_names = set(requested_outputs) if requested_outputs else set(heads_dict.keys())

        track_masks = {}
        for head_name in head_names:
            # Try to get metadata for this head
            # Map head name to output type for metadata lookup
            output_name = _HEAD_TO_OUTPUT_NAME.get(head_name)
            if output_name is None:
                continue

            try:
                from alphagenome.models import dna_output
                output_type = dna_output.OutputType[output_name]
                track_metadata = metadata.get(output_type)
            except (KeyError, AttributeError):
                continue

            if track_metadata is None:
                continue

            # Check if metadata has ontology_curie field
            if not hasattr(track_metadata, 'ontology_curie') and not isinstance(track_metadata, pd.DataFrame):
                continue

            try:
                if isinstance(track_metadata, pd.DataFrame):
                    curies = track_metadata['ontology_curie'].values
                else:
                    curies = track_metadata.ontology_curie
            except (AttributeError, KeyError):
                continue

            # Create boolean mask for tracks matching any of the ontology terms
            mask = torch.tensor(
                [str(curie) in ontology_curies for curie in curies],
                dtype=torch.bool,
                device=self.device,
            )

            track_masks[head_name] = mask

        return track_masks if track_masks else None

    def _infer_splice_site_positions_from_probs(self, probs: Tensor) -> Tensor:
        probs = probs.float()
        batch, seq_len, _ = probs.shape
        num_sites = min(self.settings.num_splice_sites, seq_len)
        pad_len = self.settings.num_splice_sites - num_sites

        positions = []
        sentinel = seq_len

        for class_idx in range(_NUM_SPLICE_CLASSES):
            scores = probs[:, :, class_idx]
            top_values, top_indices = scores.topk(num_sites, dim=1)

            if self.settings.splice_site_threshold > 0:
                invalid = top_values < self.settings.splice_site_threshold
                top_indices = top_indices.masked_fill(invalid, sentinel)

            top_indices, _ = top_indices.sort(dim=1)

            if self.settings.splice_site_threshold > 0:
                top_indices = top_indices.masked_fill(top_indices == sentinel, _PAD_VALUE)

            if pad_len > 0:
                pad = torch.full(
                    (batch, pad_len),
                    _PAD_VALUE,
                    dtype=top_indices.dtype,
                    device=top_indices.device,
                )
                top_indices = torch.cat((top_indices, pad), dim=1)

            positions.append(top_indices)

        return torch.stack(positions, dim=1)

    def _infer_splice_site_positions(self, splice_logits: Tensor) -> Tensor:
        probs = torch.softmax(splice_logits.float(), dim=-1)
        return self._infer_splice_site_positions_from_probs(probs)

    def embeddings(
        self,
        sequences: str | Sequence[str] | Tensor | np.ndarray,
        *,
        organism: str | None = None,
        organism_index: int | Tensor | None = None,
        no_grad: bool = True,
    ):
        tokens = _as_token_tensor(sequences).to(self.device)
        organism_index_tensor = self._resolve_organism_index(
            organism, organism_index, tokens.shape[0]
        )
        context = torch.no_grad if no_grad else torch.enable_grad
        with context():
            return self.model(tokens, organism_index_tensor, return_embeds=True)

    def predict(
        self,
        sequences: str | Sequence[str] | Tensor | np.ndarray,
        *,
        organism: str | None = None,
        organism_index: int | Tensor | None = None,
        include_splice_junctions: bool = True,
        splice_site_positions: Tensor | None = None,
        return_all_organisms: bool = False,
        no_grad: bool = True,
        requested_outputs: Iterable[str] | None = None,
        ontology_terms: Iterable[str] | None = None,
        **head_kwargs,
    ):
        tokens = _as_token_tensor(sequences).to(self.device)
        organism_index_tensor = self._resolve_organism_index(
            organism, organism_index, tokens.shape[0]
        )

        if splice_site_positions is None:
            splice_site_positions = head_kwargs.get('splice_site_positions')
        if splice_site_positions is not None and 'splice_site_positions' not in head_kwargs:
            head_kwargs = {**head_kwargs, 'splice_site_positions': splice_site_positions}

        # Convert requested_outputs to head names
        requested_heads = set(requested_outputs) if requested_outputs else None

        # If splice junctions are requested, ensure splice_sites_classification is also
        # included in the first pass (needed to infer splice site positions)
        if (
            include_splice_junctions
            and splice_site_positions is None
            and requested_heads is not None
            and 'splice_sites_junction' in requested_heads
        ):
            requested_heads = requested_heads | {'splice_sites_classification'}

        # Resolve ontology terms to track masks
        track_masks = self._resolve_track_masks(organism, requested_outputs, ontology_terms)

        def _inference():
            return self.model.inference(
                tokens,
                organism_index_tensor,
                requested_heads=requested_heads,
                track_masks=track_masks,
                **head_kwargs,
            )

        context = torch.no_grad if no_grad else torch.enable_grad
        with context():
            predictions = _inference()

            if (
                include_splice_junctions
                and splice_site_positions is None
                and isinstance(predictions, dict)
            ):
                organism_key = self._pick_organism_key(
                    predictions, organism, return_all_organisms
                )
                if organism_key is None:
                    raise ValueError(
                        'Multiple organisms present. Provide organism=... or '
                        'splice_site_positions=... when include_splice_junctions=True.'
                    )
                organism_preds = predictions.get(organism_key, {})
                splice_logits = organism_preds.get('splice_sites_classification')
                if splice_logits is None:
                    return predictions if return_all_organisms else organism_preds
                splice_site_positions = self._infer_splice_site_positions(splice_logits)
                # Add splice_site_positions to head_kwargs for the second inference pass
                head_kwargs = {**head_kwargs, 'splice_site_positions': splice_site_positions}
                predictions = self.model.inference(
                    tokens,
                    organism_index_tensor,
                    requested_heads=requested_heads,
                    track_masks=track_masks,
                    **head_kwargs,
                )

        if not isinstance(predictions, dict):
            return predictions

        organism_key = self._pick_organism_key(predictions, organism, return_all_organisms)
        if organism_key is None:
            return predictions
        if organism_key not in predictions:
            raise KeyError(
                f'Organism {organism_key!r} not found in model output. '
                f'Available: {list(predictions.keys())}'
            )
        return predictions[organism_key]

    def score_variant(
        self,
        seq_ref: str | Tensor | np.ndarray,
        seq_alt: str | Tensor | np.ndarray,
        *,
        organism: str | None = None,
        organism_index: int | Tensor | None = None,
        scorer,
        settings,
        variant,
        interval,
        track_metadata,
        no_grad: bool = True,
        requested_outputs: Iterable[str] | None = None,
        ontology_terms: Iterable[str] | None = None,
    ):
        seq_ref_tensor = _as_token_tensor(seq_ref).to(self.device)
        seq_alt_tensor = _as_token_tensor(seq_alt).to(self.device)
        organism_index_tensor = self._resolve_organism_index(
            organism, organism_index, seq_ref_tensor.shape[0]
        )

        # Convert requested_outputs to head names
        requested_heads = set(requested_outputs) if requested_outputs else None

        # Resolve ontology terms to track masks
        track_masks = self._resolve_track_masks(organism, requested_outputs, ontology_terms)

        context = torch.no_grad if no_grad else torch.enable_grad
        with context():
            return self.model.score_variant(
                seq_ref_tensor,
                seq_alt_tensor,
                organism_index_tensor,
                scorer=scorer,
                settings=settings,
                variant=variant,
                interval=interval,
                track_metadata=track_metadata,
                organism=organism,
                requested_heads=requested_heads,
                track_masks=track_masks,
            )

    def ism(
        self,
        sequence: str | Tensor | np.ndarray,
        *,
        output_key,
        organism: str | None = None,
        organism_index: int | Tensor | None = None,
        positions: Iterable[int] | None = None,
        alt_bases: str = 'ACGT',
        batch_size: int = 32,
        no_grad: bool = True,
        return_predictions: bool = False,
        requested_outputs: Iterable[str] | None = None,
        ontology_terms: Iterable[str] | None = None,
    ) -> dict[str, np.ndarray]:
        tokens = _as_token_tensor(sequence).to(self.device)
        if tokens.shape[0] != 1:
            raise ValueError('ISM currently supports a single sequence.')

        organism_index_tensor = self._resolve_organism_index(
            organism, organism_index, batch_size=1
        )

        output_name = _resolve_output_name(output_key)
        if output_name is None:
            raise ValueError('output_key must be a string or enum with a name attribute.')

        # Determine head name from output_key
        head_name = None
        for candidate, mapped in _HEAD_TO_OUTPUT_NAME.items():
            if mapped == output_name:
                head_name = candidate
                break

        # Convert requested_outputs to head names (include the output_key head)
        if requested_outputs is not None:
            requested_heads = set(requested_outputs)
        elif head_name is not None:
            # If no requested_outputs specified, only run the needed head for efficiency
            requested_heads = {head_name}
        else:
            requested_heads = None

        # Resolve ontology terms to track masks
        track_masks = self._resolve_track_masks(organism, requested_outputs, ontology_terms)

        context = torch.no_grad if no_grad else torch.enable_grad
        with context():
            ref_preds = self.model.inference(
                tokens,
                organism_index_tensor,
                requested_heads=requested_heads,
                track_masks=track_masks,
            )
            if isinstance(ref_preds, dict):
                organism_key = self._pick_organism_key(
                    ref_preds, organism, return_all_organisms=False
                )
                if organism_key is None:
                    raise ValueError('Provide organism=... when multiple organisms exist.')
                ref_preds = ref_preds[organism_key]

            if head_name is None or head_name not in ref_preds:
                raise KeyError(
                    f'Output {output_name!r} not found in model predictions.'
                )

            ref_output = _select_head_output(ref_preds[head_name], output_name)
            ref_output = ref_output.detach().cpu()

        if ref_output.ndim != 3:
            raise ValueError(
                'ISM currently supports 1D track outputs with shape [B, S, T].'
            )

        seq_len = tokens.shape[1]
        positions_list = list(positions) if positions is not None else list(range(seq_len))
        alt_bases_list = [b for b in alt_bases.upper() if b in _DNA_TO_INDEX]

        ref_tokens = tokens[0].cpu().numpy()
        ref_bases = _INDEX_TO_DNA[ref_tokens]

        mutations = []
        for pos in positions_list:
            if pos < 0 or pos >= seq_len:
                continue
            for base in alt_bases_list:
                if base == ref_bases[pos]:
                    continue
                mutations.append((pos, base))

        if not mutations:
            return {
                'positions': np.array([], dtype=np.int64),
                'alt_bases': np.array([], dtype='<U1'),
                'delta': np.zeros((0, ref_output.shape[-1]), dtype=np.float32),
            }

        deltas = []
        alt_preds_out = [] if return_predictions else None

        for start in range(0, len(mutations), batch_size):
            batch_mutations = mutations[start : start + batch_size]
            batch_tokens = tokens.repeat(len(batch_mutations), 1)
            for row, (pos, base) in enumerate(batch_mutations):
                batch_tokens[row, pos] = _DNA_TO_INDEX[base]

            with context():
                preds = self.model.inference(
                    batch_tokens,
                    organism_index_tensor.repeat(len(batch_mutations)),
                    requested_heads=requested_heads,
                    track_masks=track_masks,
                )
                if isinstance(preds, dict):
                    organism_key = self._pick_organism_key(
                        preds, organism, return_all_organisms=False
                    )
                    preds = preds[organism_key]

                output = _select_head_output(preds[head_name], output_name)

            output = output.detach().cpu()
            resolution = _OUTPUT_NAME_TO_RESOLUTION.get(output_name, 1)
            if resolution == 1:
                ref_at_pos = ref_output[0, [pos for pos, _ in batch_mutations], :]
                alt_at_pos = output[range(len(batch_mutations)), [pos for pos, _ in batch_mutations], :]
            else:
                bins = [pos // resolution for pos, _ in batch_mutations]
                ref_at_pos = ref_output[0, bins, :]
                alt_at_pos = output[range(len(batch_mutations)), bins, :]

            deltas.append((alt_at_pos - ref_at_pos).numpy())
            if return_predictions:
                alt_preds_out.append(alt_at_pos.numpy())

        delta = np.concatenate(deltas, axis=0)
        out = {
            'positions': np.array([pos for pos, _ in mutations], dtype=np.int64),
            'alt_bases': np.array([base for _, base in mutations]),
            'delta': delta,
        }
        if return_predictions and alt_preds_out is not None:
            out['predictions'] = np.concatenate(alt_preds_out, axis=0)
        return out

    def _get_fasta_extractor(
        self, organism: 'dna_model_types.Organism'
    ) -> 'fasta.FastaExtractor':
        """Returns the FastaExtractor for a given organism."""
        if extractor := self._fasta_extractors.get(organism):
            return extractor
        raise ValueError(f'FastaExtractor not found for {organism.name=}')

    def _convert_organism(
        self, organism: 'dna_model_types.Organism | str | None'
    ) -> tuple[str, 'dna_model_types.Organism']:
        """Converts organism to both string key and Organism enum."""
        from alphagenome.models import dna_model as dna_model_types

        if organism is None:
            return 'human', dna_model_types.Organism.HOMO_SAPIENS

        if isinstance(organism, str):
            organism_str = organism
            if organism.lower() in ('human', 'homo_sapiens'):
                organism_enum = dna_model_types.Organism.HOMO_SAPIENS
            elif organism.lower() in ('mouse', 'mus_musculus'):
                organism_enum = dna_model_types.Organism.MUS_MUSCULUS
            else:
                raise ValueError(f'Unknown organism string: {organism}')
        else:
            organism_enum = organism
            if organism == dna_model_types.Organism.HOMO_SAPIENS:
                organism_str = 'human'
            elif organism == dna_model_types.Organism.MUS_MUSCULUS:
                organism_str = 'mouse'
            else:
                raise ValueError(f'Unknown organism: {organism}')

        return organism_str, organism_enum

    def output_metadata(
        self, organism: 'dna_model_types.Organism | str | None' = None
    ) -> 'dna_output.OutputMetadata':
        """Get the metadata for a given organism.

        Args:
            organism: Organism to get metadata for. Defaults to human.

        Returns:
            OutputMetadata for the provided organism.
        """
        _, organism_enum = self._convert_organism(organism)
        return self._output_metadata_by_organism[organism_enum]

    def predict_interval(
        self,
        interval: 'genome.Interval',
        *,
        organism: 'dna_model_types.Organism | str | None' = None,
        requested_outputs: Iterable['dna_output.OutputType'],
        ontology_terms: Iterable[str] | None = None,
    ) -> 'dna_output.Output':
        """High-level interval prediction (JAX-compatible API).

        Args:
            interval: genome.Interval to predict.
            organism: 'human', 'mouse', or Organism enum. Defaults to human.
            requested_outputs: Set of OutputType enums (RNA_SEQ, DNASE, etc.).
            ontology_terms: Cell type/tissue ontology terms for filtering.

        Returns:
            dna_output.Output with TrackData attributes for visualization.
        """
        from alphagenome.data import ontology
        from alphagenome_research.model.metadata import metadata as metadata_lib

        organism_str, organism_enum = self._convert_organism(organism)

        # Parse ontology terms
        parsed_ontologies = None
        if ontology_terms is not None:
            parsed_ontologies = set(
                ontology.from_curie(o) if isinstance(o, str) else o
                for o in ontology_terms
            )

        # Get sequence from FASTA
        sequence = self._get_fasta_extractor(organism_enum).extract(interval)

        # Get metadata and create track masks
        track_metadata = self._metadata[organism_enum]
        track_masks = metadata_lib.create_track_masks(
            track_metadata,
            requested_outputs=set(requested_outputs),
            requested_ontologies=parsed_ontologies,
        )

        # Convert OutputType enums to head names for selective execution
        head_names = set()
        for output_type in requested_outputs:
            output_name = output_type.name if hasattr(output_type, 'name') else str(output_type)
            for head, mapped_name in _HEAD_TO_OUTPUT_NAME.items():
                if mapped_name == output_name:
                    head_names.add(head)
                    break

        # Run prediction with selective head execution
        # Note: Don't pass ontology_terms here - filtering happens in
        # _construct_output_from_predictions using track_masks
        predictions = self.predict(
            sequence,
            organism=organism_str,
            include_splice_junctions=True,
            no_grad=True,
            requested_outputs=head_names if head_names else None,
        )

        # Extract and filter predictions, then wrap in Output
        return _construct_output_from_predictions(
            predictions,
            track_masks=track_masks,
            metadata=track_metadata,
            interval=interval,
            negative_strand=interval.negative_strand,
        )

    def predict_variant(
        self,
        interval: 'genome.Interval',
        variant: 'genome.Variant',
        *,
        organism: 'dna_model_types.Organism | str | None' = None,
        requested_outputs: Iterable['dna_output.OutputType'],
        ontology_terms: Iterable[str] | None = None,
    ) -> 'dna_output.VariantOutput':
        """High-level variant prediction (JAX-compatible API).

        Args:
            interval: genome.Interval to predict.
            variant: genome.Variant to score.
            organism: 'human', 'mouse', or Organism enum. Defaults to human.
            requested_outputs: Set of OutputType enums (RNA_SEQ, DNASE, etc.).
            ontology_terms: Cell type/tissue ontology terms for filtering.

        Returns:
            dna_output.VariantOutput with reference and alternate Output.
        """
        from alphagenome.data import ontology
        from alphagenome.models import dna_output
        from alphagenome_research.io import genome as genome_io
        from alphagenome_research.model.metadata import metadata as metadata_lib

        organism_str, organism_enum = self._convert_organism(organism)

        # Parse ontology terms
        parsed_ontologies = None
        if ontology_terms is not None:
            parsed_ontologies = set(
                ontology.from_curie(o) if isinstance(o, str) else o
                for o in ontology_terms
            )

        # Get ref/alt sequences
        fasta_extractor = self._get_fasta_extractor(organism_enum)
        ref_sequence, alt_sequence = genome_io.extract_variant_sequences(
            interval, variant, fasta_extractor
        )

        # Get metadata and create track masks
        track_metadata = self._metadata[organism_enum]
        track_masks = metadata_lib.create_track_masks(
            track_metadata,
            requested_outputs=set(requested_outputs),
            requested_ontologies=parsed_ontologies,
        )

        splice_sites = None
        splice_site_extractor = self._splice_site_extractors.get(organism_enum)
        if splice_site_extractor is not None:
            splice_sites = splice_site_extractor.extract(interval)

        # Convert OutputType enums to head names for selective execution
        head_names = set()
        for output_type in requested_outputs:
            output_name = output_type.name if hasattr(output_type, 'name') else str(output_type)
            for head, mapped_name in _HEAD_TO_OUTPUT_NAME.items():
                if mapped_name == output_name:
                    head_names.add(head)
                    break

        splice_site_positions = None
        if head_names and 'splice_sites_junction' in head_names:
            splice_head = {'splice_sites_classification'}
            ref_splice = self.predict(
                ref_sequence,
                organism=organism_str,
                include_splice_junctions=False,
                no_grad=True,
                requested_outputs=splice_head,
            )
            alt_splice = self.predict(
                alt_sequence,
                organism=organism_str,
                include_splice_junctions=False,
                no_grad=True,
                requested_outputs=splice_head,
            )
            ref_logits = ref_splice.get('splice_sites_classification')
            alt_logits = alt_splice.get('splice_sites_classification')
            if ref_logits is not None and alt_logits is not None:
                ref_probs = torch.softmax(ref_logits.float(), dim=-1)
                alt_probs = torch.softmax(alt_logits.float(), dim=-1)
                merged_probs = torch.maximum(ref_probs, alt_probs)
                if splice_sites is not None:
                    splice_sites_tensor = torch.as_tensor(
                        splice_sites,
                        device=merged_probs.device,
                        dtype=merged_probs.dtype,
                    )
                    if splice_sites_tensor.ndim == 2:
                        splice_sites_tensor = splice_sites_tensor.unsqueeze(0)
                    if splice_sites_tensor.shape[0] != merged_probs.shape[0]:
                        splice_sites_tensor = splice_sites_tensor.expand(
                            merged_probs.shape[0], -1, -1
                        )
                    merged_probs = torch.maximum(merged_probs, splice_sites_tensor)
                splice_site_positions = self._infer_splice_site_positions_from_probs(
                    merged_probs
                )

        # Run predictions on both sequences with selective head execution
        # Note: Don't pass ontology_terms here - filtering happens in
        # _construct_output_from_predictions using track_masks
        ref_predictions = self.predict(
            ref_sequence,
            organism=organism_str,
            include_splice_junctions=True,
            no_grad=True,
            requested_outputs=head_names if head_names else None,
            splice_site_positions=splice_site_positions,
        )
        alt_predictions = self.predict(
            alt_sequence,
            organism=organism_str,
            include_splice_junctions=True,
            no_grad=True,
            requested_outputs=head_names if head_names else None,
            splice_site_positions=splice_site_positions,
        )

        # Wrap in VariantOutput
        return dna_output.VariantOutput(
            reference=_construct_output_from_predictions(
                ref_predictions,
                track_masks=track_masks,
                metadata=track_metadata,
                interval=interval,
                negative_strand=interval.negative_strand,
            ),
            alternate=_construct_output_from_predictions(
                alt_predictions,
                track_masks=track_masks,
                metadata=track_metadata,
                interval=interval,
                negative_strand=interval.negative_strand,
            ),
        )


def _extract_predictions(
    predictions: Mapping[str, object],
) -> Mapping['dna_output.OutputType', np.ndarray | Mapping[str, np.ndarray]]:
    """Extracts predictions from PyTorch model output to OutputType mapping."""
    from alphagenome.models import dna_output

    results = {}
    for output_type in dna_output.OutputType:
        prediction = None
        match output_type:
            case dna_output.OutputType.ATAC:
                if 'atac' in predictions:
                    pred = predictions['atac']
                    if isinstance(pred, dict) and 'scaled_predictions_1bp' in pred:
                        prediction = _to_numpy(pred['scaled_predictions_1bp'])
                    else:
                        prediction = _to_numpy(pred)
            case dna_output.OutputType.CAGE:
                if 'cage' in predictions:
                    pred = predictions['cage']
                    if isinstance(pred, dict) and 'scaled_predictions_1bp' in pred:
                        prediction = _to_numpy(pred['scaled_predictions_1bp'])
                    else:
                        prediction = _to_numpy(pred)
            case dna_output.OutputType.DNASE:
                if 'dnase' in predictions:
                    pred = predictions['dnase']
                    if isinstance(pred, dict) and 'scaled_predictions_1bp' in pred:
                        prediction = _to_numpy(pred['scaled_predictions_1bp'])
                    else:
                        prediction = _to_numpy(pred)
            case dna_output.OutputType.RNA_SEQ:
                if 'rna_seq' in predictions:
                    pred = predictions['rna_seq']
                    if isinstance(pred, dict) and 'scaled_predictions_1bp' in pred:
                        prediction = _to_numpy(pred['scaled_predictions_1bp'])
                    else:
                        prediction = _to_numpy(pred)
            case dna_output.OutputType.CHIP_HISTONE:
                if 'chip_histone' in predictions:
                    pred = predictions['chip_histone']
                    if isinstance(pred, dict) and 'scaled_predictions_128bp' in pred:
                        prediction = _to_numpy(pred['scaled_predictions_128bp'])
                    else:
                        prediction = _to_numpy(pred)
            case dna_output.OutputType.CHIP_TF:
                if 'chip_tf' in predictions:
                    pred = predictions['chip_tf']
                    if isinstance(pred, dict) and 'scaled_predictions_128bp' in pred:
                        prediction = _to_numpy(pred['scaled_predictions_128bp'])
                    else:
                        prediction = _to_numpy(pred)
            case dna_output.OutputType.SPLICE_SITES:
                if 'splice_sites_classification' in predictions:
                    pred = predictions['splice_sites_classification']
                    if isinstance(pred, dict) and 'predictions' in pred:
                        prediction = _to_numpy(pred['predictions'])
                    else:
                        prediction = _to_numpy(pred)
            case dna_output.OutputType.SPLICE_SITE_USAGE:
                if 'splice_sites_usage' in predictions:
                    pred = predictions['splice_sites_usage']
                    if isinstance(pred, dict) and 'predictions' in pred:
                        prediction = _to_numpy(pred['predictions'])
                    else:
                        prediction = _to_numpy(pred)
            case dna_output.OutputType.SPLICE_JUNCTIONS:
                if 'splice_sites_junction' in predictions:
                    pred = predictions['splice_sites_junction']
                    if isinstance(pred, dict):
                        prediction = {
                            'predictions': _to_numpy(pred.get('predictions')),
                            'splice_site_positions': _to_numpy(
                                pred.get('splice_site_positions')
                            ),
                        }
            case dna_output.OutputType.CONTACT_MAPS:
                if 'contact_maps' in predictions:
                    pred = predictions['contact_maps']
                    if isinstance(pred, dict) and 'predictions' in pred:
                        prediction = _to_numpy(pred['predictions'])
                    else:
                        prediction = _to_numpy(pred)
            case dna_output.OutputType.PROCAP:
                if 'procap' in predictions:
                    pred = predictions['procap']
                    if isinstance(pred, dict) and 'scaled_predictions_1bp' in pred:
                        prediction = _to_numpy(pred['scaled_predictions_1bp'])
                    else:
                        prediction = _to_numpy(pred)
        if prediction is not None:
            results[output_type] = prediction
    return results


def _to_numpy(x) -> np.ndarray | None:
    """Convert tensor to numpy array."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, Tensor):
        # Convert bfloat16 to float32 since numpy doesn't support bfloat16
        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _reverse_complement_predictions(
    predictions: Mapping['dna_output.OutputType', np.ndarray],
    strand_reindexing: Mapping['dna_output.OutputType', np.ndarray],
    sequence_length: int,
) -> Mapping['dna_output.OutputType', np.ndarray]:
    """Reverse complement predictions along positional and track axes."""
    from alphagenome.models import dna_output

    result = {}
    for output_type, pred in predictions.items():
        reindex = strand_reindexing.get(output_type)
        if reindex is None:
            result[output_type] = pred
            continue

        if output_type == dna_output.OutputType.SPLICE_JUNCTIONS:
            # Handle splice junctions separately
            result[output_type] = pred
        else:
            # Reverse positional axes and reindex tracks
            # Predictions have shape [B, S, T] or [B, S, S, T]
            if pred.ndim == 3:
                # [B, S, T] -> reverse S, reindex T
                pred_rc = pred[:, ::-1, :]
                pred_rc = pred_rc[..., reindex]
            elif pred.ndim == 4:
                # [B, S, S, T] -> reverse both S axes, reindex T
                pred_rc = pred[:, ::-1, ::-1, :]
                pred_rc = pred_rc[..., reindex]
            else:
                pred_rc = pred
            result[output_type] = pred_rc.copy()

    return result


def _construct_output_from_predictions(
    predictions: Mapping[str, object],
    *,
    track_masks: Mapping['dna_output.OutputType', np.ndarray],
    metadata: 'metadata_lib.AlphaGenomeOutputMetadata',
    interval: 'genome.Interval | None' = None,
    negative_strand: bool = False,
) -> 'dna_output.Output':
    """Constructs a dna_output.Output from PyTorch predictions."""
    from alphagenome.data import genome
    from alphagenome.data import junction_data
    from alphagenome.data import track_data
    from alphagenome.models import dna_output
    from alphagenome_research.model.variant_scoring import splice_junction

    def _unscale_track_predictions(
        prediction: np.ndarray,
        output_type: dna_output.OutputType,
        track_metadata: pd.DataFrame,
    ) -> np.ndarray:
        """Match JAX experimental-scale predictions for genome-track outputs."""
        if output_type not in {
            dna_output.OutputType.ATAC,
            dna_output.OutputType.CAGE,
            dna_output.OutputType.DNASE,
            dna_output.OutputType.RNA_SEQ,
            dna_output.OutputType.PROCAP,
            dna_output.OutputType.CHIP_HISTONE,
            dna_output.OutputType.CHIP_TF,
        }:
            return prediction

        if track_metadata is None or 'nonzero_mean' not in track_metadata.columns:
            return prediction

        track_means = track_metadata['nonzero_mean'].to_numpy(dtype=np.float32)
        if track_means.size == 0:
            return prediction

        pred = prediction.astype(np.float32, copy=False)

        # Inverse of JAX soft-clip used in predictions_scaling.
        soft_clip_value = 10.0
        pred = np.where(
            pred > soft_clip_value,
            (pred + soft_clip_value) ** 2 / (4 * soft_clip_value),
            pred,
        )

        if output_type == dna_output.OutputType.RNA_SEQ:
            pred = np.power(pred, 1.0 / 0.75)

        resolution = metadata.resolution(output_type)
        pred = pred * (track_means * resolution)
        return pred

    # Extract predictions to OutputType mapping
    extracted = _extract_predictions(predictions)

    # Apply reverse complement if on negative strand
    if negative_strand:
        extracted = _reverse_complement_predictions(
            extracted,
            metadata.strand_reindexing,
            sequence_length=interval.width if interval else 0,
        )

    def _convert_to_track_data(
        output_type: dna_output.OutputType,
    ) -> track_data.TrackData | None:
        track_metadata = metadata.get(output_type)
        prediction = extracted.get(output_type)
        if prediction is None or track_metadata is None:
            return None

        mask = track_masks.get(output_type)
        if mask is None:
            return None

        # Remove batch dimension if present
        if prediction.ndim >= 2 and prediction.shape[0] == 1:
            prediction = prediction[0]

        prediction = _unscale_track_predictions(prediction, output_type, track_metadata)

        # Filter by mask
        filtered_pred = prediction[..., mask]
        filtered_metadata = track_metadata[mask]

        return track_data.TrackData(
            values=filtered_pred.astype(np.float32),
            resolution=metadata.resolution(output_type),
            metadata=filtered_metadata,
            interval=interval,
        )

    def _convert_to_junction_data() -> junction_data.JunctionData | None:
        output_type = dna_output.OutputType.SPLICE_JUNCTIONS
        junction_metadata = metadata.get(output_type)
        splice_junction_predictions = extracted.get(output_type)
        if junction_metadata is None or splice_junction_predictions is None:
            return None

        mask = track_masks.get(output_type)
        if mask is None:
            return None

        splice_junctions = splice_junction_predictions.get('predictions')
        splice_site_positions = splice_junction_predictions.get('splice_site_positions')

        if splice_junctions is None or splice_site_positions is None:
            return None

        # Remove batch dimension
        if splice_junctions.ndim >= 2 and splice_junctions.shape[0] == 1:
            splice_junctions = splice_junctions[0]
        if splice_site_positions.ndim >= 2 and splice_site_positions.shape[0] == 1:
            splice_site_positions = splice_site_positions[0]

        junction_predictions, strands, starts, ends = (
            splice_junction.unstack_junction_predictions(
                splice_junctions,
                splice_site_positions,
                interval,
            )
        )

        chromosome = interval.chromosome if interval is not None else None
        junctions = [
            genome.Junction(chromosome, start, end, strand)
            for start, end, strand in zip(starts, ends, strands)
            if start < end
        ]

        # Filter by mask
        # Note: unstack_junction_predictions already separates strands, so
        # junction_predictions has shape [num_junctions, T] where T is the number
        # of tracks (not T*2). Apply mask directly without tiling.
        return junction_data.JunctionData(
            junctions=np.asarray(junctions),
            values=junction_predictions[..., mask],
            metadata=junction_metadata[mask],
            interval=interval,
        )

    return dna_output.Output(
        atac=_convert_to_track_data(dna_output.OutputType.ATAC),
        dnase=_convert_to_track_data(dna_output.OutputType.DNASE),
        procap=_convert_to_track_data(dna_output.OutputType.PROCAP),
        cage=_convert_to_track_data(dna_output.OutputType.CAGE),
        rna_seq=_convert_to_track_data(dna_output.OutputType.RNA_SEQ),
        chip_histone=_convert_to_track_data(dna_output.OutputType.CHIP_HISTONE),
        chip_tf=_convert_to_track_data(dna_output.OutputType.CHIP_TF),
        contact_maps=_convert_to_track_data(dna_output.OutputType.CONTACT_MAPS),
        splice_sites=_convert_to_track_data(dna_output.OutputType.SPLICE_SITES),
        splice_site_usage=_convert_to_track_data(dna_output.OutputType.SPLICE_SITE_USAGE),
        splice_junctions=_convert_to_junction_data(),
    )


def create_from_jax_model(
    jax_model,
    *,
    device: torch.device | str = 'cuda',
    strict: bool = False,
) -> DNAModel:
    """Create DNAModel from existing JAX AlphaGenomeModel with weight conversion.

    This converts JAX model weights to PyTorch and creates a DNAModel wrapper
    with all the necessary metadata and extractors for predict_interval/predict_variant.

    Args:
        jax_model: A JAX AlphaGenomeModel instance (from alphagenome_research.model.dna_model).
        device: PyTorch device to use ('cuda', 'cpu', or torch.device).
        strict: If True, require exact match when loading state_dict.

    Returns:
        DNAModel wrapping the converted PyTorch model.
    """
    from alphagenome.models import dna_model as dna_model_types
    from alphagenome.models import dna_output
    from alphagenome_pytorch.convert.convert_checkpoint import (
        convert_checkpoint,
        flatten_nested_dict,
    )
    from alphagenome_pytorch.alphagenome import set_update_running_var

    # Convert checkpoint
    flat_params = flatten_nested_dict(jax_model._params)
    flat_state = flatten_nested_dict(jax_model._state)
    state_dict = convert_checkpoint(flat_params, flat_state, verbose=False)

    # Create PyTorch model with reference heads
    model = AlphaGenome()
    for organism in jax_model._metadata.keys():
        if organism == dna_model_types.Organism.HOMO_SAPIENS:
            model.add_reference_heads('human')
        elif organism == dna_model_types.Organism.MUS_MUSCULUS:
            model.add_reference_heads('mouse')

    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    set_update_running_var(model, False)

    # Build output metadata by organism (without padding tracks)
    output_metadata_by_organism = {}
    for organism, organism_metadata in jax_model._metadata.items():
        masks = {k: ~v for k, v in organism_metadata.padding.items()}
        output_metadata = {
            output_type.name.lower(): m[masks[output_type]]
            for output_type in dna_output.OutputType
            if (m := organism_metadata.get(output_type)) is not None
        }
        output_metadata_by_organism[organism] = dna_output.OutputMetadata(
            **output_metadata
        )

    return DNAModel(
        model,
        device=device,
        fasta_extractors=jax_model._fasta_extractors,
        splice_site_extractors=getattr(jax_model, '_splice_site_extractors', None),
        metadata=jax_model._metadata,
        output_metadata_by_organism=output_metadata_by_organism,
    )


__all__ = [
    'ModelSettings',
    'DNAModel',
    'create',
    'create_from_jax_model',
]
