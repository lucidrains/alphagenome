"""Tests for selective head execution and ontology filtering."""

import pytest
import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.alphagenome import Embeds, set_update_running_var


@pytest.fixture
def model():
    """Create a model with reference heads for testing."""
    model = AlphaGenome()
    model.add_reference_heads('human')
    model.eval()
    # Disable BatchRMSNorm running variance updates for deterministic results
    set_update_running_var(model, False)
    return model


@pytest.fixture
def sample_sequence():
    """Create a sample sequence for testing.

    Minimum sequence length required by the model is ~8192 for the transformer
    to have enough positions for relative position encoding.
    """
    seq_len = 8192
    return torch.randint(0, 4, (1, seq_len))


class TestInferenceMethod:
    """Tests for AlphaGenome.inference() method."""

    def test_inference_returns_dict(self, model, sample_sequence):
        """Test that inference returns a dictionary."""
        with torch.no_grad():
            result = model.inference(sample_sequence, organism_index=0)
        assert isinstance(result, dict)
        assert 'human' in result

    def test_inference_matches_forward(self, model, sample_sequence):
        """Test that inference produces same output as forward when no filtering.

        Note: inference() and forward() compute embeddings the same way, so
        their outputs should match for the standard heads.
        """
        with torch.no_grad():
            forward_out = model(sample_sequence, organism_index=0)
            inference_out = model.inference(sample_sequence, organism_index=0)

        # Compare outputs for each head
        for head_name in forward_out['human']:
            forward_head = forward_out['human'][head_name]
            inference_head = inference_out['human'][head_name]

            if isinstance(forward_head, torch.Tensor):
                torch.testing.assert_close(forward_head, inference_head)
            elif isinstance(forward_head, dict):
                for k in forward_head:
                    if isinstance(forward_head[k], torch.Tensor):
                        torch.testing.assert_close(forward_head[k], inference_head[k])

    def test_requested_heads_filtering(self, model, sample_sequence):
        """Test that only requested heads are executed."""
        requested = {'rna_seq', 'cage'}
        with torch.no_grad():
            result = model.inference(
                sample_sequence,
                organism_index=0,
                requested_heads=requested,
            )

        human_out = result['human']
        assert set(human_out.keys()) == requested

    def test_requested_heads_empty(self, model, sample_sequence):
        """Test with empty requested_heads set."""
        with torch.no_grad():
            result = model.inference(
                sample_sequence,
                organism_index=0,
                requested_heads=set(),
            )

        # Should return empty dict for organism
        assert result == {} or result.get('human', {}) == {}

    def test_track_masks_filtering(self, model, sample_sequence):
        """Test that track masks correctly filter output tracks."""
        # Create a mask that keeps only first 10 tracks
        num_tracks = 768  # rna_seq has 768 tracks
        mask = torch.zeros(num_tracks, dtype=torch.bool)
        mask[:10] = True

        with torch.no_grad():
            result = model.inference(
                sample_sequence,
                organism_index=0,
                requested_heads={'rna_seq'},
                track_masks={'rna_seq': mask},
            )

        rna_seq_out = result['human']['rna_seq']
        # GenomeTracksHead returns dict with resolution keys
        if isinstance(rna_seq_out, dict):
            for key, tensor in rna_seq_out.items():
                if isinstance(tensor, torch.Tensor):
                    assert tensor.shape[-1] == 10
        else:
            assert rna_seq_out.shape[-1] == 10

    def test_embedding_reuse(self, model, sample_sequence):
        """Test that precomputed embeddings can be reused."""
        # get_embeds requires tensor organism_index
        organism_index = torch.zeros(sample_sequence.shape[0], dtype=torch.long)

        with torch.no_grad():
            # Compute embeddings once
            embeds = model.get_embeds(sample_sequence, organism_index=organism_index)
            assert isinstance(embeds, Embeds)

            # Use cached embeddings
            result1 = model.inference(
                sample_sequence,
                organism_index=0,
                requested_heads={'rna_seq'},
                embeds=embeds,
            )

            # Compare with fresh computation
            result2 = model.inference(
                sample_sequence,
                organism_index=0,
                requested_heads={'rna_seq'},
            )

        # Results should match
        rna1 = result1['human']['rna_seq']
        rna2 = result2['human']['rna_seq']
        if isinstance(rna1, dict):
            for k in rna1:
                if isinstance(rna1[k], torch.Tensor):
                    torch.testing.assert_close(rna1[k], rna2[k])
        else:
            torch.testing.assert_close(rna1, rna2)

    def test_return_embeds(self, model, sample_sequence):
        """Test that return_embeds=True returns embeddings."""
        with torch.no_grad():
            result = model.inference(
                sample_sequence,
                organism_index=0,
                return_embeds=True,
            )

        assert isinstance(result, tuple)
        assert len(result) == 2
        predictions, embeds = result
        assert isinstance(predictions, dict)
        assert isinstance(embeds, Embeds)

    def test_int_organism_index(self, model, sample_sequence):
        """Test that integer organism_index works."""
        with torch.no_grad():
            result = model.inference(sample_sequence, organism_index=0)
        assert 'human' in result

    def test_tensor_organism_index(self, model, sample_sequence):
        """Test that tensor organism_index works."""
        organism_index = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            result = model.inference(sample_sequence, organism_index=organism_index)
        assert 'human' in result


class TestApplyTrackMask:
    """Tests for AlphaGenome._apply_track_mask() method."""

    def test_tensor_masking(self, model):
        """Test masking a tensor output."""
        tensor = torch.randn(1, 100, 20)  # [B, S, T]
        mask = torch.tensor([True, False] * 10, dtype=torch.bool)

        result = model._apply_track_mask(tensor, mask)
        assert result.shape == (1, 100, 10)

    def test_dict_masking(self, model):
        """Test masking a dict output."""
        output = {
            'scaled_predictions_1bp': torch.randn(1, 100, 20),
            'scaled_predictions_128bp': torch.randn(1, 10, 20),
        }
        mask = torch.tensor([True, False] * 10, dtype=torch.bool)

        result = model._apply_track_mask(output, mask)
        assert result['scaled_predictions_1bp'].shape == (1, 100, 10)
        assert result['scaled_predictions_128bp'].shape == (1, 10, 10)

    def test_non_tensor_passthrough(self, model):
        """Test that non-tensor values pass through unchanged."""
        value = "some_string"
        mask = torch.tensor([True, False], dtype=torch.bool)

        result = model._apply_track_mask(value, mask)
        assert result == value


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility with forward()."""

    def test_forward_unchanged(self, model, sample_sequence):
        """Test that forward() still works without new parameters."""
        with torch.no_grad():
            result = model(sample_sequence, organism_index=0)

        assert isinstance(result, dict)
        assert 'human' in result
        # Check that standard heads are present
        human_heads = result['human']
        assert 'rna_seq' in human_heads or 'cage' in human_heads

    def test_forward_return_embeds(self, model, sample_sequence):
        """Test that forward() with return_embeds still works."""
        with torch.no_grad():
            result = model(sample_sequence, organism_index=0, return_embeds=True)

        assert isinstance(result, Embeds)


class TestMemoryEfficiency:
    """Tests for memory efficiency of selective execution."""

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3,
        reason="Requires CUDA with >= 20GB memory"
    )
    def test_single_head_uses_less_memory(self, sample_sequence):
        """Test that running a single head uses less memory."""
        device = torch.device('cuda')
        model = AlphaGenome().to(device)
        model.add_reference_heads('human')
        model.eval()

        seq = sample_sequence.to(device)

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model.inference(seq, organism_index=0)
        all_heads_memory = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model.inference(
                seq, organism_index=0,
                requested_heads={'rna_seq'},
            )
        single_head_memory = torch.cuda.max_memory_allocated()

        # Single head should use less or equal memory
        # (may be equal due to embedding computation being the same)
        assert single_head_memory <= all_heads_memory


class TestDNAModelSelectiveInference:
    """Tests for DNAModel with selective head execution and ontology filtering."""

    @pytest.fixture
    def dna_model(self, model):
        """Create a DNAModel wrapper."""
        from alphagenome_pytorch.dna_model import DNAModel
        return DNAModel(model)

    def test_predict_with_requested_outputs(self, dna_model, sample_sequence):
        """Test DNAModel.predict() with requested_outputs parameter."""
        result = dna_model.predict(
            sample_sequence,
            organism='human',
            requested_outputs=['rna_seq', 'cage'],
        )
        assert set(result.keys()) == {'rna_seq', 'cage'}

    def test_predict_without_filtering(self, dna_model, sample_sequence):
        """Test DNAModel.predict() returns all heads when no filtering."""
        result = dna_model.predict(sample_sequence, organism='human')
        # Should have multiple heads
        assert len(result) > 2
        assert 'rna_seq' in result

    def test_ism_uses_inference(self, dna_model, sample_sequence):
        """Test that ism() uses inference() internally."""
        import numpy as np
        result = dna_model.ism(
            sample_sequence,
            output_key='rna_seq',
            organism='human',
            positions=[100, 101],
        )
        assert 'positions' in result
        assert 'delta' in result
        assert isinstance(result['positions'], np.ndarray)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
