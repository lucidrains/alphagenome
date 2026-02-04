"""Tests for newly added modules: data, scoring, and evals."""

import pytest
import torch
import numpy as np


class TestDataModule:
    """Tests for alphagenome_pytorch.data module."""

    def test_bundle_name_enum(self):
        """Test BundleName enum values and methods."""
        from alphagenome_pytorch.data import BundleName

        # Test enum values exist
        assert BundleName.ATAC.value == 'atac'
        assert BundleName.RNA_SEQ.value == 'rna_seq'
        assert BundleName.CHIP_TF.value == 'chip_tf'
        assert BundleName.CONTACT_MAPS.value == 'contact_maps'

        # Test resolution method
        assert BundleName.ATAC.get_resolution() == 1
        assert BundleName.CHIP_TF.get_resolution() == 128
        assert BundleName.CONTACT_MAPS.get_resolution() == 2048

    def test_data_batch_creation(self):
        """Test DataBatch dataclass creation and methods."""
        from alphagenome_pytorch.data import DataBatch

        batch = DataBatch(
            dna_sequence=torch.randn(2, 1024, 4),
            organism_index=torch.tensor([0, 0]),
            atac=torch.randn(2, 1024, 64),
            atac_mask=torch.ones(2, 1024, 64, dtype=torch.bool),
        )

        assert batch.dna_sequence.shape == (2, 1024, 4)
        assert batch.organism_index.shape == (2,)

    def test_collate_batch(self):
        """Test collate_batch function."""
        from alphagenome_pytorch.data import collate_batch

        # collate_batch expects list of dicts (from dataset __getitem__)
        samples = [
            {
                'dna_sequence': torch.randn(1024, 4),
                'organism_index': torch.tensor(0),
                'atac': torch.randn(1024, 64),
                'atac_mask': torch.ones(1024, 64, dtype=torch.bool),
            },
            {
                'dna_sequence': torch.randn(1024, 4),
                'organism_index': torch.tensor(0),
                'atac': torch.randn(1024, 64),
                'atac_mask': torch.ones(1024, 64, dtype=torch.bool),
            },
        ]

        batch = collate_batch(samples)
        assert batch.dna_sequence.shape == (2, 1024, 4)
        assert batch.organism_index.shape == (2,)


class TestEvalsModule:
    """Tests for alphagenome_pytorch.evals module."""

    def test_pearsonr_state(self):
        """Test PearsonRState dataclass."""
        from alphagenome_pytorch.evals import PearsonRState

        state = PearsonRState(
            xy_sum=torch.tensor(10.0),
            x_sum=torch.tensor(5.0),
            xx_sum=torch.tensor(30.0),
            y_sum=torch.tensor(5.0),
            yy_sum=torch.tensor(30.0),
            count=torch.tensor(10.0),
        )

        # Test addition
        state2 = state + state
        assert state2.count.item() == 20.0
        assert state2.xy_sum.item() == 20.0

    def test_regression_state(self):
        """Test RegressionState dataclass."""
        from alphagenome_pytorch.evals import (
            RegressionState,
            initialize_regression_metrics,
        )

        state = initialize_regression_metrics()
        assert state.count.item() == 0.0
        assert state.sq_error.item() == 0.0

    def test_update_regression_metrics(self):
        """Test metric update function."""
        from alphagenome_pytorch.evals import (
            initialize_regression_metrics,
            update_regression_metrics,
            finalize_regression_metrics,
        )

        # Create correlated data
        y_true = torch.randn(4, 100, 10)  # batch, seq, tracks
        y_pred = y_true + 0.1 * torch.randn_like(y_true)  # Add noise

        state = update_regression_metrics(y_true, y_pred)

        # Finalize and check metrics
        metrics = finalize_regression_metrics(state)
        assert 'pearsonr' in metrics
        assert 'mse' in metrics
        assert 'mae' in metrics

        # Correlation should be high for similar data
        assert metrics['pearsonr'] > 0.9


class TestScoringModule:
    """Tests for alphagenome_pytorch.scoring module."""

    def test_output_type_enum(self):
        """Test OutputType enum."""
        from alphagenome_pytorch.scoring.variant_scoring import OutputType

        # OutputType uses auto() values, just verify they exist
        assert hasattr(OutputType, 'ATAC')
        assert hasattr(OutputType, 'CHIP_TF')
        assert hasattr(OutputType, 'CONTACT_MAPS')

    def test_get_resolution(self):
        """Test get_resolution function."""
        from alphagenome_pytorch.scoring import get_resolution
        from alphagenome_pytorch.scoring.variant_scoring import OutputType

        assert get_resolution(OutputType.ATAC) == 1
        assert get_resolution(OutputType.CHIP_TF) == 128
        assert get_resolution(OutputType.CONTACT_MAPS) == 2048

    def test_align_alternate_snp(self):
        """Test align_alternate for SNPs (no-op case)."""
        from alphagenome_pytorch.scoring import align_alternate
        from alphagenome_pytorch.scoring.variant_scoring import Interval, Variant

        # Create test data
        alt = torch.randn(1000, 10)
        variant = Variant(
            chromosome='chr1',
            position=500,
            reference_bases='A',
            alternate_bases='G',
        )
        interval = Interval(
            chromosome='chr1',
            start=0,
            end=1000,
        )

        # For SNPs, should return unchanged
        result = align_alternate(alt, variant, interval)
        assert torch.allclose(result, alt)

    def test_align_alternate_insertion(self):
        """Test align_alternate for insertions."""
        from alphagenome_pytorch.scoring import align_alternate
        from alphagenome_pytorch.scoring.variant_scoring import Interval, Variant

        # Create test data
        alt = torch.randn(1003, 10)  # 3 bp insertion
        variant = Variant(
            chromosome='chr1',
            position=500,
            reference_bases='A',
            alternate_bases='ATTT',  # 3 bp insertion
        )
        interval = Interval(
            chromosome='chr1',
            start=0,
            end=1003,  # Interval matches alt length for insertion
        )

        result = align_alternate(alt, variant, interval)
        # For insertions, align_alternate pools the inserted region and pads
        # The result length depends on the implementation - just verify it runs
        assert result.shape[1] == 10  # Tracks dimension preserved

    def test_create_center_mask(self):
        """Test create_center_mask function."""
        from alphagenome_pytorch.scoring import create_center_mask
        from alphagenome_pytorch.scoring.variant_scoring import Interval, Variant

        variant = Variant(
            chromosome='chr1',
            position=500,
            reference_bases='A',
            alternate_bases='G',
        )
        interval = Interval(
            chromosome='chr1',
            start=0,
            end=1000,
        )

        # Test with width
        mask = create_center_mask(interval, variant, width=100, resolution=1)
        assert mask.shape == (1000, 1)
        assert mask.dtype == bool
        assert mask.sum() == 100

        # Test full mask (no width)
        mask_full = create_center_mask(interval, variant, width=None, resolution=1)
        assert mask_full.sum() == 1000

    def test_variant_scorer_abc(self):
        """Test that VariantScorer is abstract."""
        from alphagenome_pytorch.scoring import VariantScorer

        with pytest.raises(TypeError):
            VariantScorer()  # Should fail - abstract class


class TestIntegration:
    """Integration tests for the new modules."""

    def test_evals_pipeline(self):
        """Test evaluation pipeline with mock predictions."""
        from alphagenome_pytorch.evals import (
            update_regression_metrics,
            finalize_regression_metrics,
            reduce_regression_metrics,
        )

        # Create mock model outputs (simulating what a model would produce)
        batch_size = 2
        seq_len = 1024
        num_tracks = 64

        # Simulate batched predictions vs targets
        predictions = torch.randn(batch_size, seq_len, num_tracks)
        targets = predictions + 0.1 * torch.randn_like(predictions)  # Similar with noise

        # Single batch update
        state = update_regression_metrics(targets, predictions)

        # Finalize and check metrics
        metrics = finalize_regression_metrics(state)
        assert 'pearsonr' in metrics
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'pearsonr_log1p' in metrics

        # Correlation should be high for similar data
        assert metrics['pearsonr'] > 0.9

        # Test reduction across batches
        state2 = update_regression_metrics(targets, predictions)
        combined = reduce_regression_metrics(state, state2)
        combined_metrics = finalize_regression_metrics(combined)
        assert combined_metrics['pearsonr'] > 0.9
