from __future__ import annotations

import pytest
param = pytest.mark.parametrize

import torch
from torch_einops_utils.shape import is_shape, size

from alphagenome_pytorch.alphagenome import (
    AlphaGenome,
    BatchRMSNorm,
    DownresBlock,
    MultinomialCrossEntropy,
    MultinomialLoss,
    PoissonLoss,
    TargetScaler,
    TracksScaledPrediction,
    TransformerTower,
    UpresBlock,
)
from alphagenome_pytorch.alphagenome2 import (
    AlphaGenome2,
    BinomialLoss,
    DownresBlock as DownresBlockV2,
    JunctionsLoss as JunctionsLossV2,
    MultiHeadAttentionResidual,
    MultinomialCrossEntropy as MultinomialCrossEntropyV2,
    MultinomialLoss as MultinomialLossV2,
    PoissonLoss as PoissonLossV2,
    RMSNorm as RMSNormV2,
    SingleCellAutoencoder,
    TargetScaler as TargetScalerV2,
    TracksScaledPrediction as TracksScaledPredictionV2,
    TransformerTower as TransformerTowerV2,
    UpresBlock as UpresBlockV2,
    exists,
)

TEST_MODEL_KWARGS = dict(
    dims = (8, 9, 10, 11, 12, 13, 16),
    transformer_kwargs = dict(
        depth = 1,
        heads = 2,
        dim_head_qk = 32,
        dim_head_v = 32,
        dim_pairwise = 4,
        single_to_pairwise_heads = 2,
        pool_size = 4,
    ),
)

model_param = param('model_cls', (AlphaGenome, AlphaGenome2), ids = ('alphagenome', 'alphagenome2'))
transformer_param = param('transformer_cls', (TransformerTower, TransformerTowerV2), ids = ('alphagenome', 'alphagenome2'))

# shared unit tests

@transformer_param
@param('polar_pos_emb', (False, True))
def test_attention(transformer_cls, polar_pos_emb):
    transformer = transformer_cls(dim = 8, polar_pos_emb = polar_pos_emb, **TEST_MODEL_KWARGS['transformer_kwargs'])
    single = torch.randn(2, 32, 8)

    single_repr, pairwise_repr = transformer(single)

    assert is_shape(single_repr, '2 32 8')
    assert is_shape(pairwise_repr, '2 8 8 4')

@param('blocks', ((DownresBlock, UpresBlock), (DownresBlockV2, UpresBlockV2)), ids = ('alphagenome', 'alphagenome2'))
def test_down_up(blocks):
    down_cls, up_cls = blocks
    down = down_cls(8, channels_to_add = 1)
    up = up_cls(8 + 1, channels_to_remove = 1)

    x = torch.randn(1, 8, 8)
    assert up(down(x), x).shape == x.shape

@model_param
def test_alphagenome(model_cls):
    model = model_cls(**TEST_MODEL_KWARGS)

    dna = torch.randint(0, 4, (2, 512))
    organism_index = torch.tensor([0, 0], dtype = torch.long)

    embeds_1bp, embeds_128bp, embeds_pair = model(dna, organism_index)
    embeds_1bp.sum().backward()

    pred = embeds_1bp.argmax(dim = -1)
    assert size(pred, 'b [n]') == size(dna, 'b [n]')

@param('channel_first', (False, True))
def test_batchrmsnorm(channel_first):
    rmsnorm = BatchRMSNorm(64, channel_first = channel_first)
    x = torch.randn(1, 64, 64)
    assert rmsnorm(x).shape == x.shape

    v2_norm = RMSNormV2(64)
    assert v2_norm(x).shape == x.shape

@param('model_and_head', ((AlphaGenome, TracksScaledPrediction), (AlphaGenome2, TracksScaledPredictionV2)), ids = ('alphagenome', 'alphagenome2'))
def test_add_custom_head(model_and_head):
    model_cls, head_cls = model_and_head
    model = model_cls(**TEST_MODEL_KWARGS)
    model.add_head('human', 'pred_1bp_tracks', head_cls(model.dim_1bp, 10), 'embeds_1bp')

    dna = torch.randint(0, 4, (2, 512))
    pred = model(dna, organism_index = 1)
    assert is_shape(pred['human']['pred_1bp_tracks'], '2 512 10')

@param('scaler_cls', (TargetScaler, TargetScalerV2), ids = ('alphagenome', 'alphagenome2'))
def test_target_scaler(scaler_cls):
    scaler = scaler_cls([2., 3.])
    x = torch.randn(1, 3, 2)
    assert torch.allclose(scaler.inverse(scaler(x)), x)

@param('loss_classes', (
    (MultinomialLoss, MultinomialCrossEntropy, PoissonLoss),
    (MultinomialLossV2, MultinomialCrossEntropyV2, PoissonLossV2),
), ids = ('alphagenome', 'alphagenome2'))
def test_losses(loss_classes):
    multinomial_cls, ce_cls, poisson_cls = loss_classes
    pred = torch.rand(1, 512, 5) + 0.1
    target = torch.rand(1, 512, 5)

    for loss_fn in (multinomial_cls(128), ce_cls(), poisson_cls()):
        assert loss_fn(pred, target).numel() == 1

@model_param
def test_output_heads(model_cls):
    model = model_cls(**TEST_MODEL_KWARGS)
    model.add_heads(
        'human',
        num_tracks_1bp = 10,
        num_tracks_128bp = 10,
        num_tracks_contacts = 128,
        num_splicing_contexts = 64,
    )

    dna = torch.randint(0, 4, (2, 512))
    organism_index = torch.tensor([0, 0])
    splice_donor_idx = torch.tensor([[10, 100, 34], [24, 246, 370]])
    splice_acceptor_idx = torch.tensor([[15, 103, 87], [56, 253, 400]])

    embeds_1bp, _, _ = model(dna, organism_index, return_embeds = True)
    assert is_shape(embeds_1bp, '2 512 d', d = model.dim_1bp)

    out = model(dna, organism_index, splice_donor_idx = splice_donor_idx, splice_acceptor_idx = splice_acceptor_idx)
    assert is_shape(out['human']['1bp_tracks'], '2 512 10')
    assert is_shape(out['human']['splice_juncs'], '2 3 3 64')

# alphagenome 2 end-to-end tests

def test_alphagenome2_e2e():
    model = AlphaGenome2(
        dim_cond = 16,
        add_reference_heads = True,
        include_denoised_sc_gene_expr = True,
        **TEST_MODEL_KWARGS
    )
    assert len(model.shared_heads) == 15

    # DNA strings and soft one-hot inputs

    str_preds = model(['ACGT' * 128, 'TGCA' * 128], condition_index = 2, requested_heads = {'rna_seq'})
    assert is_shape(str_preds['rna_seq']['scaled_predictions_1bp'], '2 512 12')

    soft_dna = torch.randn(2, 512, 4, requires_grad = True)
    positions = torch.tensor([[[[2, 10, -1], [8, 30, -1]], [[5, 20, -1], [15, 45, -1]]]]).expand(2, -1, -1, -1)
    exon_mask = torch.zeros(2, 512, 2, 3)
    exon_mask[:, 10:30, 0, 0] = 1.   # gene 0 on + strand
    exon_mask[:, 100:140, 1, 1] = 1. # gene 1 on - strand
    attn_mask = torch.block_diag(torch.ones(2, 2, dtype = torch.bool), torch.ones(2, 2, dtype = torch.bool))

    preds = model(
        soft_dna.softmax(dim = -1),
        condition_index = 3,
        attn_mask = attn_mask,
        splice_site_positions = positions,
        gene_mask = exon_mask
    )

    assert len(preds) == 15
    assert is_shape(preds['sc_atac']['scaled_predictions_1bp'], '2 512 5')
    assert is_shape(preds['methylation']['predictions_128bp'], '2 4 2')
    assert is_shape(preds['single_cell_gene_expression']['predictions_1bp'], '2 3 14')
    assert (preds['denoised_single_cell_gene_expression']['predictions_1bp'][:, 0, 1] == 0).all()
    assert (preds['gene_expression']['predictions_1bp'][:, 2, :] == 0).all()

    # end-to-end AG2 losses + gradients

    junc_loss = JunctionsLossV2(scale_targets = True)(
        preds['splice_sites_junction']['predictions'],
        torch.rand(2, 3, 3, 6) * 10.,
        mask = preds['splice_sites_junction']['splice_junction_mask']
    )
    cpg_mask = torch.zeros(2, 512, 2, dtype = torch.bool)
    cpg_mask[:, :8, 0] = True
    binom_loss = BinomialLoss()(
        preds['methylation']['predictions_1bp'],
        torch.where(cpg_mask, 3., -1.),
        torch.where(cpg_mask, 5., -1.),
        mask = cpg_mask
    )

    (junc_loss + binom_loss + preds['denoised_single_cell_gene_expression']['predictions_1bp'].sum()).backward()
    assert soft_dna.grad.abs().sum() > 0
    assert model.shared_heads['splice_sites_junction'].scale.grad.abs().sum() > 0

def test_alphagenome2_stage4_autoencoder_and_conditioning():
    ae = SingleCellAutoencoder(num_genes = 64, dim = 32, dim_hidden = 64, dim_latent = 16)
    counts = torch.randint(0, 20, (2, 64)).float()
    gene_mask = (torch.arange(64) < 32).float()

    loss = ae(counts, gene_mask = gene_mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert ae.to_gene_logits.weight.grad[32:].abs().sum() > 0

    model = AlphaGenome2(
        dim_cond = 16,
        dim_cell_embed = 16,
        dim_adapter_hidden = 32,
        add_reference_heads = True,
        include_denoised_sc_gene_expr = True,
        **TEST_MODEL_KWARGS
    )
    dna = torch.randint(0, 4, (2, 512))

    # single-cell / population embeddings + stage 2/4 freeze

    z = ae.encode(counts, gene_mask = gene_mask).detach()
    assert is_shape(model(dna, cell_embed = z, return_embeds = True).embeds_1bp, '2 512 8')
    assert is_shape(model(dna, cell_embed = torch.randn(2, 4, 16), return_embeds = True).embeds_1bp, '2 512 8')

    model.freeze_for_condition_refinement_()
    assert model.conditioning_embedder.condition_emb.weight.requires_grad
    assert not model.shared_heads['atac'].resolutions['resolution_1'].to_pred.weight.requires_grad

    model.freeze_for_expression_post_training_()
    assert model.conditioning_embedder.cell_adapter[1].weight.requires_grad
    assert model.shared_heads['denoised_single_cell_gene_expression'].pool_1bp.to_pred.weight.requires_grad
    assert not model.conditioning_embedder.condition_emb.weight.requires_grad

def test_alphagenome2_dna_model_and_variant_scoring():
    pytest.importorskip('anndata')
    import numpy as np
    import pandas as pd

    from alphagenome_pytorch import DNAModel, DirectGeneExpressionScorer
    from alphagenome_pytorch.scoring import (
        CenterMaskVariantScorer,
        ContactMapScorer,
        GeneVariantScorer,
        SpliceJunctionVariantScorer,
    )
    from alphagenome_pytorch.scoring.center_mask import (
        AggregationType,
        CenterMaskScorer,
        Interval as CMInterval,
        OutputType as CMOutputType,
        Variant as CMVariant,
    )
    from alphagenome_pytorch.scoring.gene_mask import BaseVariantScorer, GeneMaskSettings
    from alphagenome_pytorch.scoring.variant_scoring import (
        Interval as VSInterval,
        OutputType as VSOutputType,
        Variant as VSVariant,
    )

    model = AlphaGenome2(dim_cond = 16, add_reference_heads = True, include_denoised_sc_gene_expr = True, **TEST_MODEL_KWARGS).eval()
    wrapper = DNAModel(model)

    dna_ref = torch.randint(0, 4, (1, 512))
    dna_alt = dna_ref.clone()
    dna_alt[0, 256] = (dna_alt[0, 256] + 1) % 4

    assert set(wrapper.predict(dna_ref, organism = 'human', requested_outputs = ['rna_seq', 'sc_atac'], condition_index = 2)) == {'rna_seq', 'sc_atac'}
    assert wrapper.ism(dna_ref, output_key = 'atac', organism = 'human', positions = [10, 11], condition_index = 2)['delta'].shape == (6, 2)

    # center-mask, gene-mask, direct expression, contact-map, splice-junction scoring

    sc_atac = wrapper.score_variant(
        dna_ref, dna_alt, organism = 'human', scorer = CenterMaskVariantScorer(),
        settings = CenterMaskScorer(CMOutputType.SC_ATAC, width = 129, aggregation_type = AggregationType.DIFF_LOG2_SUM),
        variant = CMVariant('chr1', 257, 'A', 'C'), interval = CMInterval('chr1', 0, 512),
        track_metadata = {}, requested_outputs = ['sc_atac'], condition_index = 2,
    )
    assert sc_atac.shape == (1, 5) and np.isfinite(sc_atac.X).all()

    vs_var, vs_int = VSVariant('chr1', 257, 'A', 'C'), VSInterval('chr1', 0, 512)

    class Extractor:
        def extract(self, interval, variant = None):
            m = np.zeros((interval.width, 1), dtype = bool)
            m[20:120, 0] = True
            return m, pd.DataFrame(dict(gene_id = ['G1'], Strand = ['+']))

    rna_lfc = wrapper.score_variant(
        dna_ref, dna_alt, organism = 'human', scorer = GeneVariantScorer(Extractor()),
        settings = GeneMaskSettings(BaseVariantScorer.GENE_MASK_LFC, VSOutputType.RNA_SEQ),
        variant = vs_var, interval = vs_int,
        track_metadata = {VSOutputType.RNA_SEQ: pd.DataFrame(dict(name = [f't{i}' for i in range(12)], strand = ['.'] * 12))},
        requested_outputs = ['rna_seq'], condition_index = 2,
    )
    assert rna_lfc.shape == (1, 12) and np.isfinite(rna_lfc.X).all()

    exon_mask = torch.zeros(1, 512, 2, 2)
    exon_mask[:, 20:80, 0, 0] = 1.
    direct_expr = wrapper.score_variant(
        dna_ref, dna_alt, organism = 'human', scorer = DirectGeneExpressionScorer(),
        settings = VSOutputType.DENOISED_SINGLE_CELL_GENE_EXPRESSION,
        variant = vs_var, interval = vs_int, track_metadata = {},
        requested_outputs = ['denoised_single_cell_gene_expression'], condition_index = 2, gene_mask = exon_mask,
    )
    assert direct_expr.shape == (2, 2) and direct_expr.X[0, 1] == 0. and (direct_expr.X[1] == 0.).all()

    dna_4k = torch.randint(0, 4, (1, 4096))
    dna_4k_alt = dna_4k.clone()
    dna_4k_alt[0, 1024] = (dna_4k_alt[0, 1024] + 1) % 4
    contact = wrapper.score_variant(
        dna_4k, dna_4k_alt, organism = 'human', scorer = ContactMapScorer(),
        settings = VSOutputType.CONTACT_MAPS, variant = VSVariant('chr1', 1025, 'A', 'C'),
        interval = VSInterval('chr1', 0, 4096), track_metadata = pd.DataFrame(dict(name = ['hic_0'])),
        requested_outputs = ['contact_maps'], condition_index = 2,
    )
    assert contact.shape == (1, 1) and np.isfinite(contact.X).all()

    gtf = pd.DataFrame(dict(Feature = ['gene'], Chromosome = ['chr1'], Start = [0], End = [500], Strand = ['+'], gene_id = ['G1'], gene_name = ['G1'], gene_type = ['protein_coding']))
    sj_pos = torch.tensor([[[[10, 40, -1], [120, 360, -1]], [[20, 80, -1], [150, 400, -1]]]])
    sj = wrapper.score_variant(
        dna_ref, dna_alt, organism = 'human', scorer = SpliceJunctionVariantScorer(gtf),
        settings = VSOutputType.SPLICE_JUNCTIONS, variant = vs_var, interval = vs_int,
        track_metadata = pd.DataFrame(dict(name = ['t0', 't1', 't2'], gtex_tissue = ['a', 'b', 'c'])),
        requested_outputs = ['splice_sites_junction'], condition_index = 2, splice_site_positions = sj_pos,
    )
    assert sj.shape == (1, 3) and np.isfinite(sj.X).all()

# multi-head attention residuals tests

@param('dim_lora', (8, None))
@param('heads', (1, 2, 4))
@param('channel_first', (False, True))
def test_multihead_attn_residual(dim_lora, heads, channel_first):
    attn_res = MultiHeadAttentionResidual(
        dim = 32,
        dim_ins = (16, 32, 64),
        heads = heads,
        dim_lora = dim_lora,
        channel_first = channel_first
    )

    if channel_first:
        x = torch.randn(2, 32, 32, requires_grad = True)
        history = [
            torch.randn(2, 16, 128, requires_grad = True),
            torch.randn(2, 32, 32, requires_grad = True),
            torch.randn(2, 64, 8, requires_grad = True),
        ]
    else:
        x = torch.randn(2, 32, 32, requires_grad = True)
        history = [
            torch.randn(2, 128, 16, requires_grad = True),
            torch.randn(2, 32, 32, requires_grad = True),
            torch.randn(2, 8, 64, requires_grad = True),
        ]

    out, attn = attn_res(x, history, return_attn = True)
    assert out.shape == x.shape and is_shape(attn, '2 4 h 32', h = heads)

    out.sum().backward()
    assert exists(x.grad) and x.grad.abs().sum() > 0
    assert all(exists(h.grad) and h.grad.abs().sum() > 0 for h in history)
    assert exists(attn_res.pseudo_query.grad) and attn_res.pseudo_query.grad.abs().sum() > 0

def test_multihead_attn_residual_uniform_init_and_empty_history():
    with pytest.raises(AssertionError):
        MultiHeadAttentionResidual(dim = 32, heads = 3)

    attn_res = MultiHeadAttentionResidual(dim = 32, dim_ins = (32, 32), heads = 4)

    x = torch.randn(2, 16, 32)
    h0, h1 = torch.randn(2, 16, 32), torch.randn(2, 16, 32)

    assert torch.allclose(attn_res(x, [h0, h1]), (x + h0 + h1) / 3, atol = 1e-5)

    empty_res = MultiHeadAttentionResidual(dim = 32)
    assert torch.equal(empty_res(x), x)
    assert torch.allclose(empty_res(x, []), x)

    arb_res = MultiHeadAttentionResidual(dim = 16, dim_ins = (8, 24), heads = 4, pooling = 'avg')
    out_arb = arb_res(torch.randn(2, 30, 16), [torch.randn(2, 50, 8), torch.randn(2, 12, 24)])
    assert is_shape(out_arb, '2 30 16')

def test_transformer_tower_v2_attn_res():
    tower = TransformerTowerV2(
        dim = 16,
        depth = 2,
        heads = 2,
        dim_head_qk = 16,
        dim_head_v = 16,
        dim_pairwise = 4,
        single_to_pairwise_heads = 2,
        pool_size = 4,
        use_attn_res = True,
        attn_res_kwargs = dict(heads = 4)
    )

    single = torch.randn(2, 32, 16, requires_grad = True)
    single_out, pair_out = tower(single)
    assert is_shape(single_out, '2 32 16') and is_shape(pair_out, '2 8 8 4')

    (single_out.sum() + pair_out.sum()).backward()
    assert all(exists(p.grad) for p in tower.parameters())

@param('use_attn_res', (False, True))
@param('attn_res_heads', (1, 2, 4, 8))
def test_alphagenome2_attn_res(use_attn_res, attn_res_heads):
    model = AlphaGenome2(
        dims = (8, 16, 24, 32),
        dim_cond = 16,
        transformer_kwargs = TEST_MODEL_KWARGS['transformer_kwargs'],
        use_attn_res = use_attn_res,
        attn_res_kwargs = dict(heads = attn_res_heads)
    )

    soft_dna = torch.randn(2, 64, 4, requires_grad = True)
    embeds_1bp, embeds_128bp, embeds_pair = model(soft_dna.softmax(dim = -1), condition_index = 1)

    assert is_shape(embeds_1bp, '2 64 8')
    assert is_shape(embeds_128bp, '2 4 32')
    assert is_shape(embeds_pair, '2 1 1 4')

    (embeds_1bp.sum() + embeds_128bp.sum() + embeds_pair.sum()).backward()
    assert exists(soft_dna.grad) and soft_dna.grad.abs().sum() > 0

    unet = model.transformer_unet

    if use_attn_res:
        assert unet.down_attn_res[0].pseudo_query.grad.abs().sum() > 0
        assert unet.up_attn_res[-1].pseudo_query.grad.abs().sum() > 0
        assert unet.transformer.final_attn_res.pseudo_query.grad.abs().sum() > 0
        assert unet.unpooled_to_tower[0].weight.grad.abs().sum() > 0
    else:
        assert not exists(unet.down_attn_res) and not exists(unet.up_attn_res)
        assert not exists(unet.transformer.final_attn_res)

    assert all(exists(p.grad) for p in unet.parameters() if p.requires_grad)

@param('replace_skips', (False, True))
@param('pooling', ('max', 'avg'))
def test_alphagenome2_attn_res_options(replace_skips, pooling):
    model = AlphaGenome2(
        dims = (8, 16, 24, 32),
        dim_cond = 16,
        transformer_kwargs = TEST_MODEL_KWARGS['transformer_kwargs'],
        use_attn_res = True,
        attn_res_kwargs = dict(heads = 4, pooling = pooling, replace_skips = replace_skips)
    )

    dna = torch.randint(0, 4, (2, 64))
    embeds_1bp, embeds_128bp, embeds_pair = model(dna, condition_index = 1)
    assert is_shape(embeds_1bp, '2 64 8') and is_shape(embeds_128bp, '2 4 32') and is_shape(embeds_pair, '2 1 1 4')

def test_alphagenome2_single_flag_attn_res():
    from alphagenome_pytorch import AlphaGenome2

    model = AlphaGenome2(
        dims = (8, 16, 24, 32),
        transformer_kwargs = TEST_MODEL_KWARGS['transformer_kwargs'],
        use_attn_res = True
    )

    dna = torch.randint(0, 4, (2, 64))
    embeds_1bp, embeds_128bp, embeds_pair = model(dna, organism_index = 0)

    assert is_shape(embeds_1bp, '2 64 8')
    assert is_shape(embeds_128bp, '2 4 32')
    assert is_shape(embeds_pair, '2 1 1 4')



