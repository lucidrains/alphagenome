import pytest

import torch
from alphagenome_pytorch.alphagenome import TransformerTower

def test_attention():
    transformer = TransformerTower(dim = 768, dim_pairwise = 128)

    single = torch.randn(2, 512, 768)

    single_repr, pairwise_repr = transformer(single)

    assert single_repr.shape == (2, 512, 768)
    assert pairwise_repr.shape == (2, 512 // 16, 512 // 16, 128)

def test_down_up():
    from alphagenome_pytorch.alphagenome import DownresBlock, UpresBlock
    down = DownresBlock(64)
    up = UpresBlock(64 + 128)

    x = torch.randn(1, 64, 8)
    assert up(down(x), x).shape == x.shape

def test_alphagenome():
    from alphagenome_pytorch import AlphaGenome

    model = AlphaGenome()

    dna = torch.randint(0, 5, (2, 8192))
    organism_index = torch.tensor([0, 0], dtype=torch.long)

    embeds_1bp, embeds_128bp, embeds_pair = model(dna, organism_index)

    embeds_1bp.sum().backward() # test backwards

    pred = embeds_1bp.argmax(dim = -1)

    assert pred.shape[1] == dna.shape[1]

@pytest.mark.parametrize('channel_first', (False, True))
def test_batchrmsnorm(channel_first):
    from alphagenome_pytorch.alphagenome import BatchRMSNorm

    rmsnorm = BatchRMSNorm(512, channel_first = channel_first)

    x = torch.randn(1, 512, 512)
    assert rmsnorm(x).shape == x.shape

def test_add_custom_head():
    from alphagenome_pytorch.alphagenome import AlphaGenome, TracksScaledPrediction

    model = AlphaGenome()

    model.add_head('human', 'pred_1bp_tracks', TracksScaledPrediction(1536, 10), 'embeds_1bp')

    dna = torch.randint(0, 5, (2, 8192))

    pred = model(dna, organism_index = 1) # (2, 8192)
    assert pred['human']['pred_1bp_tracks'].shape == (2, 8192, 10)

def test_target_scaler():
    from alphagenome_pytorch.alphagenome import TargetScaler

    scaler = TargetScaler([2., 3.])
    x = torch.randn(1, 3, 2)
    assert torch.allclose(scaler.inverse(scaler(x)), x)

def test_multinomial_loss():
    from alphagenome_pytorch.alphagenome import MultinomialLoss
    loss_fn = MultinomialLoss(128)

    x = torch.randn(1, 8192, 5)
    target = torch.randn(1, 8192, 5)
    loss = loss_fn(x, target)
    assert loss.numel() == 1

def test_output_heads():
    from alphagenome_pytorch.alphagenome import AlphaGenome

    model = AlphaGenome()

    model.add_heads(
        'human',
        num_tracks_1bp = 10,
        num_tracks_128bp = 10,
        num_tracks_contacts = 128,
        num_splicing_contexts = 64, # 2 strands x num. CURIE conditions
    )

    dna = torch.randint(0, 5, (2, 8192))

    organism_index = torch.tensor([0, 0]) # the organism that each sequence belongs to
    splice_donor_idx = torch.tensor([[10, 100, 34], [24, 546, 870]])
    splice_acceptor_idx = torch.tensor([[15, 103, 87], [56, 653, 900]])

    # get sequence embeddings

    embeddings_1bp, embeddings_128bp, embeddings_pair = model(dna, organism_index, return_embeds = True) # (2, 8192, 1536), (2, 64, 3072), (2, 4, 4, 128)

    # get track predictions

    out = model(
        dna,
        organism_index,
        splice_donor_idx = splice_donor_idx,
        splice_acceptor_idx = splice_acceptor_idx
    )

    for organism, outputs in out.items():
        for out_head, out_values in outputs.items():
            print(organism, out_head, out_values.shape)

    # human 1bp_tracks torch.Size([2, 8192, 10])
    # human 128bp_tracks torch.Size([2, 64, 10])
    # human contact_head torch.Size([2, 4, 4, 128])
    # human splice_probs torch.Size([2, 8192, 5])
    # human splice_usage torch.Size([2, 8192, 64])
    # human splice_juncs torch.Size([2, 3, 3, 64])
