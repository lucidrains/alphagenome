<img src="./extended-figure-1.png" width="450px"></img>

## AlphaGenome (wip)

Implementation of [AlphaGenome](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/), Deepmind's updated genomic attention model


## Appreciation

- [Miquel Anglada-Girotto](https://github.com/MiqG) for contributing the organism, output embedding, loss functions, and all the splicing prediction heads!

## Install

```bash
$ pip install alphagenome-pytorch
```

## Usage

The main unet transformer, without any heads

```python
import torch
from alphagenome_pytorch import AlphaGenome

model = AlphaGenome()

dna = torch.randint(0, 5, (2, 8192))

# organism_index - 0 for human, 1 for mouse - can be changed with `num_organisms` on `AlphaGenome`

embeds_1bp, embeds_128bp, embeds_pair = model(dna, organism_index = 0) # (2, 8192, 1536), (2, 64, 3072), (2, 4, 4, 128)
```

Adding all types of output heads (thanks to [@MiqG](https://github.com/MiqG))

```python
import torch
from alphagenome_pytorch import AlphaGenome, publication_heads_config

model = AlphaGenome()

model.add_heads(
    'human',
    num_tracks_1bp = 10,
    num_tracks_128bp = 10,
    num_tracks_contacts = 128,
    num_splicing_contexts = 64, # 2 strands x num. CURIE conditions
)

dna = torch.randint(0, 5, (2, 8192))

organism_index = torch.tensor([0, 1]) # the organism that each sequence belongs to
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

# initialize published AlphaGenome for human and mouse
model = AlphaGenome()
model.add_heads(**publication_heads_config['human'])
model.add_heads(**publication_heads_config['mouse'])
model.total_parameters # 259,459,534 (vs ~450 million trainable parameters)
```

## Training

### test minimal architecture
```shell
# loss quickly decreases and stabilizes at around 1349651
# this minimal model (576,444 parameters) can be run with cpu

python train_dummy.py --config_file=configs/dummy.yaml
```

## Contributing

First install locally with the following

```bash
$ pip install '.[test]' # or uv pip install . '[test]'
```

Then make your changes, add a test to `tests/test_alphagenome.py`

```bash
$ pytest tests
```

That's it

Vibe coding with some attention network is totally welcomed, if it works

## Citations

```bibtex
@article {avsec2025alphagenome,
	title = {AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model},
	author = {Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drot{\'a}r, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
	elocation-id = {2025.06.25.661532},
	year = {2025},
	doi = {10.1101/2025.06.25.661532},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/06/27/2025.06.25.661532},
	eprint = {https://www.biorxiv.org/content/early/2025/06/27/2025.06.25.661532.full.pdf},
	journal = {bioRxiv}
}
```
