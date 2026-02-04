<img src="./extended-figure-1.png" width="450px"></img>

## AlphaGenome

Implementation of [AlphaGenome](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/), Deepmind's updated genomic attention model

The [official code](https://github.com/google-deepmind/alphagenome_research) has been released!

## Appreciation

- [Miquel Anglada-Girotto](https://github.com/MiqG) for contributing the organism, output embedding, loss functions, and all the splicing prediction heads!

- [Xinming Tu](https://github.com/XinmingTu) for aligning the architecture with the official JAX implementation, enabling pretrained weight compatibility, TFRecord dataset support, scoring and evaluation modules, and a test suite!

## Install

```bash
pip install alphagenome-pytorch
```

Or with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install alphagenome-pytorch
```

### Environment Setup

To use pretrained weights from HuggingFace, create a `.env` file with your credentials:

```bash
cp .env.template .env
# Edit .env and add your HF_TOKEN
```

Required environment variables:
- `HF_TOKEN`: Your HuggingFace API token ([get one here](https://huggingface.co/settings/tokens))

## Usage

The main unet transformer, without any heads

```python
import torch
from alphagenome_pytorch import AlphaGenome

model = AlphaGenome()

dna = torch.randint(0, 4, (2, 8192))

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

dna = torch.randint(0, 4, (2, 8192))

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
# human splice_logits torch.Size([2, 8192, 5])
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

## Loading Pretrained Weights

### Installation requirements

To use the conversion utilities, you will need to install the `alphagenome-research` package manually from GitHub

```bash
uv pip install git+https://github.com/google-deepmind/alphagenome_research.git
```

### Loading and Converting

```python
from alphagenome_pytorch import AlphaGenome

# Load AlphaGenome with official JAX weights
model = AlphaGenome()
model.add_reference_heads("human")
model.load_from_official_jax_model("all_folds")

model.eval()
```

## Contributing

### Development Setup

```bash
# Clone and install with dev dependencies using uv
git clone https://github.com/lucidrains/alphagenome.git
cd alphagenome
uv pip install -e '.[test,convert]'

# Set up environment variables
cp .env.template .env
# Edit .env and add your HF_TOKEN
```

### Running Tests

```bash
# Run unit tests
uv run pytest tests/test_alphagenome.py -v

# Run regression tests (requires HF access)
ALPHAGENOME_RUN_INTEGRATION_TESTS=1 uv run pytest tests/test_regression.py -v

# Run full integration tests
ALPHAGENOME_RUN_INTEGRATION_TESTS=1 uv run pytest tests/test_integration_jax_torch.py -v
```

### Regenerating Regression Data

If you modify the model architecture, regenerate the reference tensors:

```bash
uv run python tests/generate_regression_tensors.py
```

That's it. Vibe coding with some attention network is totally welcomed, if it works

## Citations

```bibtex
@article {avsec2025alphagenome,
    title   = {AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model},
    author  = {Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drot{\'a}r, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
    elocation-id = {2025.06.25.661532},
    year    = {2025},
    doi     = {10.1101/2025.06.25.661532},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2025/06/27/2025.06.25.661532},
    eprint  = {https://www.biorxiv.org/content/early/2025/06/27/2025.06.25.661532.full.pdf},
    journal = {bioRxiv}
}
```

```bibtex
@misc{gopalakrishnan2025decouplingwhatwherepolar,
    title   = {Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings}, 
    author  = {Anand Gopalakrishnan and Robert Csordás and Jürgen Schmidhuber and Michael C. Mozer},
    year    = {2025},
    eprint  = {2509.10534},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2509.10534}, 
}
```
