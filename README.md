<img src="./extended-figure-1.png" width="450px"></img>

## AlphaGenome (wip)

Implementation of [AlphaGenome](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/), Deepmind's updated genomic attention model

## Install

```bash
$ pip install alphagenome-pytorch
```

## Usage

```python
import torch
from alphagenome_pytorch import AlphaGenome

model = AlphaGenome()

dna = torch.randint(0, 5, (2, 8192))
organism_index = torch.tensor([0, 0], dtype=torch.long) # the organism that each sequence belongs to

pred_nucleotide, single, pairwise = model(dna, organism_index) # (2, 8192, 5), (2, 64, 1536), (2, 4, 4, 128)
print(pred_nucleotide.shape, single.shape, pairwise.shape)
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
@article{avsec2025alphagenome,
  title   = {AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model},
  author  = {Avsec, {\v{Z}}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drot{\'a}r, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
  year    = {2025}
}
```
