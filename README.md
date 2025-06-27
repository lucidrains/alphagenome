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
from alphagenome_pytorch import TransformerTower

transformer = TransformerTower(dim = 768, dim_pairwise = 128)

single = torch.randn(2, 512, 768)

attended_single, attended_pairwise = transformer(single)
```

## Citations

```bibtex
@article{avsec2025alphagenome,
  title   = {AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model},
  author  = {Avsec, {\v{Z}}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and Thomas, Raina and Dutordoir, Vincent and Perino, Matteo and De, Soham and Karollus, Alexander and Gayoso, Adam and Sargeant, Toby and Mottram, Anne and Wong, Lai Hong and Drot{\'a}r, Pavol and Kosiorek, Adam and Senior, Andrew and Tanburn, Richard and Applebaum, Taylor and Basu, Souradeep and Hassabis, Demis and Kohli, Pushmeet},
  year    = {2025}
}
```
