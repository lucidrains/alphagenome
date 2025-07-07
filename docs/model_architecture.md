# BactaGenome Model Architecture

## Overview

BactaGenome is a bacterial genome modeling system based on AlphaGenome's transformer-unet architecture, specifically adapted for bacterial sequences and synthetic biology applications.

## Architecture Components

### 1. Core Backbone

The model uses AlphaGenome's proven architecture:

- **Transformer-UNet**: Multi-scale processing with encoder-decoder structure
- **Context Length**: 100K bp (98,304 bp exactly) for computational efficiency
- **Multi-resolution Embeddings**: 1bp, 128bp, and 2048bp (pairwise) resolutions

### 2. Input Processing

```
DNA Sequence (100K bp) → One-hot Encoding → DNA Embedding → Transformer-UNet
```

- **DNA Tokenization**: 5-class encoding (A, T, G, C, N)
- **Convolutional Embedding**: 15-width convolution for local pattern detection
- **Pooling**: Progressive downsampling through the UNet

### 3. Transformer Backbone

- **Depth**: 9 transformer layers
- **Attention Heads**: 8 heads with multi-query attention
- **Pairwise Attention**: Every 2 blocks for long-range interactions
- **Rotary Embeddings**: Position encoding for sequence relationships

### 4. Output Heads

#### Priority 1: Core Expression Control
1. **Promoter Strength Prediction**
   - Input: 1bp + 128bp embeddings
   - Output: Expression levels across conditions
   - Loss: Multinomial + Poisson (like AlphaGenome RNA-seq)

2. **RBS Translation Efficiency**
   - Input: 1bp embeddings
   - Output: Translation initiation rates
   - Loss: MSE on log-transformed efficiency ratios

3. **Operon Co-regulation**
   - Input: 128bp + pairwise embeddings
   - Output: Gene co-expression within operons
   - Loss: Correlation loss between genes in same operon

#### Priority 2: Advanced Regulation
4. **Riboswitch Ligand Binding**
   - Input: 1bp + 128bp embeddings
   - Output: Binding probability for different metabolites
   - Loss: Binary cross-entropy for ligand binding

5. **Small RNA Target Prediction**
   - Input: 1bp + 128bp embeddings
   - Output: Interaction strength with known sRNAs
   - Loss: Ranking loss for target prioritization

#### Priority 3: Systems-Level Features
6. **Transcription Termination**
   - Input: 1bp embeddings
   - Output: Termination probability and type (intrinsic vs Rho-dependent)

7. **Metabolic Pathway Activity**
   - Input: 128bp + pairwise embeddings
   - Output: Pathway completeness scores

8. **Protein Secretion Signals**
   - Input: 1bp embeddings
   - Output: Secretion system type predictions (T1SS, T2SS, etc.)

## Model Scaling

### Phase 1: Proof of Concept
- **Species**: E. coli K-12 only
- **Modalities**: 3 core modalities
- **Parameters**: ~100M parameters
- **Training**: 8,000 steps, batch size 16

### Phase 2: Multi-Species Expansion
- **Species**: 7 bacterial species
- **Modalities**: All 8 modalities
- **Parameters**: ~150M parameters
- **Training**: 12,000 steps, batch size 24

## Key Innovations

1. **Bacterial-Specific Outputs**: Unlike AlphaGenome's mammalian focus, all outputs are designed for bacterial biology and synthetic biology applications.

2. **Multi-Species Learning**: Joint training across phylogenetically diverse bacterial species to learn generalizable patterns.

3. **Synthetic Biology Focus**: Outputs directly applicable to iGEM competition and bioengineering projects.

4. **Efficient Context**: 10x smaller context window than AlphaGenome while maintaining full-length outputs.

## Performance Targets

- **Promoter prediction**: R² ≥ 0.75
- **RBS efficiency**: R² ≥ 0.80
- **Operon prediction**: AUROC ≥ 0.90
- **Cross-species transfer**: <20% performance drop

## Computational Requirements

- **Training Hardware**: A800 × 4-8 GPUs
- **Memory**: ~40GB GPU memory per replica
- **Training Time**: 2-3 days per phase
- **Inference**: Single GPU sufficient for most applications