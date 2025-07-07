# BactaGenome Training Guide

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n bactagenome python=3.9
conda activate bactagenome

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Data Preparation

```bash
# Download raw data
python scripts/download_data.py --config configs/data/ecoli.yaml

# Preprocess data
python scripts/preprocess_data.py --config configs/data/ecoli.yaml --output-dir data/processed
```

### 3. Training

```bash
# Phase 1: Proof of concept (E. coli only)
python scripts/train.py --config configs/training/phase1.yaml --model-config configs/model/bacteria_base.yaml

# Phase 2: Multi-species training
python scripts/train.py --config configs/training/phase2.yaml --model-config configs/model/bacteria_base.yaml
```

## Training Phases

### Phase 1: Proof of Concept (Weeks 1-8)

**Objective**: Validate core bacterial modalities on E. coli

**Configuration**:
- Species: E. coli K-12 only
- Modalities: Promoter strength, RBS efficiency, operon co-regulation
- Context: 100K bp
- Batch size: 16
- Learning rate: 0.002
- Steps: 8,000

**Success Criteria**:
- Promoter/RBS prediction R² > 0.7
- Stable training convergence
- Training pipeline validation

**Resources**:
- Hardware: A800 × 4
- Time: ~2 days
- Budget: $3,000-5,000

### Phase 2: Multi-Species Expansion (Weeks 9-12)

**Objective**: Scale to all bacterial species and modalities

**Configuration**:
- Species: All 7 bacterial species
- Modalities: All 8 bacterial modalities
- Context: 100K bp
- Batch size: 24
- Learning rate: 0.003
- Steps: 12,000

**Key Features**:
- Cross-species learning
- Advanced regulatory modalities
- Sequence parallelism for efficiency

**Resources**:
- Hardware: A800 × 8
- Time: ~3 days
- Budget: $8,000-12,000

## Cross-Validation Strategy

### Chromosome-Based Splitting

Bacterial genomes are circular, so we split by genomic coordinates:

```python
folds = {
    "fold_0": "ori_to_90_degrees",    # Origin to 90°
    "fold_1": "90_to_180_degrees",    # 90° to 180°
    "fold_2": "180_to_270_degrees",   # 180° to 270°
    "fold_3": "270_to_ori_degrees"    # 270° to origin
}
```

This ensures:
- No data leakage between folds
- Balanced representation of genome features
- Realistic evaluation of generalization

## Optimization Strategy

### Learning Rates
- **Backbone**: 1e-4 (pre-trained transformer)
- **Heads**: 1e-3 (new bacterial-specific layers)

### Scheduling
- **Warmup**: 10% of total steps
- **Decay**: Cosine annealing to 10% of peak
- **Gradient clipping**: 1.0 for stability

### Regularization
- **Weight decay**: 0.01
- **Dropout**: 0.1 in transformer
- **Data augmentation**: 1% random mutations

## Loss Function Design

### Multi-Task Learning

Each modality has a specific loss function:

```python
losses = {
    'promoter_strength': MultinomialLoss() + PoissonLoss(),
    'rbs_efficiency': MSELoss(log_transform=True),
    'operon_coregulation': CorrelationLoss(),
    'riboswitch_binding': BCELoss(),
    'srna_target': RankingLoss(),
    'termination': CrossEntropyLoss(),
    'pathway_activity': BCELoss(),
    'secretion_signal': BCELoss()
}
```

### Loss Weighting

Priority-based weighting:
- **Core modalities** (promoter, RBS, operon): 1.0
- **Advanced modalities**: 0.5
- **Systems-level modalities**: 0.5

## Monitoring and Validation

### Key Metrics

**Regression Tasks**:
- R² coefficient
- Mean Absolute Error
- Pearson correlation

**Classification Tasks**:
- AUROC
- AUPRC
- F1 score

**Cross-Species**:
- Performance drop across species
- Transfer learning effectiveness

### Logging

- **Weights & Biases**: Experiment tracking
- **TensorBoard**: Real-time metrics
- **Model checkpoints**: Every 1000 steps
- **Validation**: Every 500 steps

## Troubleshooting

### Common Issues

**Memory Problems**:
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision training

**Convergence Issues**:
- Lower learning rate
- Increase warmup steps
- Check data preprocessing

**Poor Cross-Species Transfer**:
- Increase species diversity in training
- Adjust loss weighting
- Use domain adaptation techniques

### Performance Optimization

**Speed**:
- Use sequence parallelism for long sequences
- Enable mixed precision (FP16)
- Optimize data loading pipeline

**Quality**:
- Increase model depth/width
- Add more training data
- Improve data augmentation

## Advanced Configuration

### Custom Head Addition

```python
from bactagenome import BactaGenome
from bactagenome.model.heads import CustomBacterialHead

model = BactaGenome(config)

# Add custom head
custom_head = CustomBacterialHead(dim_1bp=512, output_dim=100)
model.add_head('E_coli_K12', 'custom_prediction', custom_head)
```

### Transfer Learning

```python
# Load pre-trained weights
model.load_pretrained_backbone('path/to/alphagenome.pt')

# Freeze backbone for fine-tuning
model.freeze_backbone(freeze=True)
```

### Distributed Training

```bash
# Multi-GPU training
torchrun --nproc_per_node=8 scripts/train.py --config configs/training/phase2.yaml
```

## Evaluation

### Model Validation

```bash
# Evaluate on test set
python scripts/evaluate.py --model-path models/best/phase2.pt --config configs/model/bacteria_base.yaml --data-config configs/data/ecoli.yaml --plot
```

### Biological Validation

- Compare with experimental data
- Test on held-out species
- Validate on synthetic constructs
- iGEM competition benchmarks