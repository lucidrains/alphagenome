# BactaGenome Configuration Guide

This document explains the configuration structure and how different config files work together in the BactaGenome project.

## Configuration Overview

BactaGenome uses a multi-layered configuration system with different types of config files serving different purposes:

```
configs/
├── training/                    # Training-specific configurations
│   ├── dummy.yaml              # Dummy/test training config
│   ├── phase1.yaml             # Phase 1 training config
│   └── phase2.yaml             # Phase 2 training config
├── data/                       # Data processing configurations
│   ├── ecoli.yaml              # E. coli specific data config
│   └── multi_species.yaml     # Multi-species data config
└── publication.yaml            # Research publication settings
```

## 1. Training Configurations (`configs/training/`)

These files define how to train the model, including hyperparameters, training schedule, and experiment settings.

### `dummy.yaml` - Test/Development Training
**Purpose**: Quick testing and development with synthetic data

```yaml
# Model architecture
model:
  dims: [768, 896, 1024, 1152, 1280, 1408, 1536]  # U-Net channel dimensions
  context_length: 98304                             # 100K bp sequences
  num_organisms: 1                                  # Phase 1: E. coli only
  
  # Transformer hyperparameters (must match TransformerTower.__init__)
  transformer_kwargs:
    depth: 9                                        # Number of transformer layers
    heads: 8                                        # Attention heads
    dim_head_qk: 128                               # Query/Key head dimension
    dim_head_v: 192                                # Value head dimension
    dropout: 0.1                                   # Dropout rate
    max_positions: 8192                            # Max sequence positions

# Training hyperparameters
training:
  epochs: 10
  batch_size: 2                                    # Small for testing
  learning_rate: 1e-4
  loss_weights:                                    # Phase 1 modalities only
    promoter_strength: 1.0
    rbs_efficiency: 1.0
    operon_coregulation: 1.0

# Data settings
data:
  seq_len: 98304                                   # Must match model.context_length
  num_train_samples: 1000                          # Synthetic data size
  num_val_samples: 200

# Target organisms
organisms:
  - "E_coli_K12"                                   # Phase 1: single organism
```

### `phase1.yaml` - Proof of Concept Training
**Purpose**: Real Phase 1 training (3 modalities, E. coli only)

```yaml
# Higher performance settings for real training
training:
  epochs: 100
  batch_size: 16                                   # Larger batch for real training
  learning_rate: 0.002                            # Higher learning rate
  total_steps: 8000                               # From plan.md

# Only Phase 1 modalities
organisms:
  - "E_coli_K12"
```

### `phase2.yaml` - Multi-Species Training
**Purpose**: Phase 2 training (all 8 modalities, 7 species)

```yaml
# Full multi-species setup
model:
  num_organisms: 7                                # All 7 bacterial species

training:
  batch_size: 24                                  # Larger batch for multi-species
  learning_rate: 0.003
  total_steps: 12000

# All modalities enabled
training:
  loss_weights:
    promoter_strength: 1.0
    rbs_efficiency: 1.0
    operon_coregulation: 1.0
    riboswitch_binding: 0.8                       # Phase 2 additions
    srna_targets: 0.8
    transcription_termination: 1.0
    pathway_activity: 1.0
    secretion_signals: 1.0

# All organisms
organisms:
  - "E_coli_K12"
  - "B_subtilis_168"
  - "Salmonella_enterica"
  - "Pseudomonas_aeruginosa"
  - "Mycobacterium_tuberculosis"
  - "Streptococcus_pyogenes"
  - "Synechocystis_sp"
```

## 2. Model Configuration Class (`bactagenome/model/config.py`)

**Purpose**: Defines the model architecture and default parameters

```python
class BactaGenomeConfig:
    def __init__(
        self,
        dims=(768, 896, 1024, 1152, 1280, 1408, 1536),    # U-Net dimensions
        context_length=98304,                               # 100K bp context
        num_organisms=7,                                    # 7 bacterial species
        transformer_kwargs=None,                            # Transformer parameters
        head_specs=None,                                    # Organism-specific heads
    ):
```

**Key Features**:
- `get_phase1_config()`: Returns Phase 1 specific settings
- `get_phase2_config()`: Returns Phase 2 specific settings  
- `get_head_spec(organism)`: Gets organism-specific head configurations
- `head_specs`: Defines organism-specific parameters (e.g., number of pathways, sRNAs)

## 3. Data Configurations (`configs/data/`)

**Purpose**: Define how to load and process genomic data

### `ecoli.yaml` - E. coli Data Processing
```yaml
organism: "E_coli_K12"
genome_path: "/path/to/ecoli/genome.fasta"
annotations_path: "/path/to/ecoli/annotations.gff"
window_size: 98304
overlap: 0.1
validation_split: 0.2
```

### `multi_species.yaml` - Multi-Species Data
```yaml
species:
  E_coli_K12:
    genome_path: "/path/to/ecoli/"
    weight: 1.0                                    # Sampling weight
  B_subtilis_168:
    genome_path: "/path/to/bacillus/"
    weight: 0.8
  # ... other species
```

## 4. Configuration Usage Patterns

### Loading Configurations in Code

```python
# Training script
config = load_config(args.config)                 # Load YAML training config
model_config = BactaGenomeConfig(                 # Create model config
    dims=tuple(config['model']['dims']),
    context_length=config['model']['context_length'],
    transformer_kwargs=config['model']['transformer_kwargs']
)
model = BactaGenome(model_config)
```

### Config File Relationships

```
train_dummy.py
    ↓ loads
configs/training/dummy.yaml                        # Training hyperparameters
    ↓ references
bactagenome/model/config.py                        # Model architecture defaults
    ↓ used by
bactagenome/model/core.py                          # Model implementation
```

## 5. Configuration Best Practices

### DO:
- ✅ Use `dummy.yaml` for quick testing
- ✅ Use `phase1.yaml` for proof-of-concept training
- ✅ Match `data.seq_len` with `model.context_length`
- ✅ Use correct transformer parameter names (`dim_head_qk`, `dim_head_v`)
- ✅ Start with Phase 1 modalities before adding Phase 2/3

### DON'T:
- ❌ Use `dim_head` (incorrect parameter name)
- ❌ Mix Phase 1 and Phase 2 modalities without planning
- ❌ Set batch_size too high for testing
- ❌ Forget to update `num_organisms` when changing organism list

## 6. Common Configuration Errors

### Error: `TransformerTower.__init__() got an unexpected keyword argument 'dim_head'`
**Solution**: Use `dim_head_qk` and `dim_head_v` instead of `dim_head`

```yaml
# WRONG
transformer_kwargs:
  dim_head: 64

# CORRECT  
transformer_kwargs:
  dim_head_qk: 128
  dim_head_v: 192
```

### Error: Mismatched sequence lengths
**Solution**: Ensure data and model configs match

```yaml
# These must be equal
data:
  seq_len: 98304
model:
  context_length: 98304
```

## 7. Configuration Validation

The system validates configurations at runtime:

```python
# Model validates transformer parameters
assert 'dim_head_qk' in transformer_kwargs
assert 'dim_head_v' in transformer_kwargs

# Training validates organism-modality compatibility
assert all(org in config['organisms'] for org in heads_cfg.keys())
```

## 8. Quick Start Recommendations

1. **Testing**: Use `configs/training/dummy.yaml`
2. **Development**: Modify `dummy.yaml` for your needs
3. **Phase 1 Training**: Use `configs/training/phase1.yaml`
4. **Production**: Create custom configs based on templates

This layered configuration system allows flexible experimentation while maintaining consistency across different training phases and data sources.