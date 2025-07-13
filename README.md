# BactaGenome

**Bacterial Genome Modeling with AlphaGenome Architecture**

BactaGenome is a bacterial-specific adaptation of AlphaGenome's transformer-unet architecture, designed for synthetic biology applications and optimized for the iGEM competition.

## 🧬 Overview

BactaGenome adapts AlphaGenome's proven architecture for bacterial sequences, featuring:

- **100K bp context window** (10x smaller than AlphaGenome for efficiency)
- **8 bacterial-specific prediction modalities** for synthetic biology
- **Multi-species learning** across 7 phylogenetically diverse bacteria
- **Full-length outputs** for precise genomic engineering

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/bactagenome.git
cd bactagenome

# Create conda environment
conda create -n bactagenome python=3.9
conda activate bactagenome

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from bactagenome import BactaGenome, BactaGenomeConfig

# Create model
config = BactaGenomeConfig()
model = BactaGenome(config)

# Add bacterial prediction heads
model.add_bacterial_heads("E_coli_K12")

# Predict on DNA sequence
import torch
sequence = torch.randint(0, 5, (1, 98304))  # 100K bp
organism_index = torch.tensor([0])  # E. coli

predictions = model(sequence, organism_index)
print(predictions.keys())  # Dict of predictions by organism and modality
```

### Training

```bash
# Phase 1: Proof of concept (E. coli only)
python scripts/train.py --config configs/training/phase1.yaml

# Phase 2: Multi-species training  
python scripts/train.py --config configs/training/phase2.yaml
```

## 🧪 Prediction Modalities

> **Note**: BactaGenome's output modalities have evolved from theoretical targets to data-driven implementations. See [OUTPUT_MODALITIES.md](docs/OUTPUT_MODALITIES.md) for detailed explanation.

### Current Implementation (Phase 1 - Data-Driven)
1. **Gene Expression**: Log-normalized TPM/FPKM values from real RNA-seq data
2. **Gene Density**: Count of genes per 128bp genomic bin  
3. **Operon Membership**: Binary classification of operon participation

### Planned Evolution (Phases 2-3)

#### Advanced Expression Control
4. **Multi-condition Expression**: Condition-specific gene expression profiles
5. **Promoter Element Detection**: Sequence motif and regulatory element prediction
6. **RBS Efficiency**: Quantitative translation initiation prediction

#### Regulatory Networks
7. **Operon Co-regulation**: Multi-track co-expression pattern modeling
8. **Riboswitch Binding**: Ligand-specific riboswitch interaction prediction
9. **sRNA Targets**: Small RNA target interaction prediction

#### Systems-Level Features  
10. **Transcription Termination**: Termination mechanism classification
11. **Pathway Activity**: Metabolic pathway completeness scoring
12. **Secretion Signals**: Multi-label secretion system prediction

**Key Design Principle**: Each phase builds on proven success from the previous phase, ensuring stable training and biological relevance.

## 🦠 Supported Species

- **E. coli K-12** (model organism)
- **B. subtilis 168** (Gram-positive)
- **Salmonella enterica** (pathogen)
- **Pseudomonas aeruginosa** (environmental)
- **Mycobacterium tuberculosis** (high GC)
- **Streptococcus pyogenes** (pathogen)
- **Synechocystis sp.** (cyanobacteria)

## 📊 Performance Targets

### Current Phase 1 Targets (Data-Driven)
- **Gene expression prediction**: R² ≥ 0.7
- **Gene density prediction**: R² ≥ 0.8  
- **Operon membership**: AUROC ≥ 0.9
- **Training stability**: Consistent loss reduction over 20 epochs

### Future Phase 2-3 Targets (Enhanced)
- **Multi-condition expression**: R² ≥ 0.75
- **Promoter element detection**: F1 ≥ 0.7
- **RBS efficiency prediction**: R² ≥ 0.80
- **Cross-species transfer**: <20% performance drop

## 🛠️ Project Structure

```
BactaGenome/
├── bactagenome/           # Main package
│   ├── model/            # Model architecture
│   ├── data/             # Data processing
│   ├── training/         # Training pipeline
│   └── evaluation/       # Evaluation utilities
├── configs/              # Configuration files
├── scripts/              # Utility scripts
├── data/                 # Data directory
├── experiments/          # Experiment tracking
├── models/               # Saved models
└── docs/                 # Documentation
```

## 📚 Documentation

- [**Output Modalities Evolution**](docs/OUTPUT_MODALITIES.md) - Detailed explanation of design changes
- [Model Architecture](docs/model_architecture.md)
- [Training Guide](docs/training_guide.md)
- [Data Formats](docs/data_format.md)

## 🏆 iGEM Competition

BactaGenome is designed specifically for iGEM (International Genetically Engineered Machine) competition applications:

- **Pathway Optimization**: Predict and optimize metabolic pathways
- **Promoter Design**: Engineer promoters with desired expression levels
- **RBS Tuning**: Design ribosome binding sites for precise translation control
- **System Integration**: Understand multi-gene system interactions

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and code of conduct.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on [AlphaGenome](https://github.com/lucidrains/alphagenome-pytorch) by Lucid Rains
- Inspired by DeepMind's AlphaFold architecture
- Bacterial datasets from RegulonDB, BioCyc, and other public resources

## 📞 Contact

- **Team**: BactaGenome Team
- **Email**: bactagenome@example.com
- **iGEM Team**: [Your iGEM Team Name]

---

*Built with ❤️ for the synthetic biology community*