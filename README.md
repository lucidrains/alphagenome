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

### Core Expression Control
1. **Promoter Strength**: Expression levels across conditions
2. **RBS Efficiency**: Translation initiation rates  
3. **Operon Co-regulation**: Gene co-expression within operons

### Advanced Regulation
4. **Riboswitch Binding**: Ligand binding probabilities
5. **sRNA Targets**: Small RNA interaction predictions

### Systems-Level Features  
6. **Transcription Termination**: Termination sites and mechanisms
7. **Pathway Activity**: Metabolic pathway completeness
8. **Secretion Signals**: Protein secretion system predictions

## 🦠 Supported Species

- **E. coli K-12** (model organism)
- **B. subtilis 168** (Gram-positive)
- **Salmonella enterica** (pathogen)
- **Pseudomonas aeruginosa** (environmental)
- **Mycobacterium tuberculosis** (high GC)
- **Streptococcus pyogenes** (pathogen)
- **Synechocystis sp.** (cyanobacteria)

## 📊 Performance Targets

- **Promoter prediction**: R² ≥ 0.75
- **RBS efficiency**: R² ≥ 0.80  
- **Operon prediction**: AUROC ≥ 0.90
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