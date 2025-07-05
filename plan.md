# Bacterial AlphaGenome: Complete Implementation Plan

## Project Overview

**Goal**: Create an "AlphaGenome for Bacteria" by adapting AlphaGenome's architecture with bacterial-specific output modalities for iGEM competition.

**Key Insight**: Keep AlphaGenome's core encoder-decoder architecture, replace only the output heads with bacterial-specific modalities.

**Context Window**: 100K bp (10x smaller than AlphaGenome's 1M bp for computational efficiency)

## Core Architecture (Reuse from AlphaGenome)

### **Input/Output Scaling (Corrected - Full Length Output)**
```python
# Full-length output (no cropping)
CONTEXT_LENGTH = 98304   # ~100K bp (2^16 + 2^15)
OUTPUT_1BP_BINS = 98304  # Same as input - full length!
OUTPUT_128BP_BINS = 768  # 98304/128 = 768 bins
PAIRWISE_SIZE = 48       # 98304/2048 ≈ 48
```

### **Core Components (Keep Unchanged)**
- **Encoder**: Convolutional layers for local pattern detection
- **Transformer**: Long-range sequence communication  
- **U-Net Decoder**: Multi-scale processing
- **Multi-resolution embeddings**: 1bp, 128bp, 2048bp (pairwise)

## Output Heads Design

### **Priority 1: Core Expression Control (Weeks 1-8)**

#### **1. Promoter Strength Prediction**
**Resolution**: 1bp + 128bp embeddings  
**Output**: Expression levels across conditions

```python
def promoter_strength_head(embeddings_1bp, embeddings_128bp):
    x_1bp = Linear(num_conditions)(embeddings_1bp)
    x_128bp = Linear(num_conditions)(embeddings_128bp) 
    combined = (x_1bp + Repeat(x_128bp, 128)) / 2
    return Softplus(combined) * learnable_scale
```

**Datasets**:
- **RegulonDB**: 5,843 E. coli promoters with σ⁷⁰ recognition
- **MPRA Dataset**: 117,556 E. coli promoter variants with quantitative activity  
- **PePPER Database**: Cross-species prokaryotic promoter elements
- **Synthetic libraries**: ~10,000 designed sequences with measured strength

**Loss**: Multinomial + Poisson (like AlphaGenome RNA-seq)

#### **2. RBS Translation Efficiency**  
**Resolution**: 1bp embeddings  
**Output**: Translation initiation rates

```python
def rbs_efficiency_head(embeddings_1bp):
    x = Linear(1)(embeddings_1bp)
    return Softplus(x) * scale_factor
```

**Datasets**:
- **RBS Calculator**: 12,000+ characterized E. coli RBS sequences
- **UTR Designer**: 10,000+ 5' UTR variants with FACS-measured efficiency
- **BioBrick Registry**: 2,000+ characterized RBS parts
- **Species-specific**: B. subtilis, S. cerevisiae RBS collections

**Loss**: MSE on log-transformed efficiency ratios

#### **3. Operon Co-regulation**
**Resolution**: 128bp + 2048bp embeddings  
**Output**: Gene co-expression within operons

```python
def operon_coregulation_head(embeddings_128bp, embeddings_pair):
    local_context = Linear(num_genes)(embeddings_128bp)
    pair_features = MeanPool2D(embeddings_pair, kernel_size=4)
    return Softplus(local_context + pair_contribution)
```

**Datasets**:
- **RegulonDB**: 2,724 E. coli operons with regulation data
- **Operon Database**: 1,300+ validated E. coli operons  
- **Multi-condition RNA-seq**: 104 B. subtilis conditions
- **Comparative genomics**: Operon conservation across 100+ species

**Loss**: Correlation loss between genes in same operon

### **Priority 2: Advanced Regulation (Weeks 9-12)**

#### **4. Riboswitch Ligand Binding**
**Resolution**: 1bp + 128bp embeddings  
**Output**: Binding probability for different metabolites

```python
def riboswitch_binding_head(embeddings_1bp, embeddings_128bp):
    motif_features = CNN_1D(embeddings_1bp, kernel_size=15)
    context_features = Linear(num_ligands)(embeddings_128bp)
    binding_logits = motif_features + context_features
    return Sigmoid(binding_logits)
```

**Datasets**:
- **Riboswitch Database**: 55+ characterized classes with ligand specificity
- **Rfam Families**: 200,000+ putative riboswitch sequences  
- **Experimental binding**: Kd values for ~500 riboswitch-ligand pairs
- **Synthetic aptamers**: In vitro selected sequences

**Loss**: Binary cross-entropy for ligand binding

#### **5. Small RNA Target Prediction**
**Resolution**: 1bp + 128bp embeddings  
**Output**: Interaction strength with known sRNAs

```python
def srna_target_head(embeddings_1bp, embeddings_128bp):
    base_pair_features = Linear(num_srnas)(embeddings_1bp)
    context_features = Linear(num_srnas)(embeddings_128bp)
    interaction_strength = base_pair_features + Repeat(context_features, 128)
    return Softplus(interaction_strength)
```

**Datasets**:
- **sRNAdb**: 1,500+ bacterial sRNAs with targets
- **CopraRNA**: 50,000+ sRNA-target interactions  
- **PAR-CLIP/CLASH**: Hfq-mediated interaction data
- **IntaRNA**: Computationally predicted targets with energy scores

**Loss**: Ranking loss for target prioritization

### **Priority 3: Systems-Level Features (Weeks 13-16)**

#### **6. Transcription Termination**
**Resolution**: 1bp embeddings  
**Output**: Termination probability and type

```python
def termination_head(embeddings_1bp):
    terminator_logits = Linear(2)(embeddings_1bp)  # Intrinsic vs Rho
    return Softmax(terminator_logits)
```

**Datasets**:
- **TransTermHP**: 50,000+ intrinsic terminators
- **Rho termination**: 1,200+ Rho-dependent sites
- **Term-seq**: Genome-wide termination mapping
- **Comparative analysis**: Cross-species terminator conservation

#### **7. Metabolic Pathway Activity**
**Resolution**: 128bp + 2048bp embeddings  
**Output**: Pathway completeness scores

```python
def pathway_activity_head(embeddings_128bp, embeddings_pair):
    gene_activities = MeanPool1D(embeddings_128bp, kernel_size=50)
    organization_features = GlobalMeanPool(embeddings_pair)
    pathway_scores = Linear(num_pathways)(
        Concatenate([gene_activities, organization_features])
    )
    return Sigmoid(pathway_scores)
```

**Datasets**:
- **KEGG Pathways**: 478+ bacterial reference pathways
- **BioCyc Database**: 20,000+ organism-specific variants
- **MetaCyc**: 3,000+ experimentally validated pathways
- **Multi-omics**: RNA-seq + metabolomics validation

#### **8. Protein Secretion Signals**
**Resolution**: 1bp embeddings  
**Output**: Secretion system type predictions

```python
def secretion_signal_head(embeddings_1bp):
    signal_logits = Linear(num_secretion_types)(embeddings_1bp)
    return Sigmoid(signal_logits)  # Multi-label: T1SS, T2SS, etc.
```

**Datasets**:
- **SignalP**: 100,000+ signal peptides across species
- **MacSyFinder**: Secretion system component models
- **SecretomeP**: Non-classical secretion predictions
- **Experimental proteomics**: Secreted protein identification

## Training Strategy

### **Phase 1: Proof of Concept (Weeks 1-8)**
**Scope**: 3 core modalities (promoter, RBS, operon) on E. coli only

**Configuration (Updated for Full-Length)**:
```python
training_config_phase1 = {
    "species": ["E_coli_K12"],
    "modalities": ["promoter_strength", "rbs_efficiency", "operon_coregulation"], 
    "context_length": 98304,
    "output_length": 98304,  # Full length output
    "batch_size": 16,        # May need to reduce due to higher memory
    "learning_rate": 0.002,
    "total_steps": 8000,
    "hardware": "A800x4"
}
```

**Budget**: $3,000-5,000  
**Success Criteria**: R² > 0.7 for promoter/RBS prediction

### **Phase 2: Multi-Species Expansion (Weeks 9-12)**  
**Scope**: All 8 modalities, 7 bacterial species

**Species Selection**:
- **E. coli K-12** (model organism)
- **B. subtilis 168** (Gram-positive representative)
- **Salmonella enterica** (E. coli relative)
- **Pseudomonas aeruginosa** (environmental bacteria)
- **Mycobacterium tuberculosis** (high GC content)
- **Streptococcus pyogenes** (pathogen)
- **Synechocystis sp.** (cyanobacteria)

**Cross-Validation Strategy**:
```python
# Chromosome-based splitting for bacterial genomes
def bacterial_cv_folds():
    return {
        "fold_0": "ori_to_90_degrees",    # Origin to 90°
        "fold_1": "90_to_180_degrees",    # 90° to 180°  
        "fold_2": "180_to_270_degrees",   # 180° to 270°
        "fold_3": "270_to_ori_degrees"    # 270° to origin
    }
```

**Configuration (Updated)**:
```python
training_config_phase2 = {
    "species": all_7_species,
    "modalities": all_8_modalities,
    "context_length": 98304,
    "output_length": 98304,  # Full length
    "batch_size": 24,        # Slightly reduced due to full-length memory usage
    "learning_rate": 0.003,
    "total_steps": 12000,
    "hardware": "A800x8",
    "sequence_parallelism": True
}
```

**Budget**: $8,000-12,000

### **Phase 3: Optional Distillation (Weeks 13-14)**
**Decision Point**: Only if Phase 2 models show inconsistencies

**Simple 4-Teacher Distillation**:
```python
distillation_config = {
    "teachers": [fold_0_model, fold_1_model, fold_2_model, fold_3_model],
    "augmentation_rate": 0.04,  # 4% mutations
    "total_steps": 10000,
    "learning_rate": 0.001
}
```

**Budget**: $2,000-3,000 (optional)

### **Phase 4: Validation & Deployment (Weeks 15-20)**

**Validation Strategy**:
- **Hold-out species**: Test on 3 unseen bacterial species
- **Cross-modality validation**: Verify biological consistency
- **iGEM use cases**: Pathway optimization, promoter design

**Performance Targets**:
- **Promoter prediction**: R² ≥ 0.75
- **RBS efficiency**: R² ≥ 0.80  
- **Operon prediction**: AUROC ≥ 0.90
- **Cross-species transfer**: <20% performance drop

## Implementation Milestones

### **Week 4 Checkpoint**: Infrastructure Ready
- [ ] 100K context architecture implemented
- [ ] Bacterial genome preprocessing pipeline
- [ ] Priority datasets collected and processed
- [ ] Cross-validation folds defined

### **Week 8 Checkpoint**: Proof of Concept  
- [ ] 3 core modalities training successfully
- [ ] E. coli model achieving target performance
- [ ] Training pipeline validated and stable

### **Week 12 Checkpoint**: Full Model
- [ ] All 8 modalities integrated
- [ ] Multi-species training completed
- [ ] 4-fold cross-validation results available

### **Week 16 Checkpoint**: Model Selection
- [ ] Best model selected (single fold or ensemble)
- [ ] Validation on hold-out species completed
- [ ] Performance benchmarking finished

### **Week 20 Checkpoint**: Competition Ready
- [ ] iGEM-specific applications developed
- [ ] API and documentation completed
- [ ] Model submitted for competition

## Resource Summary

**Total Computational Budget**: $15,000-20,000
**Total Timeline**: 20 weeks
**Key Deliverable**: Bacterial genome model optimized for synthetic biology applications

**Success Metrics**:
- Technical: Outperform existing bacterial prediction tools
- Practical: Enable novel iGEM project applications  
- Competition: Win best model award through innovation and performance

This plan balances ambitious technical goals with practical constraints, providing a clear path to creating a groundbreaking bacterial genome modeling tool for the synthetic biology community.