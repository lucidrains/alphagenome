"""
RegulonDB data processing pipeline for BactaGenome
Converts RegulonDB BSON files to training-ready tensors
"""

import os
import json
import bson
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass
from collections import defaultdict
import logging
from tqdm import tqdm


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenomeSequenceData:
    """Container for genome sequence and annotations"""
    sequence: str
    length: int
    gc_content: float
    genes: Dict[str, Dict]
    operons: Dict[str, Dict]
    expression_data: Dict[str, np.ndarray]


class RegulonDBProcessor:
    """
    Processes RegulonDB BSON files for BactaGenome training
    
    Key data sources:
    - geneExpression.bson: Expression levels across conditions  
    - geneDatamart.bson: Gene sequences with 5' UTR (RBS regions)
    - operonDatamart.bson: Operon structure and co-regulation
    - transcriptionStartSite.bson: Promoter positions
    """
    
    def __init__(self, regulondb_path: str, output_dir: str):
        self.regulondb_path = Path(regulondb_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # E. coli K-12 MG1655 genome length
        self.genome_length = 4641652
        
        # Data containers
        self.genes = {}
        self.operons = {}
        self.expression_data = defaultdict(list)
        self.promoter_positions = []
        
        # DNA encoding
        self.dna_to_int = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}

    def load_bson_file(self, file_path: Path, batch_size: int = 1000) -> Iterator[Dict]:
        """Load and decode BSON file as iterator to handle large files efficiently"""
        try:
            with open(file_path, 'rb') as f:
                count = 0
                while True:
                    try:
                        # Use BSON's built-in file iterator
                        document = bson.decode_file_iter(f).__next__()
                        yield document
                        count += 1
                        
                        if count % batch_size == 0:
                            logger.debug(f"Processed {count} documents from {file_path.name}")
                            
                    except StopIteration:
                        # End of file reached
                        break
                    except Exception as doc_error:
                        logger.warning(f"Error decoding document {count}: {doc_error}")
                        break
                        
            logger.info(f"Loaded {count} documents from {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return iter([])
    
    def process_gene_datamart(self) -> Dict[str, Dict]:
        """
        Process geneDatamart.bson to extract gene sequences and 5' UTR regions
        
        Returns:
            Dictionary mapping gene IDs to gene information including sequences
        """
        logger.info("Processing gene datamart...")
        
        gene_file = self.regulondb_path / "regulondbdatamarts" / "geneDatamart.bson"
        documents = self.load_bson_file(gene_file)
        
        genes = {}
        for doc in tqdm(documents, desc="Processing genes"):
            gene_info = doc.get('gene', {})
            gene_id = gene_info.get('_id')
            
            if not gene_id:
                continue
                
            # Extract key gene information
            genes[gene_id] = {
                'bnumber': gene_info.get('bnumber'),
                'name': gene_info.get('name'),
                'sequence': gene_info.get('sequence', ''),
                'left_pos': gene_info.get('leftEndPosition'),
                'right_pos': gene_info.get('rightEndPosition'),
                'strand': gene_info.get('strand'),
                'gc_content': gene_info.get('gcContent', 0.0),
                'synonyms': gene_info.get('synonyms', [])
            }
            
            # Extract 5' UTR region (for RBS analysis)
            # UTR is typically 100-200 bp upstream of gene start
            if genes[gene_id]['left_pos'] and genes[gene_id]['strand']:
                if genes[gene_id]['strand'] == 'forward':
                    utr_start = max(0, genes[gene_id]['left_pos'] - 200)
                    utr_end = genes[gene_id]['left_pos']
                else:
                    utr_start = genes[gene_id]['right_pos']
                    utr_end = min(self.genome_length, genes[gene_id]['right_pos'] + 200)
                
                genes[gene_id]['utr_region'] = (utr_start, utr_end)
        
        logger.info(f"Processed {len(genes)} genes")
        self.genes = genes
        # print(f'genes: {genes}')
        return genes
    
    def process_expression_data(self) -> Dict[str, np.ndarray]:
        """
        Process geneExpression.bson to extract expression levels across conditions
        
        Returns:
            Dictionary mapping gene IDs to expression arrays across conditions
        """
        logger.info("Processing gene expression data...")
        
        expr_file = self.regulondb_path / "regulondbht" / "geneExpression.bson"
        documents = self.load_bson_file(expr_file)
        
        # Group by gene and collect expression across datasets/conditions
        expression_by_gene = defaultdict(list)
        dataset_ids = set()

        limit = None
        now_at = 0
        for doc in tqdm(documents, desc="Processing expression data"):  # Limit for testing
            gene_info = doc.get('gene', {})
            gene_id = gene_info.get('_id')
            bnumber = gene_info.get('bnumber')
            dataset_ids.update(doc.get('datasetIds', []))
            
            if gene_id and bnumber:
                expression_by_gene[gene_id].append({
                    'bnumber': bnumber,
                    'datasets': doc.get('datasetIds', []),
                    'temporal_id': doc.get('temporalId')
                })
            now_at += 1
            if limit is not None and now_at >= limit:
                break
        
        logger.info(f"Found {len(dataset_ids)} unique datasets")
        logger.info(f"Expression data for {len(expression_by_gene)} genes")
        
        # Convert to structured arrays (simplified for now)
        expression_data = {}
        for gene_id, expr_list in expression_by_gene.items():
            # Create a simple binary presence matrix (gene expressed in condition)
            num_conditions = min(50, len(dataset_ids))  # Limit to 50 conditions
            condition_vector = np.zeros(num_conditions)
            
            # Mark conditions where gene has expression data
            for i, expr in enumerate(expr_list[:num_conditions]):
                condition_vector[i] = 1.0  # Binary presence for now
            
            expression_data[gene_id] = condition_vector
        
        self.expression_data = expression_data
        return expression_data
    
    def process_operon_data(self) -> Dict[str, Dict]:
        """
        Process operonDatamart.bson to extract operon structure and co-regulation
        
        Returns:
            Dictionary mapping operon IDs to operon information
        """
        logger.info("Processing operon data...")
        
        operon_file = self.regulondb_path / "regulondbdatamarts" / "operonDatamart.bson"
        documents = self.load_bson_file(operon_file)
        
        operons = {}
        for doc in tqdm(documents, desc="Processing operons"):
            operon_info = doc.get('operon', {})
            operon_id = operon_info.get('_id')
            
            if not operon_id:
                continue
            
            # Extract transcription units and genes
            transcription_units = doc.get('transcriptionUnits', [])
            genes_in_operon = []
            
            for tu in transcription_units:
                tu_genes = tu.get('genes', [])
                for gene in tu_genes:
                    genes_in_operon.append({
                        'gene_id': gene.get('_id'),
                        'gene_name': gene.get('name')
                    })
            
            operons[operon_id] = {
                'name': operon_info.get('name'),
                'left_pos': operon_info.get('regulationPositions', {}).get('leftEndPosition'),
                'right_pos': operon_info.get('regulationPositions', {}).get('rightEndPosition'),
                'strand': operon_info.get('strand'),
                'genes': genes_in_operon,
                'num_genes': len(genes_in_operon),
                'transcription_units': transcription_units
            }
        
        logger.info(f"Processed {len(operons)} operons")
        self.operons = operons
        return operons
    
    def create_training_windows(
        self, 
        window_size: int = 98304,
        overlap: float = 0.6
    ) -> List[Dict]:
        """
        Create training windows from the E. coli genome
        
        Args:
            window_size: Size of each training window (98K bp for BactaGenome)
            overlap: Fraction of overlap between consecutive windows
            
        Returns:
            List of training windows with genomic coordinates and target data
        """
        logger.info(f"Creating training windows of size {window_size} bp...")
        
        step_size = int(window_size * (1 - overlap))
        windows = []
        
        for start_pos in range(0, self.genome_length - window_size + 1, step_size):
            end_pos = start_pos + window_size
            
            # Find genes and operons in this window
            genes_in_window = []
            operons_in_window = []
            
            for gene_id, gene_info in self.genes.items():
                # print(f'gene_id: {gene_id}, gene_info: {gene_info}')
                gene_start = gene_info.get('left_pos', 0) if gene_info.get('left_pos') is not None else 0
                gene_end = gene_info.get('right_pos', 0) if gene_info.get('right_pos') is not None else 0
                
                # Check if gene overlaps with window
                if gene_start < end_pos and gene_end > start_pos:
                    genes_in_window.append({
                        'gene_id': gene_id,
                        'relative_start': max(0, gene_start - start_pos),
                        'relative_end': min(window_size, gene_end - start_pos),
                        'expression': self.expression_data.get(gene_id, np.zeros(50))
                    })
            
            for operon_id, operon_info in self.operons.items():
                operon_start = operon_info.get('left_pos', 0)
                operon_end = operon_info.get('right_pos', 0)
                
                # Check if operon overlaps with window
                if operon_start and operon_end and operon_start < end_pos and operon_end > start_pos:
                    operons_in_window.append({
                        'operon_id': operon_id,
                        'relative_start': max(0, operon_start - start_pos),
                        'relative_end': min(window_size, operon_end - start_pos),
                        'genes': operon_info['genes']
                    })
            
            # Create window data structure
            window_data = {
                'window_id': f"ecoli_window_{start_pos}_{end_pos}",
                'genomic_start': start_pos,
                'genomic_end': end_pos,
                'genes': genes_in_window,
                'operons': operons_in_window,
                'sequence_placeholder': f"PLACEHOLDER_{window_size}_BP",  # Will be filled with actual sequence
            }
            
            windows.append(window_data)
        
        logger.info(f"Created {len(windows)} training windows")
        return windows
    
    def create_target_tensors(self, windows: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Create target tensors for each modality from processed windows
        
        Args:
            windows: List of training windows with genomic annotations
            
        Returns:
            Dictionary of target tensors by modality
        """
        logger.info("Creating target tensors...")
        
        num_windows = len(windows)
        window_size = 98304
        
        # Initialize target tensors
        targets = {
            'promoter_strength': torch.zeros(num_windows, window_size, 50),  # 50 conditions
            'rbs_efficiency': torch.zeros(num_windows, window_size, 1),
            'operon_coregulation': torch.zeros(num_windows, window_size // 128, 20)  # 20 co-expression tracks
        }
        
        for window_idx, window in enumerate(tqdm(windows, desc="Creating targets")):
            
            # Promoter strength targets (expression levels)
            for gene in window['genes']:
                gene_start = gene['relative_start']
                gene_end = gene['relative_end']
                expression = gene['expression']
                
                # Assign expression to gene region
                if gene_start < window_size and gene_end > 0:
                    start_idx = max(0, gene_start)
                    end_idx = min(window_size, gene_end)
                    
                    # Broadcast expression across gene region
                    for pos in range(start_idx, end_idx):
                        targets['promoter_strength'][window_idx, pos][:expression.shape[0]] = torch.from_numpy(expression)
            
            # RBS efficiency targets (simplified - binary presence)
            for gene in window['genes']:
                gene_start = gene['relative_start']
                # RBS is typically ~100 bp upstream of gene start
                rbs_start = max(0, gene_start - 100)
                rbs_end = min(window_size, gene_start + 20)
                
                if rbs_start < rbs_end:
                    # Simple binary RBS presence
                    targets['rbs_efficiency'][window_idx, rbs_start:rbs_end, 0] = 1.0
            
            # Operon co-regulation targets (gene co-expression)
            for operon in window['operons']:
                operon_start = operon['relative_start'] // 128  # Convert to 128bp resolution
                operon_end = operon['relative_end'] // 128
                
                if operon_start < operon_end and operon_end <= window_size // 128:
                    # Create co-expression pattern for genes in operon
                    coexpr_pattern = np.random.rand(20)  # Placeholder - would use real co-expression data
                    
                    for pos in range(operon_start, operon_end):
                        targets['operon_coregulation'][window_idx, pos] = torch.from_numpy(coexpr_pattern)
        
        logger.info(f"Created target tensors for {len(targets)} modalities")
        return targets
    
    def save_processed_data(self, windows: List[Dict], targets: Dict[str, torch.Tensor]):
        """Save processed data for training"""
        logger.info("Saving processed data...")
        
        # Save windows metadata
        windows_file = self.output_dir / "training_windows.json"
        with open(windows_file, 'w') as f:
            json.dump(windows, f, indent=2, default=str)
        
        # Save target tensors
        targets_file = self.output_dir / "target_tensors.pt"
        torch.save(targets, targets_file)
        
        # Save gene and operon metadata
        metadata = {
            'genes': self.genes,
            'operons': self.operons,
            'expression_stats': {
                'num_genes_with_expression': len(self.expression_data),
                'num_conditions': 50
            }
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved processed data to {self.output_dir}")
    
    def process_all(self) -> Tuple[List[Dict], Dict[str, torch.Tensor]]:
        """
        Complete processing pipeline
        
        Returns:
            Tuple of (training_windows, target_tensors)
        """
        logger.info("Starting complete RegulonDB processing pipeline...")
        
        # Process each data type
        self.process_gene_datamart()
        self.process_expression_data()
        self.process_operon_data()
        
        # Create training data
        windows = self.create_training_windows()
        targets = self.create_target_tensors(windows)
        
        # Save results
        self.save_processed_data(windows, targets)
        
        logger.info("RegulonDB processing complete!")
        return windows, targets


def main():
    """Example usage of RegulonDB processor"""
    regulondb_path = "./data/raw/RegulonDB"
    output_dir = "./data/processed/regulondb"
    
    processor = RegulonDBProcessor(regulondb_path, output_dir)
    windows, targets = processor.process_all()
    
    print(f"Processing complete:")
    print(f"- {len(windows)} training windows created")
    print(f"- Target tensors for modalities: {list(targets.keys())}")
    print(f"- Target tensor shapes:")
    for modality, tensor in targets.items():
        print(f"  {modality}: {tensor.shape}")


if __name__ == "__main__":
    main()