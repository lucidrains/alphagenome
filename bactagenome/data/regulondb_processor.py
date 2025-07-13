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
        self.expression_data = {}
        self.expression_stats = {}
        self.promoter_positions = []
        
        # DNA encoding
        self.dna_to_int = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}

    def load_bson_file(self, file_path: Path, batch_size: int = 1000, max_docs: Optional[int] = None) -> Iterator[Dict]:
        """Load and decode BSON file as iterator to handle large files efficiently"""
        try:
            with open(file_path, 'rb') as f:
                count = 0
                while True:
                    if max_docs and count >= max_docs:
                        break
                        
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
    
    def process_gene_datamart(self, max_docs: Optional[int] = None) -> Dict[str, Dict]:
        """
        Process geneDatamart.bson to extract gene sequences and 5' UTR regions
        
        Returns:
            Dictionary mapping gene IDs to gene information including sequences
        """
        logger.info("Processing gene datamart...")
        
        gene_file = self.regulondb_path / "regulondbdatamarts" / "geneDatamart.bson"
        documents = self.load_bson_file(gene_file, max_docs=max_docs)
        
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
    
    def process_expression_data(self, max_docs: Optional[int] = None) -> Dict[str, Dict]:
        """
        Process real expression data from RegulonDB with proper normalization
        
        Returns:
            Dictionary mapping gene IDs to expression dictionaries with normalized values
        """
        logger.info("Processing gene expression data...")
        
        expr_file = self.regulondb_path / "regulondbht" / "geneExpression.bson"
        documents = self.load_bson_file(expr_file)
        
        # Collect raw expression values
        raw_expression = defaultdict(list)
        all_tpm_values = []
        all_fpkm_values = []
        all_count_values = []
        
        count = 0
        for doc in tqdm(documents, desc="Processing expression"):
            if max_docs and count >= max_docs:
                break
                
            gene_info = doc.get('gene', {})
            gene_id = gene_info.get('_id')
            bnumber = gene_info.get('bnumber')
            
            # Extract expression values if available
            tpm = doc.get('tpm')
            fpkm = doc.get('fpkm') 
            count_val = doc.get('count')
            
            if gene_id and any([tpm, fpkm, count_val]):
                expression_record = {
                    'gene_id': gene_id,
                    'bnumber': bnumber,
                    'tpm': tpm,
                    'fpkm': fpkm,
                    'count': count_val,
                    'dataset_ids': doc.get('datasetIds', [])
                }
                
                raw_expression[gene_id].append(expression_record)
                
                # Collect for normalization stats
                if tpm is not None:
                    all_tpm_values.append(tpm)
                if fpkm is not None:
                    all_fpkm_values.append(fpkm)
                if count_val is not None:
                    all_count_values.append(count_val)
            
            count += 1
        
        logger.info(f"Found expression data for {len(raw_expression)} genes")
        logger.info(f"Total TPM values: {len(all_tpm_values)}")
        logger.info(f"Total FPKM values: {len(all_fpkm_values)}")
        logger.info(f"Total count values: {len(all_count_values)}")
        
        # Compute normalization statistics (log-transform for very large values)
        expression_stats = {
            'tpm_log_mean': 0.0,
            'tpm_log_std': 1.0,
            'fpkm_log_mean': 0.0,
            'fpkm_log_std': 1.0,
            'count_log_mean': 0.0,
            'count_log_std': 1.0
        }
        
        if all_tpm_values:
            log_tpm = np.log1p(all_tpm_values)  # log(1+x) for numerical stability
            expression_stats['tpm_log_mean'] = np.mean(log_tpm)
            expression_stats['tpm_log_std'] = np.std(log_tpm)
            
        if all_fpkm_values:
            log_fpkm = np.log1p(all_fpkm_values)
            expression_stats['fpkm_log_mean'] = np.mean(log_fpkm)
            expression_stats['fpkm_log_std'] = np.std(log_fpkm)
            
        if all_count_values:
            log_count = np.log1p(all_count_values)
            expression_stats['count_log_mean'] = np.mean(log_count)
            expression_stats['count_log_std'] = np.std(log_count)
        
        self.expression_stats = expression_stats
        logger.info(f"Expression normalization stats: {expression_stats}")
        
        # Process and normalize expression data
        processed_expression = {}
        for gene_id, expr_list in raw_expression.items():
            # Average across multiple measurements for the same gene
            tpm_values = [e['tpm'] for e in expr_list if e['tpm'] is not None]
            fpkm_values = [e['fpkm'] for e in expr_list if e['fpkm'] is not None]
            count_values = [e['count'] for e in expr_list if e['count'] is not None]
            
            processed_record = {
                'bnumber': expr_list[0]['bnumber'],
                'num_measurements': len(expr_list),
                'datasets': list(set().union(*[e['dataset_ids'] for e in expr_list]))
            }
            
            # Compute mean and normalized values
            if tpm_values:
                mean_tpm = np.mean(tpm_values)
                log_tpm = np.log1p(mean_tpm)
                processed_record['tpm_raw'] = mean_tpm
                processed_record['tpm_log'] = log_tpm
                processed_record['tpm_normalized'] = (log_tpm - expression_stats['tpm_log_mean']) / expression_stats['tpm_log_std']
                
            if fpkm_values:
                mean_fpkm = np.mean(fpkm_values)
                log_fpkm = np.log1p(mean_fpkm)
                processed_record['fpkm_raw'] = mean_fpkm
                processed_record['fpkm_log'] = log_fpkm
                processed_record['fpkm_normalized'] = (log_fpkm - expression_stats['fpkm_log_mean']) / expression_stats['fpkm_log_std']
                
            if count_values:
                mean_count = np.mean(count_values)
                log_count = np.log1p(mean_count)
                processed_record['count_raw'] = mean_count
                processed_record['count_log'] = log_count
                processed_record['count_normalized'] = (log_count - expression_stats['count_log_mean']) / expression_stats['count_log_std']
            
            processed_expression[gene_id] = processed_record
        
        self.expression_data = processed_expression
        return processed_expression
    
    def process_operon_data(self, max_docs: Optional[int] = None) -> Dict[str, Dict]:
        """
        Process operonDatamart.bson to extract operon structure and co-regulation
        
        Returns:
            Dictionary mapping operon IDs to operon information
        """
        logger.info("Processing operon data...")
        
        operon_file = self.regulondb_path / "regulondbdatamarts" / "operonDatamart.bson"
        documents = self.load_bson_file(operon_file, max_docs=max_docs)
        
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
                        'expression': self.expression_data.get(gene_id, {})
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
        Create realistic training targets based on available data
        
        Target 1: Gene Expression Level (log-normalized TPM/FPKM)
        Target 2: Gene Density (genes per region)  
        Target 3: Operon Membership (binary classification)
        """
        logger.info("Creating realistic training targets...")
        
        num_windows = len(windows)
        window_size = 98304
        
        # Initialize target tensors with appropriate designs
        targets = {
            # Gene expression prediction (continuous values, properly normalized)
            'gene_expression': torch.zeros(num_windows, window_size, 1),
            
            # Gene density prediction (count of genes per 128bp bin)
            'gene_density': torch.zeros(num_windows, window_size // 128, 1),
            
            # Operon membership (binary classification for gene regions)
            'operon_membership': torch.zeros(num_windows, window_size, 1)
        }
        
        for window_idx, window in enumerate(tqdm(windows, desc="Creating targets")):
            
            # Target 1: Gene Expression Level
            for gene in window['genes']:
                gene_start = gene['relative_start']
                gene_end = gene['relative_end']
                
                if gene_start < window_size and gene_end > 0:
                    start_idx = max(0, gene_start)
                    end_idx = min(window_size, gene_end)
                    
                    # Use normalized TPM as expression target (default to 0 if no data)
                    expression_value = 0.0
                    if 'expression' in gene and isinstance(gene['expression'], dict):
                        expr_data = gene['expression']
                        if 'tpm_normalized' in expr_data:
                            expression_value = expr_data['tpm_normalized']
                        elif 'fpkm_normalized' in expr_data:
                            expression_value = expr_data['fpkm_normalized']
                    
                    # Assign expression value to gene region
                    targets['gene_expression'][window_idx, start_idx:end_idx, 0] = expression_value
            
            # Target 2: Gene Density (genes per 128bp bin)
            bin_size = 128
            num_bins = window_size // bin_size
            
            for gene in window['genes']:
                gene_start = gene['relative_start']
                gene_end = gene['relative_end']
                
                # Determine which bins this gene overlaps
                start_bin = max(0, gene_start // bin_size)
                end_bin = min(num_bins - 1, gene_end // bin_size)
                
                for bin_idx in range(start_bin, end_bin + 1):
                    targets['gene_density'][window_idx, bin_idx, 0] += 1.0
            
            # Target 3: Operon Membership
            for operon in window['operons']:
                for gene_info in operon['genes']:
                    gene_id = gene_info['gene_id']
                    
                    # Find the corresponding gene in window['genes']
                    for gene in window['genes']:
                        if gene['gene_id'] == gene_id:
                            gene_start = gene['relative_start']
                            gene_end = gene['relative_end']
                            
                            if gene_start < window_size and gene_end > 0:
                                start_idx = max(0, gene_start)
                                end_idx = min(window_size, gene_end)
                                
                                # Mark as part of operon
                                targets['operon_membership'][window_idx, start_idx:end_idx, 0] = 1.0
                            break
        
        # Log target statistics
        for target_name, target_tensor in targets.items():
            non_zero = (target_tensor != 0).float().mean().item()
            mean_val = target_tensor.mean().item()
            std_val = target_tensor.std().item()
            
            logger.info(f"{target_name}: {non_zero:.1%} non-zero, mean={mean_val:.3f}, std={std_val:.3f}")
        
        return targets
    
    def save_processed_data(self, windows: List[Dict], targets: Dict[str, torch.Tensor]):
        """Save processed data for training"""
        logger.info("Saving processed data...")
        
        # Save windows metadata
        windows_file = self.output_dir / "training_windows.json"
        with open(windows_file, 'w') as f:
            json.dump(windows, f, indent=2, default=str)
        
        # Save target tensors
        targets_file = self.output_dir / "training_targets.pt"
        torch.save(targets, targets_file)
        
        # Save normalization statistics separately
        stats_file = self.output_dir / "normalization_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.expression_stats, f, indent=2)
        
        # Save gene and operon metadata
        metadata = {
            'genes': self.genes,
            'operons': self.operons,
            'expression_stats': self.expression_stats,
            'data_summary': {
                'num_genes_with_expression': len(self.expression_data),
                'num_genes_total': len(self.genes),
                'num_operons_total': len(self.operons)
            }
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved processed data to {self.output_dir}")
    
    def process_all(self, max_docs_per_file: Optional[int] = None) -> Tuple[List[Dict], Dict[str, torch.Tensor]]:
        """
        Complete processing pipeline
        
        Returns:
            Tuple of (training_windows, target_tensors)
        """
        logger.info("Starting complete RegulonDB processing pipeline...")
        
        # Process each data type
        self.process_gene_datamart(max_docs_per_file)
        self.process_expression_data(max_docs_per_file)
        self.process_operon_data(max_docs_per_file)
        
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