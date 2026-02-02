#!/usr/bin/env python3
"""Detailed benchmark showing practical benefits of selective inference.

This script demonstrates:
1. Output tensor memory savings (main benefit for post-processing/storage)
2. Time savings from skipping head computations
3. Comparison between different filtering scenarios
"""

import argparse
import gc
import time

import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.alphagenome import set_update_running_var


def format_bytes(nbytes):
    """Format bytes as human-readable string."""
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.2f} GB"
    elif nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.1f} MB"
    else:
        return f"{nbytes / 1024:.1f} KB"


def count_output_tensors(outputs):
    """Count tensors and total bytes in output."""
    total_bytes = 0
    num_tensors = 0

    def walk(obj):
        nonlocal total_bytes, num_tensors
        if isinstance(obj, torch.Tensor):
            total_bytes += obj.numel() * obj.element_size()
            num_tensors += 1
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)

    walk(outputs)
    return num_tensors, total_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', type=int, default=131072)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 70)
    print("SELECTIVE HEAD EXECUTION BENCHMARK")
    print("=" * 70)
    print(f"Sequence length: {args.seq_len:,} bp")
    print(f"Device: {device}")

    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    print()

    # Create model
    print("Loading model...")
    model = AlphaGenome()
    model.add_reference_heads('human')
    model.to(device)
    model.eval()
    set_update_running_var(model, False)

    # Input
    seq = torch.randint(0, 4, (1, args.seq_len), device=device)

    # Track configurations
    rna_tracks = 768
    cage_tracks = 640
    dnase_tracks = 384
    atac_tracks = 256
    procap_tracks = 128
    chip_tf_tracks = 1664
    chip_histone_tracks = 1152
    contact_tracks = 28

    print("=" * 70)
    print("OUTPUT COMPARISON")
    print("=" * 70)
    print()

    scenarios = [
        ("Full (all heads)", None, None),
        ("1bp tracks only", {'rna_seq', 'cage', 'dnase', 'atac', 'procap'}, None),
        ("128bp tracks only", {'chip_tf', 'chip_histone'}, None),
        ("RNA-seq only", {'rna_seq'}, None),
        ("RNA-seq + 10% filter", {'rna_seq'}, {'rna_seq': torch.arange(rna_tracks, device=device) < rna_tracks // 10}),
        ("RNA-seq + 1% filter", {'rna_seq'}, {'rna_seq': torch.arange(rna_tracks, device=device) < max(1, rna_tracks // 100)}),
        ("CAGE only", {'cage'}, None),
        ("Contact maps only", {'contact_maps'}, None),
    ]

    # Warmup
    with torch.no_grad():
        _ = model.inference(seq, 0, requested_heads={'rna_seq'})
    if device.type == 'cuda':
        torch.cuda.synchronize()

    results = []
    for name, heads, masks in scenarios:
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.inference(seq, 0, requested_heads=heads, track_masks=masks)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        num_tensors, output_bytes = count_output_tensors(outputs)
        peak_mem = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0

        results.append({
            'name': name,
            'time_ms': elapsed * 1000,
            'output_bytes': output_bytes,
            'num_tensors': num_tensors,
            'peak_mem': peak_mem,
        })

        del outputs

    # Print results
    print(f"{'Scenario':<30} {'Time (ms)':<12} {'Output Size':<12} {'Peak Mem':<12}")
    print("-" * 66)

    baseline = results[0]
    for r in results:
        time_str = f"{r['time_ms']:.1f}"
        output_str = format_bytes(r['output_bytes'])
        peak_str = format_bytes(r['peak_mem']) if r['peak_mem'] > 0 else "N/A"
        print(f"{r['name']:<30} {time_str:<12} {output_str:<12} {peak_str:<12}")

    print()
    print("=" * 70)
    print("SAVINGS vs FULL INFERENCE")
    print("=" * 70)
    print()
    print(f"{'Scenario':<30} {'Time Saved':<15} {'Output Saved':<15} {'Output %':<10}")
    print("-" * 70)

    for r in results[1:]:  # Skip baseline
        time_saved = baseline['time_ms'] - r['time_ms']
        output_saved = baseline['output_bytes'] - r['output_bytes']
        output_pct = (output_saved / baseline['output_bytes']) * 100

        time_str = f"{time_saved:.1f} ms"
        output_str = format_bytes(output_saved)
        pct_str = f"{output_pct:.0f}%"
        print(f"{r['name']:<30} {time_str:<15} {output_str:<15} {pct_str:<10}")

    # Additional insights
    print()
    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()

    rna_only = next(r for r in results if r['name'] == 'RNA-seq only')
    rna_10pct = next(r for r in results if r['name'] == 'RNA-seq + 10% filter')
    rna_1pct = next(r for r in results if r['name'] == 'RNA-seq + 1% filter')

    print(f"1. Full -> RNA-seq only: Output {format_bytes(baseline['output_bytes'])} -> {format_bytes(rna_only['output_bytes'])}")
    print(f"   Saves {format_bytes(baseline['output_bytes'] - rna_only['output_bytes'])} of output tensor memory")
    print()
    print(f"2. RNA-seq -> 10% ontology filter: {format_bytes(rna_only['output_bytes'])} -> {format_bytes(rna_10pct['output_bytes'])}")
    print(f"   Saves {format_bytes(rna_only['output_bytes'] - rna_10pct['output_bytes'])} of output tensor memory")
    print()
    print(f"3. For 1MB sequences (extrapolated from {args.seq_len:,}bp):")
    scale = 1048576 / args.seq_len
    full_1mb = baseline['output_bytes'] * scale
    rna_1mb = rna_only['output_bytes'] * scale
    filtered_1mb = rna_10pct['output_bytes'] * scale
    print(f"   - Full inference output: ~{format_bytes(full_1mb)}")
    print(f"   - RNA-seq only output: ~{format_bytes(rna_1mb)}")
    print(f"   - RNA-seq + 10% filter: ~{format_bytes(filtered_1mb)}")
    print(f"   - Total savings: ~{format_bytes(full_1mb - filtered_1mb)}")


if __name__ == '__main__':
    main()
