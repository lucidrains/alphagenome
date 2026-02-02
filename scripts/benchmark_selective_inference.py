#!/usr/bin/env python3
"""Benchmark selective head execution and ontology filtering.

Compares:
1. Full inference (all heads)
2. Single head (rna_seq only)
3. Single head with track masking (simulated ontology filter)

Measures:
- Inference time
- Peak GPU memory
- Output tensor memory
"""

import argparse
import gc
import time
from contextlib import contextmanager

import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.alphagenome import set_update_running_var


@contextmanager
def track_memory(device):
    """Context manager to track peak GPU memory."""
    if device.type != 'cuda':
        yield {'peak_mb': 0, 'allocated_mb': 0}
        return

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start_mem = torch.cuda.memory_allocated(device)

    stats = {}
    try:
        yield stats
    finally:
        torch.cuda.synchronize(device)
        stats['peak_mb'] = torch.cuda.max_memory_allocated(device) / 1024**2
        stats['allocated_mb'] = (torch.cuda.memory_allocated(device) - start_mem) / 1024**2


def measure_output_memory(outputs, device):
    """Measure memory of output tensors in MB."""
    total_bytes = 0

    def count_tensor_bytes(obj):
        nonlocal total_bytes
        if isinstance(obj, torch.Tensor):
            total_bytes += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                count_tensor_bytes(v)

    count_tensor_bytes(outputs)
    return total_bytes / 1024**2


def benchmark_scenario(model, seq, organism_index, name, requested_heads=None, track_masks=None, num_warmup=2, num_runs=5, use_autocast=False):
    """Run benchmark for a specific scenario."""
    device = seq.device

    def run_inference():
        if use_autocast and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                return model.inference(
                    seq, organism_index,
                    requested_heads=requested_heads,
                    track_masks=track_masks,
                )
        else:
            return model.inference(
                seq, organism_index,
                requested_heads=requested_heads,
                track_masks=track_masks,
            )

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = run_inference()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    # Timed runs
    times = []
    peak_memories = []
    output_memories = []

    for _ in range(num_runs):
        with track_memory(device) as mem_stats:
            start = time.perf_counter()
            with torch.no_grad():
                outputs = run_inference()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            end = time.perf_counter()

        times.append(end - start)
        peak_memories.append(mem_stats['peak_mb'])
        output_memories.append(measure_output_memory(outputs, device))

        # Clean up
        del outputs
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        'name': name,
        'time_mean': sum(times) / len(times),
        'time_std': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        'peak_memory_mb': sum(peak_memories) / len(peak_memories),
        'output_memory_mb': sum(output_memories) / len(output_memories),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark selective inference')
    parser.add_argument('--seq-len', type=int, default=131072, help='Sequence length (default: 131072 = 128KB)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-warmup', type=int, default=2, help='Number of warmup runs')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of timed runs')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'bfloat16'])
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for optimization')
    args = parser.parse_args()

    device = torch.device(args.device)
    use_autocast = args.dtype == 'bfloat16'

    print(f"Device: {device}")
    print(f"Dtype: {'bfloat16 (autocast)' if use_autocast else 'float32'}")
    print(f"Sequence length: {args.seq_len:,}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Show GPU info
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}")
        print(f"GPU Memory: {props.total_memory / 1024**3:.1f} GB")
        print()

    # Create model (always float32 weights, use autocast for bf16)
    print("Loading model...")
    model = AlphaGenome()
    model.add_reference_heads('human')
    model.to(device=device)
    model.eval()
    set_update_running_var(model, False)

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.1f}M")

    # Create input sequence
    seq = torch.randint(0, 4, (args.batch_size, args.seq_len), device=device)
    organism_index = 0

    # Get number of tracks for rna_seq head
    num_rna_tracks = 768  # From jax_genome_track_heads

    # Create track mask for "single ontology" simulation
    # Simulate filtering to ~10% of tracks (like selecting a single cell type)
    num_filtered_tracks = max(1, num_rna_tracks // 10)
    track_mask = torch.zeros(num_rna_tracks, dtype=torch.bool, device=device)
    track_mask[:num_filtered_tracks] = True

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    # Scenario 1: Full inference (all heads)
    print("\n[1/3] Running: Full inference (all heads)...")
    try:
        result_full = benchmark_scenario(
            model, seq, organism_index,
            name="Full (all heads)",
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            use_autocast=use_autocast,
        )
        print(f"      Done!")
    except torch.cuda.OutOfMemoryError:
        print(f"      OUT OF MEMORY!")
        result_full = None

    # Clean up
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Scenario 2: Single head (rna_seq only)
    print("\n[2/3] Running: Single head (rna_seq only)...")
    try:
        result_single = benchmark_scenario(
            model, seq, organism_index,
            name="Single head (rna_seq)",
            requested_heads={'rna_seq'},
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            use_autocast=use_autocast,
        )
        print(f"      Done!")
    except torch.cuda.OutOfMemoryError:
        print(f"      OUT OF MEMORY!")
        result_single = None

    # Clean up
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Scenario 3: Single head with track mask (ontology filter)
    print(f"\n[3/3] Running: Single head + ontology filter ({num_filtered_tracks}/{num_rna_tracks} tracks)...")
    try:
        result_filtered = benchmark_scenario(
            model, seq, organism_index,
            name=f"rna_seq + filter ({num_filtered_tracks} tracks)",
            requested_heads={'rna_seq'},
            track_masks={'rna_seq': track_mask},
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            use_autocast=use_autocast,
        )
        print(f"      Done!")
    except torch.cuda.OutOfMemoryError:
        print(f"      OUT OF MEMORY!")
        result_filtered = None

    # Print results
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Scenario':<40} {'Time (ms)':<15} {'Peak Mem (GB)':<15} {'Output (MB)':<15}")
    print("-" * 85)

    results = [result_full, result_single, result_filtered]
    for r in results:
        if r is None:
            continue
        time_str = f"{r['time_mean']*1000:.1f} ± {r['time_std']*1000:.1f}"
        peak_str = f"{r['peak_memory_mb']/1024:.2f}"
        output_str = f"{r['output_memory_mb']:.1f}"
        print(f"{r['name']:<40} {time_str:<15} {peak_str:<15} {output_str:<15}")

    # Speedup and memory savings
    print()
    if result_full and result_single:
        speedup = result_full['time_mean'] / result_single['time_mean']
        mem_saving = (result_full['peak_memory_mb'] - result_single['peak_memory_mb']) / 1024
        output_saving = result_full['output_memory_mb'] - result_single['output_memory_mb']
        print(f"Single head vs Full:")
        print(f"  - Speedup: {speedup:.2f}x")
        print(f"  - Peak memory saved: {mem_saving:.2f} GB")
        print(f"  - Output memory saved: {output_saving:.1f} MB")

    if result_single and result_filtered:
        output_saving = result_single['output_memory_mb'] - result_filtered['output_memory_mb']
        print(f"\nOntology filter vs Single head:")
        print(f"  - Output memory saved: {output_saving:.1f} MB ({output_saving/result_single['output_memory_mb']*100:.0f}%)")

    if result_full and result_filtered:
        speedup = result_full['time_mean'] / result_filtered['time_mean']
        mem_saving = (result_full['peak_memory_mb'] - result_filtered['peak_memory_mb']) / 1024
        output_saving = result_full['output_memory_mb'] - result_filtered['output_memory_mb']
        print(f"\nOntology filter vs Full:")
        print(f"  - Speedup: {speedup:.2f}x")
        print(f"  - Peak memory saved: {mem_saving:.2f} GB")
        print(f"  - Output memory saved: {output_saving:.1f} MB")


if __name__ == '__main__':
    main()
