#!/usr/bin/env python3
"""Analyze peak memory composition at each stage of AlphaGenome inference.

Breaks down memory usage into:
- Model weights
- DNA embedding
- Encoder (downsampling)
- Transformer tower
- Decoder (upsampling)
- Output embeddings
- Head outputs
"""

import argparse
import gc

import torch

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.alphagenome import set_update_running_var


def format_bytes(nbytes):
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.2f} GB"
    elif nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.1f} MB"
    else:
        return f"{nbytes / 1024:.1f} KB"


def get_memory_stats(device):
    torch.cuda.synchronize(device)
    return {
        'allocated': torch.cuda.memory_allocated(device),
        'reserved': torch.cuda.memory_reserved(device),
        'peak': torch.cuda.max_memory_allocated(device),
    }


def reset_peak(device):
    torch.cuda.reset_peak_memory_stats(device)


def analyze_memory(model, seq, organism_index, device, use_autocast=False):
    """Analyze memory at each stage of inference."""

    results = []

    # Initial state
    gc.collect()
    torch.cuda.empty_cache()
    reset_peak(device)

    initial = get_memory_stats(device)
    results.append(('Initial (model loaded)', initial['allocated'], initial['peak']))

    # Get internal modules
    transformer_unet = model.transformer_unet
    dna_embed = transformer_unet.dna_embed
    downs = transformer_unet.downs
    transformer = transformer_unet.transformer
    ups = transformer_unet.ups

    organism_embed_mod = model.organism_embed
    outembed_1bp = model.outembed_1bp
    outembed_128bp = model.outembed_128bp
    outembed_pair = model.outembed_pair

    from einops import rearrange
    from einx import add

    context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_autocast else torch.no_grad()

    with torch.no_grad(), context:
        # Organism embedding
        organism_index_tensor = torch.full((seq.shape[0],), organism_index, device=device)
        organism_embed = organism_embed_mod(organism_index_tensor)

        stats = get_memory_stats(device)
        results.append(('After organism embed', stats['allocated'], stats['peak']))

        # DNA embedding
        reset_peak(device)
        dna_out, skip_first = dna_embed(seq)

        stats = get_memory_stats(device)
        results.append(('After DNA embed', stats['allocated'], stats['peak']))
        print(f"  DNA embed output: {dna_out.shape}, skip: {skip_first.shape}")

        # Encoder (downs)
        skips = [skip_first]
        x = dna_out

        reset_peak(device)
        for i, down in enumerate(downs):
            x, skip = down(x, return_pre_pool=True)
            skips.append(skip)

        stats = get_memory_stats(device)
        results.append(('After encoder (downs)', stats['allocated'], stats['peak']))
        print(f"  Encoder output: {x.shape}")

        # Prepare for transformer
        x = rearrange(x, 'b d n -> b n d')
        x = add('b n d, b d', x, organism_embed)

        # Transformer
        reset_peak(device)
        single, pairwise = transformer(x)

        stats = get_memory_stats(device)
        results.append(('After transformer', stats['allocated'], stats['peak']))
        print(f"  Transformer single: {single.shape}, pairwise: {pairwise.shape}")

        # Decoder (ups)
        reset_peak(device)
        x = rearrange(single, 'b n d -> b d n')
        for up in ups:
            skip = skips.pop()
            x = up(x, skip=skip)

        unet_out = rearrange(x, 'b d n -> b n d')

        stats = get_memory_stats(device)
        results.append(('After decoder (ups)', stats['allocated'], stats['peak']))
        print(f"  Decoder output: {unet_out.shape}")

        # Output embeddings
        reset_peak(device)
        embeds_128bp = outembed_128bp(single, organism_index_tensor)
        embeds_1bp = outembed_1bp(unet_out, organism_index_tensor, embeds_128bp)
        embeds_pair = outembed_pair(pairwise, organism_index_tensor)

        stats = get_memory_stats(device)
        results.append(('After output embeds', stats['allocated'], stats['peak']))
        print(f"  embeds_1bp: {embeds_1bp.shape}")
        print(f"  embeds_128bp: {embeds_128bp.shape}")
        print(f"  embeds_pair: {embeds_pair.shape}")

        # Calculate embedding sizes
        embed_bytes = (
            embeds_1bp.numel() * embeds_1bp.element_size() +
            embeds_128bp.numel() * embeds_128bp.element_size() +
            embeds_pair.numel() * embeds_pair.element_size()
        )
        print(f"  Total embedding size: {format_bytes(embed_bytes)}")

        # Head outputs (full)
        reset_peak(device)
        head_outputs = {}
        for org, heads in model.heads.items():
            head_outputs[org] = {}
            for head_name, head in heads.items():
                head_args = model.head_forward_arg_names[org][head_name]
                kwargs = {}
                for arg in head_args:
                    if arg == 'embeds_1bp':
                        kwargs[arg] = embeds_1bp
                    elif arg == 'embeds_128bp':
                        kwargs[arg] = embeds_128bp
                    elif arg == 'embeds_pair':
                        kwargs[arg] = embeds_pair
                    elif arg == 'organism_index':
                        kwargs[arg] = organism_index_tensor
                    elif arg == 'splice_site_positions':
                        kwargs[arg] = None
                head_outputs[org][head_name] = head(**kwargs)

        stats = get_memory_stats(device)
        results.append(('After all heads', stats['allocated'], stats['peak']))

        # Calculate output sizes
        def count_output_bytes(obj):
            total = 0
            if isinstance(obj, torch.Tensor):
                total += obj.numel() * obj.element_size()
            elif isinstance(obj, dict):
                for v in obj.values():
                    total += count_output_bytes(v)
            return total

        output_bytes = count_output_bytes(head_outputs)
        print(f"  Total head output size: {format_bytes(output_bytes)}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-len', type=int, default=524288)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    args = parser.parse_args()

    device = torch.device(args.device)
    use_autocast = args.dtype == 'bfloat16'

    print("=" * 70)
    print("MEMORY COMPOSITION ANALYSIS")
    print("=" * 70)
    print(f"Sequence length: {args.seq_len:,} bp ({args.seq_len // 1024}KB)")
    print(f"Device: {device}")
    print(f"Dtype: {'bfloat16 (autocast)' if use_autocast else 'float32'}")

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

    # Model size
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Model size: {format_bytes(model_bytes)}")
    print()

    # Input
    seq = torch.randint(0, 4, (1, args.seq_len), device=device)

    print("=" * 70)
    print("STAGE-BY-STAGE MEMORY ANALYSIS")
    print("=" * 70)
    print()

    try:
        results = analyze_memory(model, seq, 0, device, use_autocast)

        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print(f"{'Stage':<30} {'Allocated':<15} {'Peak':<15} {'Delta':<15}")
        print("-" * 75)

        prev_alloc = 0
        for name, alloc, peak in results:
            delta = alloc - prev_alloc
            delta_str = f"+{format_bytes(delta)}" if delta > 0 else format_bytes(delta)
            print(f"{name:<30} {format_bytes(alloc):<15} {format_bytes(peak):<15} {delta_str:<15}")
            prev_alloc = alloc

        # Overall peak
        print()
        overall_peak = max(r[2] for r in results)
        print(f"Overall peak memory: {format_bytes(overall_peak)}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"OUT OF MEMORY: {e}")


if __name__ == '__main__':
    main()
