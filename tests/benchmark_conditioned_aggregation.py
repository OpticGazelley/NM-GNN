"""Compare original and optimized S2 on synthetic sampled graphs, not full training."""

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'genagg')]

import torch

from conditioned_reference import original_conditioned_forward
from models import MultiEdgeAggModule


def benchmark(args, mixed_fraction):
    device = torch.device(args.device)
    torch.manual_seed(42)
    pairs = torch.arange(args.edges, device=device) // 4
    pair_count = (args.edges + 3) // 4
    edge_index = torch.stack([pairs, pairs + 1])
    edge_cat = torch.zeros((args.edges, 3), device=device, dtype=torch.long)
    mixed = pairs < int(pair_count * mixed_fraction)
    edge_cat[:, 2] = torch.where(mixed, torch.arange(args.edges, device=device) % 4, 0)
    features = torch.randn(args.edges, 64, device=device)
    original = MultiEdgeAggModule(64, 'gin', conditioned_by_payment=True, num_payment_formats=7).to(device)
    optimized = copy.deepcopy(original)
    inputs = {name: features.clone().requires_grad_() for name in ('original', 'optimized')}
    modules = {'original': original, 'optimized': optimized}
    timings = {name: [] for name in modules}
    peaks = {name: [] for name in modules}

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def step(name, measure=False):
        module = modules[name]
        module.zero_grad(set_to_none=True)
        inputs[name].grad = None
        if measure and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            initial_memory = torch.cuda.memory_allocated(device)
        synchronize()
        started = time.perf_counter()
        arguments = (edge_index, inputs[name], pairs, None, edge_cat)
        out = original_conditioned_forward(module, *arguments) if name == 'original' else module(*arguments)
        (out[1].square().mean() + out[3].square().mean()).backward()
        synchronize()
        if measure:
            timings[name].append((time.perf_counter() - started) * 1000)
            if device.type == 'cuda':
                peaks[name].append((torch.cuda.max_memory_allocated(device) - initial_memory) / 2**20)

    for _ in range(2):
        step('original')
        step('optimized')
    for repetition in range(args.repeats):
        order = ('original', 'optimized') if repetition % 2 == 0 else ('optimized', 'original')
        for name in order:
            step(name, measure=True)
    result = {'edges': args.edges, 'mixed_pair_fraction': mixed_fraction,
              'median_step_ms': {name: round(statistics.median(values), 3) for name, values in timings.items()},
              'peak_increment_mib': {name: round(max(values), 2) if values else None for name, values in peaks.items()}}
    result['speedup'] = round(statistics.median(timings['original']) / statistics.median(timings['optimized']), 2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--edges', type=int, default=12000)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.edges < 4 or args.repeats < 1:
        parser.error('Use at least four edges and one repetition')
    torch.set_num_threads(1)
    print(json.dumps({'torch': torch.__version__, 'device': args.device,
                      'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None}), flush=True)
    for fraction in (0.0, 0.1, 0.5, 1.0):
        benchmark(args, fraction)
