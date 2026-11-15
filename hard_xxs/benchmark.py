#!/usr/bin/env python3
"""FP32 batch-one forward benchmark, including full-resolution class logits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone

import torch

try:
    from .model import Config, HARDXXS
except ImportError:
    from model import Config, HARDXXS


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def telemetry():
    result = {'utc': datetime.now(timezone.utc).isoformat()}
    for key, command in {
        'throttled': ['vcgencmd', 'get_throttled'],
        'temperature': ['vcgencmd', 'measure_temp'],
        'clock': ['vcgencmd', 'measure_clock', 'arm'],
    }.items():
        try:
            result[key] = subprocess.check_output(command, text=True, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            result[key] = None
    return result


def percentile(values, fraction):
    values = sorted(values)
    i = (len(values) - 1) * fraction
    lo = int(i)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height', type=positive, default=360)
    parser.add_argument('--width', type=positive, default=640)
    parser.add_argument('--classes', type=positive, default=19)
    parser.add_argument('--threads', type=positive, default=2)
    parser.add_argument('--cpus', help='Linux CPU affinity, for example 2,3')
    parser.add_argument('--warmup', type=positive, default=30)
    parser.add_argument('--iterations', type=positive, default=1000)
    parser.add_argument('--noise-images', type=positive, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.cpus:
        if not hasattr(os, 'sched_setaffinity'):
            parser.error('--cpus requires Linux')
        try:
            requested = {int(x) for x in args.cpus.split(',')}
            os.sched_setaffinity(0, requested)
            if set(os.sched_getaffinity(0)) != requested:
                raise ValueError('affinity did not match request')
        except (ValueError, OSError) as error:
            parser.error(str(error))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    model = HARDXXS(Config(classes=args.classes)).cpu().eval()
    inputs = [torch.rand(1, 3, args.height, args.width) for _ in range(args.noise_images)]
    samples, conditions = [], [telemetry()]
    last_sample = time.monotonic()
    with torch.inference_mode():
        output = model(inputs[0])
        assert output.shape == (1, args.classes, args.height, args.width)
        assert torch.isfinite(output).all()
        del output
        for i in range(args.warmup):
            output = model(inputs[i % len(inputs)])
            del output
        for i in range(args.iterations):
            image = inputs[i % len(inputs)]
            start = time.perf_counter_ns()
            output = model(image)
            samples.append((time.perf_counter_ns() - start) / 1e6)
            del output
            if time.monotonic() - last_sample >= 10:
                conditions.append(telemetry())
                last_sample = time.monotonic()
                print(f'{i + 1}/{args.iterations}', flush=True)
    conditions.append(telemetry())
    thread_affinity = {}
    if hasattr(os, 'sched_getaffinity'):
        for task in Path('/proc/self/task').iterdir():
            try:
                thread_affinity[task.name] = sorted(os.sched_getaffinity(int(task.name)))
            except ProcessLookupError:
                pass
        if args.cpus and any(not set(cpus).issubset(requested)
                             for cpus in thread_affinity.values()):
            raise RuntimeError('a native thread escaped requested CPUs; launch with taskset')
    root = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ('model.py', 'benchmark.py')}
    mean = statistics.mean(samples)
    result = {
        'model': 'HARD-XXS reconstruction', 'official_equivalence_verified': False,
        'weights': 'random; no accuracy claim', 'configuration': model.configuration(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'environment': {'host': platform.node(), 'platform': platform.platform(),
                        'python': platform.python_version(), 'torch': torch.__version__,
                        'cpu_affinity': sorted(os.sched_getaffinity(0))
                        if hasattr(os, 'sched_getaffinity') else None,
                        'intraop_threads': torch.get_num_threads(),
                        'interop_threads': torch.get_num_interop_threads(),
                        'native_thread_affinity': thread_affinity,
                        'torch_build': torch.__config__.show()},
        'protocol': {'batch': 1, 'precision': 'FP32', 'runtime': 'PyTorch eager CPU',
                     'input_shape': [1, 3, args.height, args.width],
                     'output_shape': [1, args.classes, args.height, args.width],
                     'warmup': args.warmup, 'iterations': args.iterations,
                     'noise_images': args.noise_images, 'seed': args.seed,
                     'memory_format': 'contiguous', 'conv_bn_fused': False,
                     'scope': 'forward including logit resize; excludes preprocessing and argmax'},
        'sources_sha256': sources, 'mean_ms': mean, 'median_ms': statistics.median(samples),
        'p90_ms': percentile(samples, .9), 'p99_ms': percentile(samples, .99),
        'fps': 1000 / mean, 'samples_ms': samples, 'conditions': conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('model', 'parameters', 'mean_ms', 'fps')}, indent=2))


if __name__ == '__main__':
    main()
