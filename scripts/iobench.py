#!/usr/bin/env python3
# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Benchmark iteration over the IOBench dataset.

For each requested split (``raw`` and/or ``preprocessed``), this script builds
an :class:`~torchgeo.datasets.IOBench` dataset, wraps it in a
:class:`~torchgeo.samplers.GridGeoSampler` (non-overlapping patches), feeds it
through a :class:`torch.utils.data.DataLoader`, and reports timing information
for one full epoch.

Example usage::

    python scripts/iobench.py --root data/io --batch-size 32 --patch-size 256
"""

import argparse
import time
from typing import Literal

from torch.utils.data import DataLoader
from tqdm import tqdm

from torchgeo.datasets import IOBench, stack_samples
from torchgeo.samplers import GridGeoSampler


def benchmark_split(
    split: Literal['raw', 'preprocessed'],
    root: str,
    batch_size: int,
    patch_size: int,
    num_workers: int,
) -> None:
    """Benchmark one full epoch over a single ``IOBench`` split.

    Args:
        split: Dataset split to benchmark.
        root: Root directory where dataset can be found.
        batch_size: Number of patches per mini-batch.
        patch_size: Size of each square patch in pixels.
        num_workers: Number of dataloader worker processes.
    """
    print(f'\n=== split={split} ===')
    dataset = IOBench(root=root, split=split, download=True, checksum=True)
    sampler = GridGeoSampler(dataset, size=patch_size, stride=patch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=stack_samples,
    )

    num_samples = len(sampler)
    num_batches = 0
    start = time.perf_counter()
    for _ in tqdm(dataloader, total=len(dataloader), desc=split):
        num_batches += 1
    elapsed = time.perf_counter() - start

    print(f'samples:       {num_samples}')
    print(f'batches:       {num_batches}')
    print(f'total time:    {elapsed:.3f} s')
    print(f'samples/sec:   {num_samples / elapsed:.2f}')
    print(f'batches/sec:   {num_batches / elapsed:.2f}')


def main() -> None:
    """Parse CLI args and run the benchmark for each requested split."""
    parser = argparse.ArgumentParser(
        description='Benchmark iteration over the IOBench dataset.'
    )
    parser.add_argument(
        '--root',
        default='data/io',
        help='root directory where IOBench data can be found',
    )
    parser.add_argument('--batch-size', type=int, default=32, help='mini-batch size')
    parser.add_argument(
        '--patch-size', type=int, default=256, help='patch size in pixels'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='number of dataloader worker processes',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        choices=['raw', 'preprocessed'],
        default=['raw', 'preprocessed'],
        help='which IOBench splits to benchmark',
    )
    args = parser.parse_args()

    print('IOBench benchmark')
    print(
        f'root={args.root} batch_size={args.batch_size}'
        f' patch_size={args.patch_size} num_workers={args.num_workers}'
    )

    for split in args.splits:
        benchmark_split(
            split=split,
            root=args.root,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            num_workers=args.num_workers,
        )


if __name__ == '__main__':
    main()
