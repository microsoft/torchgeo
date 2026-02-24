#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Benchmark script comparing VectorDataset and SedonaDBDataset performance."""

import argparse
import json
import os
import random
import time

import numpy as np
import torch.nn as nn
from tqdm import tqdm

from torchgeo.datasets import SedonaDBDataset, VectorDataset


class WashingtonVectorDataset(VectorDataset):
    """VectorDataset for Washington.parquet."""

    filename_glob = '*.parquet'
    filename_regex = '.*'


class WashingtonSedonaDBDataset(SedonaDBDataset):
    """SedonaDBDataset for Washington.parquet."""

    filename_glob = '*.parquet'
    filename_regex = '.*'


def generate_random_slices(
    bounds: tuple[slice, slice, slice], num_slices: int, seed: int | None = None
) -> list[tuple[slice, slice, slice]]:
    """Generate random slices within the dataset bounds.

    Args:
        bounds: (x_slice, y_slice, t_slice) bounds of the dataset
        num_slices: number of random slices to generate
        seed: random seed for reproducibility

    Returns:
        list of random (x, y, t) slices
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    x_slice, y_slice, t_slice = bounds
    xmin, xmax = x_slice.start, x_slice.stop
    ymin, ymax = y_slice.start, y_slice.stop
    tmin, tmax = t_slice.start, t_slice.stop

    slices = []
    for _ in range(num_slices):
        x_size = random.uniform(0.01, 0.1)
        y_size = random.uniform(0.01, 0.1)

        x_start = random.uniform(xmin, xmax - x_size)
        x_stop = x_start + x_size

        y_start = random.uniform(ymin, ymax - y_size)
        y_stop = y_start + y_size

        x = slice(x_start, x_stop, x_slice.step)
        y = slice(y_start, y_stop, y_slice.step)
        t = slice(tmin, tmax, t_slice.step)

        slices.append((x, y, t))

    return slices


def initialize_dataset(
    dataset_class: type[VectorDataset | SedonaDBDataset],
    name: str,
    parquet_path: str,
    transforms: nn.Module,
    verbose: bool = False,
) -> tuple[VectorDataset | SedonaDBDataset, float]:
    """Initialize a dataset and measure initialization time.

    Args:
        dataset_class: dataset class to instantiate
        name: name of the dataset for logging
        parquet_path: path to the parquet file
        transforms: transforms to apply
        verbose: whether to print verbose output

    Returns:
        tuple of (dataset, init_time)
    """
    if verbose:
        print(f'Initializing {name}...')
    start = time.perf_counter()
    dataset = dataset_class(
        paths=parquet_path, res=(0.0001, 0.0001), transforms=transforms
    )
    init_time = time.perf_counter() - start

    if verbose:
        print(f'{name} initialized in {init_time:.3f}s')
        print(f'  Bounds: {dataset.bounds}')
        print(f'  CRS: {dataset.crs}')
        print(f'  Size: {len(dataset)}')

    return dataset, init_time


def benchmark_filter_index(
    dataset: VectorDataset | SedonaDBDataset,
    slices: list[tuple[slice, slice, slice]],
    warmup: int = 3,
    verbose: bool = False,
) -> tuple[float, int]:
    """Benchmark filter_index operations.

    Args:
        dataset: dataset to benchmark
        slices: list of slices to query
        warmup: number of warmup iterations
        verbose: whether to show progress bar

    Returns:
        tuple of (total_time, num_geometries)
    """
    if verbose:
        warmup_iter = tqdm(range(warmup), desc='Warmup', leave=False)
    else:
        warmup_iter = range(warmup)

    for i in warmup_iter:
        try:
            x, y, t = slices[i % len(slices)]
            _ = dataset.filter_index((x, y, t))
        except IndexError:
            pass

    total_time = 0.0
    total_geometries = 0

    if verbose:
        slice_iter = tqdm(slices, desc='Benchmarking')
    else:
        slice_iter = slices

    for x, y, t in slice_iter:
        start = time.perf_counter()
        try:
            shapes = dataset.filter_index((x, y, t))
            total_geometries += len(shapes)
        except IndexError:
            pass
        end = time.perf_counter()
        total_time += end - start

    return total_time, total_geometries


def create_result_row(
    dataset_name: str,
    init_time: float,
    filter_time: float,
    num_slices: int,
    num_geometries: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Create a result row dictionary.

    Args:
        dataset_name: name of the dataset
        init_time: initialization time in seconds
        filter_time: total filter time in seconds
        num_slices: number of slices processed
        num_geometries: total number of geometries found
        seed: random seed used

    Returns:
        result row dictionary
    """
    return {
        'dataset': dataset_name,
        'init_time': init_time,
        'filter_time': filter_time,
        'num_slices': num_slices,
        'num_geometries': num_geometries,
        'time_per_slice': filter_time / num_slices,
        'geometries_per_second': num_geometries / filter_time if filter_time > 0 else 0,
        'seed': seed,
    }


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--parquet-path',
        default='Washington.parquet',
        help='path to Washington.parquet file',
        metavar='PATH',
    )
    parser.add_argument(
        '-n',
        '--num-slices',
        default=100,
        type=int,
        help='number of random slices to sample',
        metavar='NUM',
    )
    parser.add_argument(
        '--seed', default=0, type=int, help='random seed for reproducibility'
    )
    parser.add_argument(
        '--output-fn',
        default='vector-dataset-benchmark-results.json',
        type=str,
        help='path to the JSON file to write results',
        metavar='FILE',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='print results to stdout'
    )
    parser.add_argument(
        '--dataset',
        choices=['vector', 'sedona', 'both'],
        default='sedona',
        help='which dataset(s) to benchmark',
    )

    return parser


def main(args: argparse.Namespace) -> None:
    """High-level pipeline.

    Benchmarks performance of VectorDataset vs SedonaDBDataset.

    Args:
        args: command-line arguments
    """
    if not os.path.exists(args.parquet_path):
        raise FileNotFoundError(f'Parquet file not found: {args.parquet_path}')

    transforms = nn.Identity()
    results_rows = []
    speedup = None

    if args.dataset in ['vector', 'both']:
        vector_dataset, vector_init_time = initialize_dataset(
            WashingtonVectorDataset,
            'VectorDataset',
            args.parquet_path,
            transforms,
            args.verbose,
        )
        dataset_for_slices = vector_dataset
    else:
        vector_dataset = None
        vector_init_time = 0.0

    if args.dataset in ['sedona', 'both']:
        if args.verbose and args.dataset == 'both':
            print()
        sedona_dataset, sedona_init_time = initialize_dataset(
            WashingtonSedonaDBDataset,
            'SedonaDBDataset',
            args.parquet_path,
            transforms,
            args.verbose,
        )
        if args.dataset == 'sedona':
            dataset_for_slices = sedona_dataset
    else:
        sedona_dataset = None
        sedona_init_time = 0.0

    if args.verbose:
        print(f'\nGenerating {args.num_slices} random slices...')
    slices = generate_random_slices(
        dataset_for_slices.bounds, args.num_slices, args.seed
    )

    if args.dataset in ['vector', 'both']:
        if args.verbose:
            print('Benchmarking VectorDataset...')
        assert vector_dataset is not None
        vector_time, vector_geometries = benchmark_filter_index(
            vector_dataset, slices, verbose=args.verbose
        )
        results_rows.append(
            create_result_row(
                'VectorDataset',
                vector_init_time,
                vector_time,
                args.num_slices,
                vector_geometries,
                args.seed,
            )
        )

    if args.dataset in ['sedona', 'both']:
        if args.verbose:
            print('Benchmarking SedonaDBDataset...')
        assert sedona_dataset is not None
        sedona_time, sedona_geometries = benchmark_filter_index(
            sedona_dataset, slices, verbose=args.verbose
        )
        results_rows.append(
            create_result_row(
                'SedonaDBDataset',
                sedona_init_time,
                sedona_time,
                args.num_slices,
                sedona_geometries,
                args.seed,
            )
        )

        if args.dataset == 'both' and vector_dataset is not None:
            speedup = vector_time / sedona_time if sedona_time > 0 else 0.0
            if args.verbose:
                print(f'\nSpeedup: {speedup:.2f}x')

    if args.verbose:
        print('\nResults:')
        for row in results_rows:
            print(f'  {row["dataset"]}:')
            print(f'    Init time: {row["init_time"]:.3f}s')
            print(f'    Filter time: {row["filter_time"]:.3f}s')
            print(f'    Time per slice: {row["time_per_slice"]:.4f}s')
            print(f'    Geometries: {row["num_geometries"]}')
            print(f'    Geometries/sec: {row["geometries_per_second"]:.2f}')

    output_data = {
        'parquet_path': args.parquet_path,
        'num_slices': args.num_slices,
        'seed': args.seed,
        'results': results_rows,
    }

    if len(results_rows) == 2 and speedup is not None:
        output_data['speedup'] = speedup

    with open(args.output_fn, 'w') as f:
        json.dump(output_data, f, indent=2)


if __name__ == '__main__':
    parser = set_up_parser()
    args = parser.parse_args()
    main(args)
