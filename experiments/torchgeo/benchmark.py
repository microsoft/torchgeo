#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""dataset and sampler benchmarking script."""

import argparse
import csv
import os
import time

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet34

from torchgeo.datasets import CDL, Landsat8, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomBatchGeoSampler, RandomGeoSampler


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--landsat-root',
        default=os.path.join('data', 'landsat'),
        help='directory containing Landsat data',
        metavar='ROOT',
    )
    parser.add_argument(
        '--cdl-root',
        default=os.path.join('data', 'cdl'),
        help='directory containing CDL data',
        metavar='ROOT',
    )
    parser.add_argument(
        '-d', '--device', default=0, type=int, help='CPU/GPU ID to use', metavar='ID'
    )
    parser.add_argument(
        '-c',
        '--cache',
        action='store_true',
        help='cache file handles during data loading',
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        default=2**4,
        type=int,
        help='number of samples in each mini-batch',
        metavar='SIZE',
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-n',
        '--num-batches',
        type=int,
        help='number of batches to load',
        metavar='SIZE',
    )
    group.add_argument(
        '-e',
        '--epoch-size',
        type=int,
        help='number of samples to load, should be evenly divisible by batch size',
        metavar='SIZE',
    )
    parser.add_argument(
        '-p',
        '--patch-size',
        default=224,
        type=int,
        help='height/width of each patch in pixels',
        metavar='PIXELS',
    )
    parser.add_argument(
        '-s',
        '--stride',
        default=112,
        type=int,
        help='sampling stride for GridGeoSampler in pixels',
        metavar='PIXELS',
    )
    parser.add_argument(
        '-w',
        '--num-workers',
        default=0,
        type=int,
        help='number of workers for parallel data loading',
        metavar='NUM',
    )
    parser.add_argument(
        '--seed', default=0, type=int, help='random seed for reproducibility'
    )
    parser.add_argument(
        '--output-fn',
        default='benchmark-results.csv',
        type=str,
        help='path to the CSV file to write results',
        metavar='FILE',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', help='print results to stdout'
    )

    return parser


def main(args: argparse.Namespace) -> None:
    """High-level pipeline.

    Benchmarks performance of various samplers with and without caching.

    Args:
        args: command-line arguments
    """
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']

    # Benchmark samplers

    # Initialize datasets
    cdl = CDL(args.cdl_root, cache=args.cache)
    landsat = Landsat8(
        args.landsat_root, crs=cdl.crs, res=cdl.res, cache=args.cache, bands=bands
    )
    dataset = landsat & cdl

    # Initialize samplers
    if args.epoch_size:
        length = args.epoch_size
        num_batches = args.epoch_size // args.batch_size
    elif args.num_batches:
        length = args.num_batches * args.batch_size
        num_batches = args.num_batches

    samplers = [
        RandomGeoSampler(landsat, size=args.patch_size, length=length),
        GridGeoSampler(landsat, size=args.patch_size, stride=args.stride),
        RandomBatchGeoSampler(
            landsat, size=args.patch_size, batch_size=args.batch_size, length=length
        ),
    ]

    results_rows = []
    for sampler in samplers:
        if args.verbose:
            print(f'\n{sampler.__class__.__name__}:')

        if isinstance(sampler, RandomBatchGeoSampler):
            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=args.num_workers,
                collate_fn=stack_samples,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                collate_fn=stack_samples,
            )

        tic = time.time()
        num_total_patches = 0
        for i, batch in enumerate(dataloader):
            num_total_patches += args.batch_size
            # This is to stop the GridGeoSampler from enumerating everything
            if i == num_batches - 1:
                break
        toc = time.time()
        duration = toc - tic

        if args.verbose:
            print(f'  duration: {duration:.3f} sec')
            print(f'  count: {num_total_patches} patches')
            print(f'  rate: {num_total_patches / duration:.3f} patches/sec')

        if args.cache:
            if args.verbose:
                print(landsat._cached_load_warp_file.cache_info())

            # Clear cache for fair comparison between samplers
            # Both `landsat` and `cdl` share the same cache
            landsat._cached_load_warp_file.cache_clear()

        results_rows.append(
            {
                'cached': args.cache,
                'seed': args.seed,
                'duration': duration,
                'count': num_total_patches,
                'rate': num_total_patches / duration,
                'sampler': sampler.__class__.__name__,
                'batch_size': args.batch_size,
                'num_workers': args.num_workers,
            }
        )

    # Benchmark model
    model = resnet34()
    # Change number of input channels to match Landsat
    model.conv1 = nn.Conv2d(
        len(bands), 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    criterion = nn.CrossEntropyLoss()
    params = model.parameters()
    optimizer = optim.SGD(params, lr=0.0001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.device)
    model = model.to(device)

    tic = time.time()
    num_total_patches = 0
    for _ in range(num_batches):
        num_total_patches += args.batch_size
        x = torch.rand(args.batch_size, len(bands), args.patch_size, args.patch_size)
        # y = torch.randint(0, 256, (args.batch_size, args.patch_size, args.patch_size))
        y = torch.randint(0, 256, (args.batch_size,))
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()

    toc = time.time()
    duration = toc - tic

    if args.verbose:
        print('\nResNet-34:')
        print(f'  duration: {duration:.3f} sec')
        print(f'  count: {num_total_patches} patches')
        print(f'  rate: {num_total_patches / duration:.3f} patches/sec')

    results_rows.append(
        {
            'cached': args.cache,
            'seed': args.seed,
            'duration': duration,
            'count': num_total_patches,
            'rate': num_total_patches / duration,
            'sampler': 'ResNet-34',
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
        }
    )

    fieldnames = [
        'cached',
        'seed',
        'duration',
        'count',
        'rate',
        'sampler',
        'batch_size',
        'num_workers',
    ]
    if not os.path.exists(args.output_fn):
        with open(args.output_fn, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    with open(args.output_fn, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(results_rows)


if __name__ == '__main__':
    parser = set_up_parser()
    args = parser.parse_args()

    if args.epoch_size:
        assert args.epoch_size % args.batch_size == 0

    pl.seed_everything(args.seed)

    main(args)
