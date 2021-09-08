#!/usr/bin/env python3

"""dataset and sampler benchmarking script."""

import argparse
import os
import random
import time

from rasterio.crs import CRS
from torch.utils.data import DataLoader

from torchgeo.datasets import CDL, Landsat8
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
        "--landsat-root",
        default=os.path.join("data", "landsat"),
        help="directory containing Landsat data",
        metavar="ROOT",
    )
    parser.add_argument(
        "--cdl-root",
        default=os.path.join("data", "cdl"),
        help="directory containing CDL data",
        metavar="ROOT",
    )
    parser.add_argument(
        "-c",
        "--cache",
        action="store_true",
        help="cache file handles during data loading",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=2 ** 4,
        type=int,
        help="number of samples in each mini-batch",
        metavar="SIZE",
    )
    parser.add_argument(
        "-e",
        "--epoch-size",
        default=2 ** 10,
        type=int,
        help="number of samples in each epoch",
        metavar="SIZE",
    )
    parser.add_argument(
        "-p",
        "--patch-size",
        default=2 ** 8,
        type=int,
        help="height/width of each patch",
        metavar="SIZE",
    )
    parser.add_argument(
        "-s",
        "--stride",
        default=2 ** 7,
        type=int,
        help="sampling stride for GridGeoSampler",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        default=0,
        type=int,
        help="number of workers for parallel data loading",
        metavar="NUM",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed for reproducibility",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    """High-level pipeline.

    Benchmarks performance of various samplers with and without caching.

    Args:
        args: command-line arguments
    """
    # Initialize datasets
    crs = CRS.from_epsg(32610)  # UTM, Zone 10
    res = 15
    landsat = Landsat8(args.landsat_root, crs, res, cache=args.cache)
    cdl = CDL(args.cdl_root, crs, res, cache=args.cache)
    dataset = landsat + cdl

    # Initialize samplers
    samplers = [
        RandomGeoSampler(
            landsat.index,
            size=args.patch_size,
            length=args.epoch_size // args.batch_size,
        ),
        GridGeoSampler(landsat.index, size=args.patch_size, stride=args.stride),
        RandomBatchGeoSampler(
            landsat.index,
            size=args.patch_size,
            batch_size=args.batch_size,
            length=args.epoch_size // args.batch_size,
        ),
    ]
    for sampler in samplers:
        print(f"\n{sampler.__class__.__name__}:")

        if isinstance(sampler, RandomBatchGeoSampler):
            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,  # type: ignore[arg-type]
                num_workers=args.num_workers,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,  # type: ignore[arg-type]
                num_workers=args.num_workers,
            )

        tic = time.time()
        patches = 0
        for batch in dataloader:
            patches += len(batch)
        toc = time.time()
        duration = toc - tic
        print(f"  duration: {duration:.3f} s")
        print(f"  count: {patches}")
        print(f"  rate: {patches / duration} patches/sec")


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()

    random.seed(args.seed)

    main(args)
