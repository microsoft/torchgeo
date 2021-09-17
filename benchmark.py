#!/usr/bin/env python3

"""dataset and sampler benchmarking script."""

import argparse
import csv
import os
import time

import pytorch_lightning as pl
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-n",
        "--num-batches",
        type=int,
        help="number of batches to load",
        metavar="SIZE",
    )
    group.add_argument(
        "-e",
        "--epoch-size",
        type=int,
        help="number of samples to load, should be evenly divisible by batch size",
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
    parser.add_argument(
        "--output-fn",
        default="benchmark-results.csv",
        type=str,
        help="path to the CSV file to write results",
        metavar="FILE",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print results to stdout",
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
    landsat = Landsat8(
        args.landsat_root,
        crs,
        res,
        cache=args.cache,
        bands=[
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
        ],
    )
    cdl = CDL(args.cdl_root, crs, res, cache=args.cache)
    dataset = landsat + cdl

    # Initialize samplers
    if args.epoch_size:
        length = args.epoch_size
        num_batches = args.epoch_size // args.batch_size
    elif args.num_batches:
        length = args.num_batches * args.batch_size
        num_batches = args.num_batches

    samplers = [
        RandomGeoSampler(
            landsat.index,
            size=args.patch_size,
            length=length,
        ),
        GridGeoSampler(landsat.index, size=args.patch_size, stride=args.stride),
        RandomBatchGeoSampler(
            landsat.index,
            size=args.patch_size,
            batch_size=args.batch_size,
            length=length,
        ),
    ]

    results_rows = []
    for sampler in samplers:
        if args.verbose:
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
        num_total_patches = 0
        for i, batch in enumerate(dataloader):
            num_total_patches += args.batch_size
            # This is to stop the GridGeoSampler from enumerating everything
            if i == num_batches - 1:
                break
        toc = time.time()
        duration = toc - tic

        if args.verbose:
            print(f"  duration: {duration:.3f} sec")
            print(f"  count: {num_total_patches} patches")
            print(f"  rate: {num_total_patches / duration:.3f} patches/sec")

        if args.cache:
            if args.verbose:
                print(landsat._cached_load_warp_file.cache_info())
                print(cdl._cached_load_warp_file.cache_info())

            # Clear cache for fair comparison between samplers
            landsat._cached_load_warp_file.cache_clear()
            cdl._cached_load_warp_file.cache_clear()

        results_rows.append(
            {
                "cached": args.cache,
                "seed": args.seed,
                "duration": duration,
                "count": num_total_patches,
                "rate": num_total_patches / duration,
                "sampler": sampler.__class__.__name__,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            }
        )

    fieldnames = [
        "cached",
        "seed",
        "duration",
        "count",
        "rate",
        "sampler",
        "batch_size",
        "num_workers",
    ]
    if not os.path.exists(args.output_fn):
        with open(args.output_fn, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    with open(args.output_fn, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(results_rows)


if __name__ == "__main__":
    os.environ["GDAL_CACHEMAX"] = "50%"

    parser = set_up_parser()
    args = parser.parse_args()

    if args.epoch_size:
        assert args.epoch_size % args.batch_size == 0

    pl.seed_everything(args.seed)

    main(args)
