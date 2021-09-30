#!/usr/bin/env python3

"""Script for comparing execute time cpu/gpu transforms+augmentations."""
import argparse
import csv
import os
import time

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchgeo.transforms import AugmentationSequential, indices


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        type=str,
        help="CPU or GPU",
    )
    parser.add_argument(
        "-c",
        "--channels",
        type=int,
        default=24,
        help="number of channels in the image",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=int,
        default=128,
        help="image shape",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        help="number of samples in each mini-batch",
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
        "-m",
        "--mask",
        action="store_true",
        help="Benchmark with masks in the batch",
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

    Benchmarks execution time of CPU/GPU transforms & augmentations
    ​
    Args:
        args: command-line arguments
    """
    device = torch.device(args.device)  # type: ignore[attr-defined]
    image = torch.randn(args.batch_size, args.channels, *(args.shape, args.shape))
    image = image.to(device)
    batch = dict(image=image)
    data_keys = ["image"]

    if args.mask:
        mask = torch.randint(  # type: ignore[attr-defined]
            0, 10, size=(args.batch_size, 1, *(args.shape, args.shape))
        )
        mask = mask.to(torch.float).to(device)  # type: ignore[attr-defined]
        batch["mask"] = mask
        data_keys.append("mask")

    augmentations = AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0),
        K.RandomVerticalFlip(p=1.0),
        K.RandomAffine(degrees=(0, 90), p=1.0),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=1.0),
        K.RandomResizedCrop(size=(args.shape, args.shape), scale=(0.8, 1.0), p=1.0),
        data_keys=data_keys,
    )
    transforms = nn.Sequential(  # type: ignore[attr-defined]
        indices.AppendNDBI(index_swir=0, index_nir=1),
        indices.AppendNDSI(index_green=0, index_swir=1),
        indices.AppendNDVI(index_red=0, index_nir=1),
        indices.AppendNDWI(index_green=0, index_nir=1),
        augmentations,
    ).to(device)

    tic = time.time()
    _ = transforms(batch)
    toc = tic = time.time()
    duration = toc - tic

    if args.verbose:
        print(f"  duration: {duration:.3f} sec")

    fieldnames = [
        "device",
        "seed",
        "duration",
        "batch_size",
        "channels",
        "image_shape",
        "mask",
    ]
    results = dict(
        device=args.device,
        seed=args.seed,
        duration=duration,
        batch_size=args.batch_size,
        channels=args.channels,
        image_shape=args.shape,
        mask=args.mask,
    )
    if not os.path.exists(args.output_fn):
        with open(args.output_fn, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with open(args.output_fn, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(results)


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    main(args)
