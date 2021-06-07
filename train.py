#!/usr/bin/env python3

"""TorchGeo training script"""

import argparse

import torch

from torchgeo.datasets import get_datasets


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    # Initialize new parser
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Common arguments
    parser.add_argument(
        "-d",
        "--data-dir",
        default="data",
        help="directory containing datasets",
        metavar="DIR",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        default="checkpoints",
        help="directory to save checkpoints to",
        metavar="DIR",
    )
    parser.add_argument(
        "-s", "--seed", default=0, type=int, help="seed for random number generation"
    )
    parser.add_argument(
        "dataset",
        choices=["cv4akenyacroptype", "landcoverai", "vhr10"],
        help="geospatial dataset",
    )
    parser.add_argument(
        "task",
        choices=["detection", "segmentation"]
        help="task to try to perform"
    )
    parser.add_argument("model", choices=["maskrcnn"], help="deep learning model")

    return parser


def main(args: argparse.Namespace) -> None:
    """Main training procedure.

    Parameters:
        args: command-line arguments
    """
    train_dataset, test_dataset = get_datasets(args.dataset, args.data_dir, args.seed)
    train_transforms, test_transforms = get_transforms(args.task)

    train_dataset.transforms = train_transforms
    test_dataset.transforms = test_transforms

    # TODO: add collate_fn, custom loss functions, ConvertCocoPolysToMask, etc.


if __name__ == "__main__":
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    # Set random seed for reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(args.seed)

    # Main training procedure
    main(args)
