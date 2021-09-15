#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""script for testing saved models on different ChesapeakeCVPR dataset splits."""

import argparse
import csv
import os

import pytorch_lightning as pl

from torchgeo.trainers import ChesapeakeCVPRDataModule, ChesapeakeCVPRSegmentationTask

STATES = ["de", "md", "va", "wv", "pa", "ny"]


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="directory containing the experiment run directories",
        metavar="ROOT",
    )
    parser.add_argument(
        "--chesapeakecvpr-root",
        required=True,
        type=str,
        help="directory containing the ChesapeakeCVPR dataset",
        metavar="ROOT",
    )
    parser.add_argument(
        "--output-fn",
        default="chesapeakecvpr-results.csv",
        type=str,
        help="path to the CSV file to write results",
        metavar="FILE",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    """High-level pipeline.

    Args:
        args: command-line arguments
    """
    if os.path.exists(args.output_fn):
        print(f"The output file {args.output_fn} already exists, exiting...")
        return

    # Set up the result file
    fieldnames = [
        "train-state",
        "model",
        "learning-rate",
        "initialization",
        "test-state",
        "acc",
        "iou",
    ]
    with open(args.output_fn, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Test loop
    trainer = pl.Trainer(
        gpus=1, logger=False, progress_bar_refresh_rate=0, checkpoint_callback=False
    )

    for experiment_dir in os.listdir(args.input_dir):

        checkpoint_fn = os.path.join(args.input_dir, experiment_dir, "last.ckpt")
        try:

            model = ChesapeakeCVPRSegmentationTask.load_from_checkpoint(checkpoint_fn)
            model.freeze()
            model.eval()
        except KeyError:
            print(
                f"Skipping {experiment_dir} as we are not able to load a valid"
                + f" ChesapeakeCVPRSegmentationTask from {checkpoint_fn}"
            )
            continue

        try:
            experiment_dir_parts = experiment_dir.split("_")
            train_state = experiment_dir_parts[0]
            model_name = experiment_dir_parts[1]
            learning_rate = experiment_dir_parts[2]
            initialization = "random" if len(experiment_dir_parts) == 4 else "imagenet"
        except IndexError:
            print(
                f"Skipping {experiment_dir} as the directory name is not in the"
                + " expected format"
            )
            continue

        # Test the loaded model against the test set from all states
        for state in STATES:

            dm = ChesapeakeCVPRDataModule(
                args.chesapeakecvpr_root,
                train_state=f"{state}",
                batch_size=32,
                num_workers=8,
            )
            results = trainer.test(model=model, datamodule=dm, verbose=False)
            print(experiment_dir, state, results[0])

            row = {
                "train-state": train_state,
                "model": model_name,
                "learning-rate": learning_rate,
                "initialization": initialization,
                "test-state": state,
                "acc": results[0]["test_Accuracy"],
                "iou": results[0]["test_IoU"],
            }
            with open(args.output_fn, "a") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()

    main(args)
