#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""script for testing saved models on different ChesapeakeCVPR dataset splits."""

import argparse
import csv
import os

import pytorch_lightning as pl
import torch

from torchgeo.trainers import ChesapeakeCVPRDataModule, ChesapeakeCVPRSegmentationTask

ALL_TEST_SPLITS = [["de-val"], ["pa-test"], ["ny-test"], ["pa-test", "ny-test"]]


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
    parser.add_argument(
        "-d",
        "--device",
        default=0,
        type=int,
        help="GPU ID to use, ignored if no GPUs are available",
        metavar="ID",
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
        "loss",
        "test-state",
        "acc",
        "iou",
    ]
    with open(args.output_fn, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # Test loop
    trainer = pl.Trainer(
        gpus=[args.device] if torch.cuda.is_available() else None,
        logger=False,
        progress_bar_refresh_rate=0,
        checkpoint_callback=False,
    )

    for experiment_dir in os.listdir(args.input_dir):

        checkpoint_fn = None
        for fn in os.listdir(os.path.join(args.input_dir, experiment_dir)):
            if fn.startswith("epoch") and fn.endswith(".ckpt"):
                checkpoint_fn = fn
                break
        if checkpoint_fn is None:
            print(
                f"Skipping {os.path.join(args.input_dir, experiment_dir)} as we are not"
                + " able to find a checkpoint file"
            )
            continue
        checkpoint_fn = os.path.join(args.input_dir, experiment_dir, checkpoint_fn)

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
            loss = experiment_dir_parts[3]
            initialization = "random" if len(experiment_dir_parts) == 5 else "imagenet"
        except IndexError:
            print(
                f"Skipping {experiment_dir} as the directory name is not in the"
                + " expected format"
            )
            continue

        # Test the loaded model against the test set from all states
        for test_splits in ALL_TEST_SPLITS:

            dm = ChesapeakeCVPRDataModule(
                args.chesapeakecvpr_root,
                train_splits=["de-train"],
                val_splits=["de-val"],
                test_splits=test_splits,
                batch_size=32,
                num_workers=8,
                class_set=5,
            )
            results = trainer.test(model=model, datamodule=dm, verbose=False)
            print(experiment_dir, test_splits, results[0])

            row = {
                "train-state": train_state,
                "model": model_name,
                "learning-rate": learning_rate,
                "initialization": initialization,
                "loss": loss,
                "test-state": "_".join(test_splits),
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
