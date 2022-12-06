#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""torchgeo model evaluation script."""

import argparse
import csv
import os
from typing import Any, Dict, Union, cast

import pytorch_lightning as pl
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex

from torchgeo.trainers import (
    ClassificationTask,
    ObjectDetectionTask,
    SemanticSegmentationTask,
)
from train import TASK_TO_MODULES_MAPPING


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--task",
        choices=TASK_TO_MODULES_MAPPING.keys(),
        type=str,
        help="name of task to test",
    )
    parser.add_argument(
        "--input-checkpoint",
        required=True,
        help="path to the checkpoint file to test",
        metavar="CKPT",
    )
    parser.add_argument(
        "--gpu", default=0, type=int, help="GPU ID to use", metavar="ID"
    )
    parser.add_argument(
        "--root",
        required=True,
        type=str,
        help="root directory of the dataset for the accompanying task",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=2**4,
        type=int,
        help="number of samples in each mini-batch",
        metavar="SIZE",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        default=6,
        type=int,
        help="number of workers for parallel data loading",
        metavar="NUM",
    )
    parser.add_argument(
        "--seed", default=0, type=int, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--output-fn",
        required=True,
        type=str,
        help="path to the CSV file to write results",
        metavar="FILE",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print results to stdout"
    )

    return parser


def run_eval_loop(
    model: pl.LightningModule,
    dataloader: Any,
    device: torch.device,
    metrics: MetricCollection,
) -> Any:
    """Runs a standard test loop over a dataloader and records metrics.

    Args:
        model: the model used for inference
        dataloader: the dataloader to get samples from
        device: the device to put data on
        metrics: a torchmetrics compatible metric collection to score the output
            from the model

    Returns:
        the result of ``metrics.compute()``
    """
    for batch in dataloader:
        x = batch["image"].to(device)
        if "mask" in batch:
            y = batch["mask"].to(device)
        elif "label" in batch:
            y = batch["label"].to(device)
        elif "boxes" in batch:
            y = [
                {
                    "boxes": batch["boxes"][i].to(device),
                    "labels": batch["labels"][i].to(device),
                }
                for i in range(len(batch["image"]))
            ]
        with torch.inference_mode():
            y_pred = model(x)
        metrics(y_pred, y)
    results = metrics.compute()
    metrics.reset()
    return results


def main(args: argparse.Namespace) -> None:
    """High-level pipeline.

    Runs a model checkpoint on a test set and saves results to file.

    Args:
        args: command-line arguments
    """
    assert os.path.exists(args.input_checkpoint)
    assert os.path.exists(args.root)
    TASK = TASK_TO_MODULES_MAPPING[args.task][0]
    DATAMODULE = TASK_TO_MODULES_MAPPING[args.task][1]

    # Loads the saved model from checkpoint based on the `args.task` name that was
    # passed as input
    model = TASK.load_from_checkpoint(args.input_checkpoint)
    model = cast(pl.LightningModule, model)
    model.freeze()
    model.eval()

    dm = DATAMODULE(  # type: ignore[call-arg]
        seed=args.seed,
        root=args.root,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    dm.setup("validate")

    # Record model hyperparameters
    hparams = cast(Dict[str, Union[str, float]], model.hparams)
    if issubclass(TASK, ClassificationTask):
        val_row = {
            "split": "val",
            "model": hparams["model"],
            "learning_rate": hparams["learning_rate"],
            "weights": hparams["weights"],
            "loss": hparams["loss"],
        }

        test_row = {
            "split": "test",
            "model": hparams["model"],
            "learning_rate": hparams["learning_rate"],
            "weights": hparams["weights"],
            "loss": hparams["loss"],
        }
    elif issubclass(TASK, SemanticSegmentationTask):
        val_row = {
            "split": "val",
            "segmentation_model": hparams["segmentation_model"],
            "encoder_name": hparams["encoder_name"],
            "encoder_weights": hparams["encoder_weights"],
            "learning_rate": hparams["learning_rate"],
            "loss": hparams["loss"],
        }

        test_row = {
            "split": "test",
            "segmentation_model": hparams["segmentation_model"],
            "encoder_name": hparams["encoder_name"],
            "encoder_weights": hparams["encoder_weights"],
            "learning_rate": hparams["learning_rate"],
            "loss": hparams["loss"],
        }
    elif issubclass(TASK, ObjectDetectionTask):
        val_row = {
            "split": "val",
            "detection_model": hparams["detection_model"],
            "backbone": hparams["backbone"],
            "learning_rate": hparams["learning_rate"],
        }

        test_row = {
            "split": "test",
            "detection_model": hparams["detection_model"],
            "backbone": hparams["backbone"],
            "learning_rate": hparams["learning_rate"],
        }
    else:
        raise ValueError(f"{TASK} is not supported")

    # Compute metrics
    device = torch.device("cuda:%d" % (args.gpu))
    model = model.to(device)

    if args.task == "etci2021":  # Custom metric setup for testing ETCI2021

        metrics = MetricCollection([BinaryAccuracy(), BinaryJaccardIndex()]).to(device)

        val_results = run_eval_loop(model, dm.val_dataloader(), device, metrics)
        test_results = run_eval_loop(model, dm.test_dataloader(), device, metrics)

        val_row.update(
            {
                "overall_accuracy": val_results["Accuracy"].item(),
                "jaccard_index": val_results["JaccardIndex"][1].item(),
            }
        )
        test_row.update(
            {
                "overall_accuracy": test_results["Accuracy"].item(),
                "jaccard_index": test_results["JaccardIndex"][1].item(),
            }
        )
    else:  # Test with PyTorch Lightning as usual
        model.val_metrics = cast(MetricCollection, model.val_metrics)
        model.test_metrics = cast(MetricCollection, model.test_metrics)

        val_results = run_eval_loop(
            model, dm.val_dataloader(), device, model.val_metrics
        )
        test_results = run_eval_loop(
            model, dm.test_dataloader(), device, model.test_metrics
        )

        # Save the results and model hyperparameters to a CSV file
        if issubclass(TASK, ClassificationTask):
            val_row.update(
                {
                    "average_accuracy": val_results["val_AverageAccuracy"].item(),
                    "overall_accuracy": val_results["val_OverallAccuracy"].item(),
                }
            )
            test_row.update(
                {
                    "average_accuracy": test_results["test_AverageAccuracy"].item(),
                    "overall_accuracy": test_results["test_OverallAccuracy"].item(),
                }
            )
        elif issubclass(TASK, SemanticSegmentationTask):
            val_row.update(
                {
                    "overall_accuracy": val_results["val_Accuracy"].item(),
                    "jaccard_index": val_results["val_JaccardIndex"].item(),
                }
            )
            test_row.update(
                {
                    "overall_accuracy": test_results["test_Accuracy"].item(),
                    "jaccard_index": test_results["test_JaccardIndex"].item(),
                }
            )
        elif issubclass(TASK, ObjectDetectionTask):
            val_row.update({"map": val_results["map"].item()})
            test_row.update({"map": test_results["map"].item()})

    assert set(val_row.keys()) == set(test_row.keys())
    fieldnames = list(test_row.keys())

    # Write to file
    if not os.path.exists(args.output_fn):
        with open(args.output_fn, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    with open(args.output_fn, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(val_row)
        writer.writerow(test_row)


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    main(args)
