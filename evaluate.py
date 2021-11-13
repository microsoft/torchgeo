#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""torchgeo model evaluation script."""

import argparse
import csv
import os
from typing import Dict, Tuple, Type, Union

import pytorch_lightning as pl

from torchgeo.datasets import (
    BigEarthNetDataModule,
    ChesapeakeCVPRDataModule,
    COWCCountingDataModule,
    CycloneDataModule,
    ETCI2021DataModule,
    LandCoverAIDataModule,
    NAIPChesapeakeDataModule,
    RESISC45DataModule,
    SEN12MSDataModule,
    So2SatDataModule,
    UCMercedDataModule,
)
from torchgeo.trainers import (
    BYOLTask,
    ClassificationTask,
    MultiLabelClassificationTask,
    RegressionTask,
    SemanticSegmentationTask,
)
from torchgeo.trainers.chesapeake import ChesapeakeCVPRSegmentationTask
from torchgeo.trainers.landcoverai import LandCoverAISegmentationTask
from torchgeo.trainers.naipchesapeake import NAIPChesapeakeSegmentationTask
from torchgeo.trainers.resisc45 import RESISC45ClassificationTask
from torchgeo.trainers.so2sat import So2SatClassificationTask

TASK_TO_MODULES_MAPPING: Dict[
    str, Tuple[Type[pl.LightningModule], Type[pl.LightningDataModule]]
] = {
    "bigearthnet": (MultiLabelClassificationTask, BigEarthNetDataModule),
    "byol": (BYOLTask, ChesapeakeCVPRDataModule),
    "chesapeake_cvpr": (ChesapeakeCVPRSegmentationTask, ChesapeakeCVPRDataModule),
    "cowc_counting": (RegressionTask, COWCCountingDataModule),
    "cyclone": (RegressionTask, CycloneDataModule),
    "etci2021": (SemanticSegmentationTask, ETCI2021DataModule),
    "landcoverai": (LandCoverAISegmentationTask, LandCoverAIDataModule),
    "naipchesapeake": (NAIPChesapeakeSegmentationTask, NAIPChesapeakeDataModule),
    "resisc45": (RESISC45ClassificationTask, RESISC45DataModule),
    "sen12ms": (SemanticSegmentationTask, SEN12MSDataModule),
    "so2sat": (So2SatClassificationTask, So2SatDataModule),
    "ucmerced": (ClassificationTask, UCMercedDataModule),
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
        "--root-dir",
        required=True,
        type=str,
        help="root directory of the dataset for the accompanying task",
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


def main(args: argparse.Namespace) -> None:
    """High-level pipeline.

    Runs a model checkpoint on a test set and saves results to file.

    Args:
        args: command-line arguments
    """
    assert os.path.exists(args.input_checkpoint)
    assert os.path.exists(args.root_dir)
    TASK = TASK_TO_MODULES_MAPPING[args.task][0]
    DATAMODULE = TASK_TO_MODULES_MAPPING[args.task][1]

    trainer = pl.Trainer(
        gpus=[args.gpu],
        logger=False,
        progress_bar_refresh_rate=0,
        checkpoint_callback=False,
    )

    # Loads the saved model from checkpoint based on the `args.task` name that was
    # passed as input
    model = TASK.load_from_checkpoint(args.input_checkpoint)
    model.freeze()
    model.eval()

    dm = DATAMODULE(  # type: ignore[call-arg]
        seed=args.seed,
        root_dir=args.root_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    dm.setup()

    # Run the model checkpoint on the validation set and the test set and save the
    # results.
    # NOTE: we might want to manually run these loops so that we can record different
    # metrics than those in the different generic tasks
    val_results = trainer.validate(
        model=model, dataloaders=dm.val_dataloader(), verbose=False
    )[0]
    test_results = trainer.test(model=model, datamodule=dm, verbose=False)[0]

    # Save the results and model hyperparameters to a CSV file
    if issubclass(TASK, ClassificationTask):
        val_row: Dict[str, Union[str, float]] = {
            "split": "val",
            "classification_model": model.hparams["classification_model"],
            "learning_rate": model.hparams["learning_rate"],
            "weights": model.hparams["weights"],
            "loss": model.hparams["loss"],
            "average_accuracy": val_results["val_AverageAccuracy"],
            "overall_accuracy": val_results["val_OverallAccuracy"],
        }

        test_row: Dict[str, Union[str, float]] = {
            "split": "test",
            "classification_model": model.hparams["classification_model"],
            "learning_rate": model.hparams["learning_rate"],
            "weights": model.hparams["weights"],
            "loss": model.hparams["loss"],
            "average_accuracy": test_results["test_AverageAccuracy"],
            "overall_accuracy": test_results["test_OverallAccuracy"],
        }
        assert set(val_row.keys()) == set(test_row.keys())

        fieldnames = list(test_results.keys())
    elif issubclass(TASK, SemanticSegmentationTask):
        val_row: Dict[str, Union[str, float]] = {  # type: ignore[no-redef]
            "split": "val",
            "segmentation_model": model.hparams["segmentation_model"],
            "encoder_name": model.hparams["encoder_name"],
            "encoder_weights": model.hparams["encoder_weights"],
            "learning_rate": model.hparams["learning_rate"],
            "loss": model.hparams["loss"],
            "overall_accuracy": val_results["val_Accuracy"],
            "iou": val_results["val_IoU"],
        }

        test_row: Dict[str, Union[str, float]] = {  # type: ignore[no-redef]
            "split": "test",
            "segmentation_model": model.hparams["segmentation_model"],
            "encoder_name": model.hparams["encoder_name"],
            "encoder_weights": model.hparams["encoder_weights"],
            "learning_rate": model.hparams["learning_rate"],
            "loss": model.hparams["loss"],
            "overall_accuracy": test_results["test_Accuracy"],
            "iou": test_results["test_IoU"],
        }
        assert set(val_row.keys()) == set(test_row.keys())

        fieldnames = list(test_row.keys())
    else:
        raise ValueError(f"{TASK} is not supported")

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
