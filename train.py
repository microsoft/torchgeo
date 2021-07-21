#!/usr/bin/env python3

"""torchgeo model training script."""

import argparse
import os
from typing import Optional

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from torchvision import models

from torchgeo.trainers import (
    CycloneDataModule,
    CycloneSimpleRegressionTask,
    SEN12MSDataModule,
    SEN12MSSegmentationTask,
)


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser with program level arguments.

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ###########################
    # Add _program_ level arguments to the parser
    ###########################
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to use in training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use in the Dataloaders",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random number generator seed for numpy and torch",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of this experiment (used in TensorBoard and as the subdirectory "
        + "name to save results)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to store experiment results",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory where datasets are/will be stored",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory where logs will be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Flag to enable overwriting existing output",
    )

    # TODO: may want to eventually switch to an OmegaConf based configuration system
    # See https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    # for best practices here

    ###########################
    # Add _trainer_ level arguments to the parser
    ###########################
    parser = pl.Trainer.add_argparse_args(parser)

    ###########################
    # TODO: Add _task_ level arguments to the parser for each _task_ we have implemented
    ###########################
    parser.add_argument(
        "--task",
        choices=["cyclone", "sen12ms"],
        type=str,
        default="cyclone",
        help="Task to perform",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--learning_rate_schedule_patience",
        type=int,
        default=2,
        help="Patience factor for the ReduceLROnPlateau schedule",
    )

    return parser


def main(args: argparse.Namespace) -> None:
    """Main training loop."""
    ######################################
    # Setup output directory
    ######################################

    if os.path.isfile(args.output_dir):
        raise NotADirectoryError("`--output_dir` must be a directory")
    os.makedirs(args.output_dir, exist_ok=True)

    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    if len(os.listdir(experiment_dir)) > 0:
        if args.overwrite:
            # TODO: convert this to logging.WARNING
            print(
                f"WARNING! The experiment directory, {experiment_dir}, already exists, "
                + "we might overwrite data in it!"
            )
        else:
            raise FileExistsError(
                f"The experiment directory, {experiment_dir}, already exists and isn't "
                + "empty. We don't want to overwrite any existing results, exiting..."
            )

    ######################################
    # Choose task to run based on arguments or configuration
    ######################################
    # Convert the argparse Namespace into a dictionary so that we can pass as kwargs
    dict_args = vars(args)

    datamodule: Optional[LightningDataModule] = None
    task: Optional[LightningModule] = None
    if args.task == "cyclone":
        datamodule = CycloneDataModule(
            args.data_dir,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = models.resnet18(pretrained=False, num_classes=1)
        task = CycloneSimpleRegressionTask(model, **dict_args)
    elif args.task == "sen12ms":
        import segmentation_models_pytorch as smp

        datamodule = SEN12MSDataModule(
            args.data_dir,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=15,
            classes=11,
        )
        loss = nn.CrossEntropyLoss()  # type: ignore[attr-defined]
        task = SEN12MSSegmentationTask(model, loss, **dict_args)

    ######################################
    # Setup trainer
    ######################################
    tb_logger = pl_loggers.TensorBoardLogger(args.log_dir, name=args.experiment_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=experiment_dir,
        save_top_k=3,
        save_last=True,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=[checkpoint_callback, early_stopping_callback]
    )

    ######################################
    # Run experiment
    ######################################
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)


if __name__ == "__main__":
    parser = set_up_parser()
    args = parser.parse_args()

    # Set random seed for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
    pl.seed_everything(args.seed)

    # Main training procedure
    main(args)
