import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models

from torchgeo.trainers import CycloneDataModule, CycloneSimpleRegressionTask

DATA_ROOT_DIR = os.path.expanduser("~/mount/data/")
LOG_DIR_NAME = "logs/"


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser with program level arguments

    Returns:
        the argument parser
    """
    parser = argparse.ArgumentParser(description="TorchGeo model training script")

    ######################################
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

    ######################################
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
        "--root_output_dir",
        type=str,
        default="output/",
        help="Directory to store experiment results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Flag to enable overwriting existing output",
    )

    return parser


def main(args: argparse.Namespace) -> None:

    ######################################
    # Setup output directory
    ######################################

    experiment_dir = os.path.join(args.root_output_dir, args.experiment_name)
    log_dir = os.path.join(args.root_output_dir, LOG_DIR_NAME)

    if os.path.isfile(experiment_dir):
        print("A file was passed as `--root_output_dir`, please pass a directory!")
        return
    if not os.path.exists(args.root_output_dir):
        os.makedirs(args.root_output_dir)

    if os.path.exists(experiment_dir) and len(os.listdir(experiment_dir)) > 0:
        if args.overwrite:
            print(
                f"WARNING! The experiment directory, {experiment_dir}, already exists, "
                + "we might overwrite data in it!"
            )
        else:
            print(
                f"The experiment directory, {experiment_dir}, already exists and isn't "
                + "empty. We don't want to overwrite and existing results, exiting..."
            )
            return
    else:
        os.makedirs(experiment_dir, exist_ok=True)

    ######################################
    # Choose task to run based on arguments or configuration
    ######################################
    # TODO: Logic to switch between tasks

    model = models.resnet18(pretrained=False, num_classes=1)
    datamodule = CycloneDataModule(
        DATA_ROOT_DIR,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dict_args = vars(
        args
    )  # convert the argparse Namespace into a dictionary so that we can pass as kwargs
    task = CycloneSimpleRegressionTask(model, **dict_args)

    ######################################
    # Setup trainer
    ######################################
    tb_logger = pl_loggers.TensorBoardLogger(log_dir, name=args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_dir,
        monitor="val_loss",
        save_top_k=3,
        save_last=True,
    )

    trainer = pl.Trainer.from_argparse_args(
        args, logger=tb_logger, callbacks=[checkpoint_callback]
    )

    ######################################
    # Run experiment
    ######################################
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)


if __name__ == "__main__":
    # TODO: may want to eventually switch to an OmegaConf based configuration system
    # See https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    # for best practices here
    parser = set_up_parser()  # Add _program_ level arguments to the parser
    parser = pl.Trainer.add_argparse_args(
        parser
    )  # Add _trainer_ level arguments to the parser

    # TODO: Add _task_ level arguments to the parser for each _task_ we have implemented
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
    pl.seed_everything(args.seed)

    # Main training procedure
    main(args)
