import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models

from torchgeo.trainers import CycloneDataModule, CycloneSimpleRegressionTask


DATA_ROOT_DIR = os.path.expanduser("~/mount/data/")
LOG_DIR = "logs/"


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
        help="Name of this experiment in TensorBoard",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store output files",
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
    if os.path.isfile(args.output_dir):
        print("A file was passed as `--output_dir`, please pass a directory!")
        return

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        if args.overwrite:
            print(
                f"WARNING! The output directory, {args.output_dir}, already exists, "
                + "we might overwrite data in it!"
            )
        else:
            print(
                f"The output directory, {args.output_dir}, already exists and isn't "
                + "empty. We don't want to overwrite and existing results, exiting..."
            )
            return
    else:
        print("The output directory doesn't exist or is empty.")
        os.makedirs(args.output_dir, exist_ok=True)

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
    tb_logger = pl_loggers.TensorBoardLogger(LOG_DIR, name=args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, args.experiment_name),
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
