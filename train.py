#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""torchgeo model training script."""

import os
from typing import cast

import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from torchgeo.datamodules import MisconfigurationException
from torchgeo.trainers import BYOLTask, MoCoTask, ObjectDetectionTask, SimCLRTask


def set_up_omegaconf() -> DictConfig:
    """Loads program arguments from either YAML config files or command line arguments.

    This method loads defaults/a schema from "conf/defaults.yaml" as well as potential
    arguments from the command line. If one of the command line arguments is
    "config_file", then we additionally read arguments from that YAML file. One of the
    config file based arguments or command line arguments must specify task.name. The
    task.name value is used to grab a task specific defaults from its respective
    trainer. The final configuration is given as merge(task_defaults, defaults,
    config file, command line). The merge() works from the first argument to the last,
    replacing existing values with newer values. Additionally, if any values are
    merged into task_defaults without matching types, then there will be a runtime
    error.

    Returns:
        an OmegaConf DictConfig containing all the validated program arguments

    Raises:
        FileNotFoundError: when ``config_file`` does not exist
    """
    conf = OmegaConf.load("conf/defaults.yaml")
    command_line_conf = OmegaConf.from_cli()

    if "config_file" in command_line_conf:
        config_fn = command_line_conf.config_file
        if not os.path.isfile(config_fn):
            raise FileNotFoundError(f"config_file={config_fn} is not a valid file")

        user_conf = OmegaConf.load(config_fn)
        conf = OmegaConf.merge(conf, user_conf)

    conf = OmegaConf.merge(  # Merge in any arguments passed via the command line
        conf, command_line_conf
    )
    conf = cast(DictConfig, conf)  # convince mypy that everything is alright
    return conf


def main(conf: DictConfig) -> None:
    """Main training loop."""
    if conf.program.experiment_name is not None:
        experiment_name = conf.program.experiment_name
    else:
        experiment_name = (
            f"{conf.datamodule._target_.lower()}_{conf.module._target_.lower()}"
        )
    if os.path.isfile(conf.program.output_dir):
        raise NotADirectoryError("`program.output_dir` must be a directory")
    os.makedirs(conf.program.output_dir, exist_ok=True)

    experiment_dir = os.path.join(conf.program.output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    if len(os.listdir(experiment_dir)) > 0:
        if conf.program.overwrite:
            print(
                f"WARNING! The experiment directory, {experiment_dir}, already exists, "
                + "we might overwrite data in it!"
            )
        else:
            raise FileExistsError(
                f"The experiment directory, {experiment_dir}, already exists and isn't "
                + "empty. We don't want to overwrite any existing results, exiting..."
            )

    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)

    # Define module and datamodule
    datamodule: LightningDataModule = instantiate(conf.datamodule)
    task: LightningModule = instantiate(conf.module)

    # Define callbacks
    tb_logger = TensorBoardLogger(conf.program.log_dir, name=experiment_name)
    csv_logger = CSVLogger(conf.program.log_dir, name=experiment_name)

    if isinstance(task, ObjectDetectionTask):
        monitor_metric = "val_map"
        mode = "max"
    elif isinstance(task, (BYOLTask, MoCoTask, SimCLRTask)):
        monitor_metric = "train_loss"
        mode = "min"
    else:
        monitor_metric = "val_loss"
        mode = "min"

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        filename=f"checkpoint-{{epoch:02d}}-{{{monitor_metric}:.2f}}",
        dirpath=experiment_dir,
        save_top_k=1,
        save_last=True,
        mode=mode,
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric, min_delta=0.00, patience=18, mode=mode
    )

    # Define trainer
    trainer: Trainer = instantiate(
        conf.trainer,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=[tb_logger, csv_logger],
        default_root_dir=experiment_dir,
    )

    # Train
    trainer.fit(model=task, datamodule=datamodule)

    # Test
    try:
        trainer.test(ckpt_path="best", datamodule=datamodule)
    except MisconfigurationException:
        pass


if __name__ == "__main__":
    # Taken from https://github.com/pangeo-data/cog-best-practices
    _rasterio_best_practices = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
        "GDAL_SWATH_SIZE": "200000000",
        "VSI_CURL_CACHE_SIZE": "200000000",
    }
    os.environ.update(_rasterio_best_practices)

    conf = set_up_omegaconf()

    # Set random seed for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
    pl.seed_everything(conf.program.seed)

    # Main training procedure
    main(conf)
