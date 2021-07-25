#!/usr/bin/env python3

"""torchgeo model training script."""

import os
from typing import Any, Dict, cast

import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
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
        ValueError: when ``task.name`` is not a valid task
    """
    conf = OmegaConf.load("conf/defaults.yaml")
    command_line_conf = OmegaConf.from_cli()

    if "config_file" in command_line_conf:
        config_fn = command_line_conf.config_file
        if os.path.isfile(config_fn):
            user_conf = OmegaConf.load(config_fn)
            conf = OmegaConf.merge(conf, user_conf)
        else:
            raise FileNotFoundError(f"config_file={config_fn} is not a valid file")

    conf = OmegaConf.merge(  # Merge in any arguments passed via the command line
        conf, command_line_conf
    )

    # These OmegaConf structured configs enforce a schema at runtime, see:
    # https://omegaconf.readthedocs.io/en/2.0_branch/structured_config.html#merging-with-other-configs
    if conf.task.name == "cyclone":
        task_conf = OmegaConf.load("conf/task_defaults/cyclone.yaml")
    elif conf.task.name == "sen12ms":
        task_conf = OmegaConf.load("conf/task_defaults/sen12ms.yaml")
    elif conf.task.name == "test":
        task_conf = OmegaConf.create()
    else:
        raise ValueError(
            f"task.name={conf.task.name} is not recognized as a valid task"
        )

    conf = OmegaConf.merge(task_conf, conf)
    conf = cast(DictConfig, conf)  # convince mypy that everything is alright

    return conf


def main(conf: DictConfig) -> None:
    """Main training loop."""
    ######################################
    # Setup output directory
    ######################################

    if os.path.isfile(conf.program.output_dir):
        raise NotADirectoryError("`program.output_dir` must be a directory")
    os.makedirs(conf.program.output_dir, exist_ok=True)

    experiment_dir = os.path.join(conf.program.output_dir, conf.program.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    if len(os.listdir(experiment_dir)) > 0:
        if conf.program.overwrite:
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
    # Convert the DictConfig into a dictionary so that we can pass as kwargs. We use
    # var() to convert the @dataclass from to_object() to a dictionary and to help mypy
    task_args = OmegaConf.to_object(conf.task)
    task_args = cast(Dict[str, Any], task_args)

    datamodule: LightningDataModule
    task: LightningModule
    if conf.task.name == "cyclone":
        datamodule = CycloneDataModule(
            conf.program.data_dir,
            seed=conf.program.seed,
            batch_size=conf.program.batch_size,
            num_workers=conf.program.num_workers,
        )
        model = models.resnet18(pretrained=False, num_classes=1)
        task = CycloneSimpleRegressionTask(model, **task_args)
    elif conf.task.name == "sen12ms":
        import segmentation_models_pytorch as smp

        datamodule = SEN12MSDataModule(
            conf.program.data_dir,
            seed=conf.program.seed,
            batch_size=conf.program.batch_size,
            num_workers=conf.program.num_workers,
        )
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=15,
            classes=11,
        )
        loss = nn.CrossEntropyLoss()  # type: ignore[attr-defined]
        task = SEN12MSSegmentationTask(model, loss, **task_args)
    else:
        raise ValueError(
            f"task.name={conf.task.name} is not recognized as a valid task"
        )

    ######################################
    # Setup trainer
    ######################################
    tb_logger = pl_loggers.TensorBoardLogger(
        conf.program.log_dir, name=conf.program.experiment_name
    )

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

    trainer_args = OmegaConf.to_object(conf.trainer)
    trainer_args = cast(Dict[str, Any], trainer_args)

    trainer_args["callbacks"] = [checkpoint_callback, early_stopping_callback]
    trainer_args["logger"] = tb_logger
    trainer = pl.Trainer(**trainer_args)

    ######################################
    # Run experiment
    ######################################
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)


if __name__ == "__main__":
    conf = set_up_omegaconf()

    # Set random seed for reproducibility
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.utilities.seed.html#pytorch_lightning.utilities.seed.seed_everything
    pl.seed_everything(conf.program.seed)

    # Main training procedure
    main(conf)
