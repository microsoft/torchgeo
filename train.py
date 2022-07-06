#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""torchgeo model training script."""

import os
from typing import Any, Dict, Tuple, Type, cast

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torchgeo.datamodules import (
    BigEarthNetDataModule,
    ChesapeakeCVPRDataModule,
    COWCCountingDataModule,
    CycloneDataModule,
    ETCI2021DataModule,
    EuroSATDataModule,
    InriaAerialImageLabelingDataModule,
    LandCoverAIDataModule,
    NAIPChesapeakeDataModule,
    OSCDDataModule,
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

TASK_TO_MODULES_MAPPING: Dict[
    str, Tuple[Type[pl.LightningModule], Type[pl.LightningDataModule]]
] = {
    "bigearthnet": (MultiLabelClassificationTask, BigEarthNetDataModule),
    "byol": (BYOLTask, ChesapeakeCVPRDataModule),
    "chesapeake_cvpr": (SemanticSegmentationTask, ChesapeakeCVPRDataModule),
    "cowc_counting": (RegressionTask, COWCCountingDataModule),
    "cyclone": (RegressionTask, CycloneDataModule),
    "eurosat": (ClassificationTask, EuroSATDataModule),
    "etci2021": (SemanticSegmentationTask, ETCI2021DataModule),
    "inria": (SemanticSegmentationTask, InriaAerialImageLabelingDataModule),
    "landcoverai": (SemanticSegmentationTask, LandCoverAIDataModule),
    "naipchesapeake": (SemanticSegmentationTask, NAIPChesapeakeDataModule),
    "oscd": (SemanticSegmentationTask, OSCDDataModule),
    "resisc45": (ClassificationTask, RESISC45DataModule),
    "sen12ms": (SemanticSegmentationTask, SEN12MSDataModule),
    "so2sat": (ClassificationTask, So2SatDataModule),
    "ucmerced": (ClassificationTask, UCMercedDataModule),
}


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
        if not os.path.isfile(config_fn):
            raise FileNotFoundError(f"config_file={config_fn} is not a valid file")

        user_conf = OmegaConf.load(config_fn)
        conf = OmegaConf.merge(conf, user_conf)

    conf = OmegaConf.merge(  # Merge in any arguments passed via the command line
        conf, command_line_conf
    )

    # These OmegaConf structured configs enforce a schema at runtime, see:
    # https://omegaconf.readthedocs.io/en/2.0_branch/structured_config.html#merging-with-other-configs
    task_name = conf.experiment.task
    task_config_fn = os.path.join("conf", f"{task_name}.yaml")
    if task_name == "test":
        task_conf = OmegaConf.create()
    elif os.path.exists(task_config_fn):
        task_conf = cast(DictConfig, OmegaConf.load(task_config_fn))
    else:
        raise ValueError(
            f"experiment.task={task_name} is not recognized as a valid task"
        )

    conf = OmegaConf.merge(task_conf, conf)
    conf = cast(DictConfig, conf)  # convince mypy that everything is alright

    return conf


def main(conf: DictConfig) -> None:
    """Main training loop."""
    ######################################
    # Setup output directory
    ######################################

    experiment_name = conf.experiment.name
    task_name = conf.experiment.task
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

    with open(os.path.join(experiment_dir, "experiment_config.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)

    ######################################
    # Choose task to run based on arguments or configuration
    ######################################
    # Convert the DictConfig into a dictionary so that we can pass as kwargs.
    task_args = cast(Dict[str, Any], OmegaConf.to_object(conf.experiment.module))
    datamodule_args = cast(
        Dict[str, Any], OmegaConf.to_object(conf.experiment.datamodule)
    )

    datamodule: pl.LightningDataModule
    task: pl.LightningModule
    if task_name in TASK_TO_MODULES_MAPPING:
        task_class, datamodule_class = TASK_TO_MODULES_MAPPING[task_name]
        task = task_class(**task_args)
        datamodule = datamodule_class(**datamodule_args)
    else:
        raise ValueError(
            f"experiment.task={task_name} is not recognized as a valid task"
        )

    ######################################
    # Setup trainer
    ######################################
    tb_logger = pl_loggers.TensorBoardLogger(conf.program.log_dir, name=experiment_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath=experiment_dir, save_top_k=1, save_last=True
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=18
    )

    trainer_args = cast(Dict[str, Any], OmegaConf.to_object(conf.trainer))

    trainer_args["callbacks"] = [checkpoint_callback, early_stopping_callback]
    trainer_args["logger"] = tb_logger
    trainer_args["default_root_dir"] = experiment_dir
    trainer = pl.Trainer(**trainer_args)

    if trainer_args.get("auto_lr_find"):
        trainer.tune(model=task, datamodule=datamodule)

    ######################################
    # Run experiment
    ######################################
    trainer.fit(model=task, datamodule=datamodule)
    trainer.test(model=task, datamodule=datamodule)


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
