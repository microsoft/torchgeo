#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""torchgeo model inference script."""

import argparse
import os
from typing import Dict, Tuple, Type, cast

import pytorch_lightning as pl
import rasterio as rio
import torch
from kornia.contrib import CombineTensorPatches
from omegaconf import OmegaConf

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


def write_mask(mask: torch.Tensor, output_dir: str, input_filename: str) -> None:
    """Write mask to specified output directory."""
    output_path = os.path.join(output_dir, os.path.basename(input_filename))
    with rio.open(input_filename) as src:
        profile = src.profile
    profile["count"] = 1
    profile["dtype"] = "uint8"
    mask = mask.cpu().numpy()
    with rio.open(output_path, "w", **profile) as ds:
        ds.write(mask)


def main(config_dir: str, predict_on: str, output_dir: str, device: str) -> None:
    """Main inference loop."""
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint and config
    conf = OmegaConf.load(os.path.join(config_dir, "experiment_config.yaml"))
    ckpt = os.path.join(config_dir, "last.ckpt")

    # Load model
    task_name = conf.experiment.task
    datamodule: pl.LightningDataModule
    task: pl.LightningModule
    if task_name not in TASK_TO_MODULES_MAPPING:
        raise ValueError(
            f"experiment.task={task_name} is not recognized as a valid task"
        )
    task_class, datamodule_class = TASK_TO_MODULES_MAPPING[task_name]
    task = task_class.load_from_checkpoint(ckpt)
    task = task.to(device)
    task.eval()

    # Load datamodule and dataloader
    conf.experiment.datamodule["predict_on"] = predict_on
    datamodule = datamodule_class(**conf.experiment.datamodule)
    datamodule.setup()
    dataloader = datamodule.predict_dataloader()

    if len(os.listdir(output_dir)) > 0:
        if conf.program.overwrite:
            print(
                f"WARNING! The output directory, {output_dir}, already exists, "
                + "we will overwrite data in it!"
            )
        else:
            raise FileExistsError(
                f"The predictions directory, {output_dir}, already exists and isn't "
                + "empty. We don't want to overwrite any existing results, exiting..."
            )

    for i, batch in enumerate(dataloader):
        x = batch["image"].to(device)  # (N, B, C, H, W)
        assert len(x.shape) in {4, 5}
        if len(x.shape) == 5:
            masks = []

            def tensor_to_int(
                tensor_tuple: Tuple[torch.Tensor, ...]
            ) -> Tuple[int, ...]:
                """Convert tuple of tensors to tuple of ints."""
                return tuple(int(i.item()) for i in tensor_tuple)

            original_shape = cast(
                Tuple[int, int], tensor_to_int(batch["original_shape"])
            )
            patch_shape = cast(Tuple[int, int], tensor_to_int(batch["patch_shape"]))
            padding = cast(Tuple[int, int], tensor_to_int(batch["padding"]))
            patch_combine = CombineTensorPatches(
                original_size=original_shape, window_size=patch_shape, unpadding=padding
            )
            for tile in x:
                mask = task(tile)
                mask = mask.argmax(dim=1)
                masks.append(mask)

            masks_arr = torch.stack(masks, dim=0)
            masks_arr = masks_arr.unsqueeze(0)
            masks_combined = patch_combine(masks_arr)[0]
            filename = datamodule.predict_dataset.files[i]["image"]
            write_mask(masks_combined, output_dir, filename)
        else:
            mask = task(x)
            mask = mask.argmax(dim=1)
            filename = datamodule.predict_dataset.files[i]["image"]
            write_mask(mask, output_dir, filename)


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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Path to config-dir to load config and ckpt",
    )

    parser.add_argument(
        "--predict_on",
        type=str,
        required=True,
        help="Directory/Dataset to run inference on",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output_directory to save predicted mask geotiffs",
    )

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    main(args.config_dir, args.predict_on, args.output_dir, args.device)
