# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NAIP + Chesapeake trainer."""

from typing import Any, Dict, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
from torchmetrics import Accuracy, IoU

from ..datasets import NAIP, BoundingBox, Chesapeake13
from ..models import FCN
from ..samplers import GridGeoSampler, RandomBatchGeoSampler

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"


class NAIPChesapeakeSegmentationTask(pl.LightningModule):
    """LightningModule for training models on the NAIP and Chesapeake datasets.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    in_channels = 4
    classes = 13
    # TODO: tune this hyperparam
    num_filters = 64

    def config_task(self, kwargs: Any) -> None:
        """Configures the task based on kwargs parameters."""
        if kwargs["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                in_channels=self.in_channels,
                classes=self.classes,
            )
        elif kwargs["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                encoder_output_stride=kwargs["encoder_output_stride"],
                in_channels=self.in_channels,
                classes=self.classes,
            )
        elif kwargs["segmentation_model"] == "fcn":
            self.model = FCN(self.in_channels, self.classes, self.num_filters)
        else:
            raise ValueError(
                f"Model type '{kwargs['segmentation_model']}' is not valid."
            )

        if kwargs["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss()  # type: ignore[attr-defined]
        elif kwargs["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass")
        else:
            raise ValueError(f"Loss type '{kwargs['loss']}' is not valid.")

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            encoder_output_stride: The output stride parameter in DeepLabV3+ models
            loss: Name of the loss function
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task(kwargs)

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.train_iou = IoU(num_classes=self.num_classes)
        self.val_iou = IoU(num_classes=self.num_classes)
        self.test_iou = IoU(num_classes=self.num_classes)

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_accuracy(y_hat_hard, y)
        self.train_iou(y_hat_hard, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics."""
        self.log("train_acc", self.train_accuracy.compute())
        self.log("train_iou", self.train_iou.compute())
        self.train_accuracy.reset()
        self.train_iou.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("val_loss", loss)
        self.val_accuracy(y_hat_hard, y)
        self.val_iou(y_hat_hard, y)

        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                batch["image"][0].cpu().numpy(), 0, 3
            )
            mask = batch["mask"][0].cpu().numpy()
            pred = y_hat_hard[0].cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img)
            axs[0].axis("off")
            axs[1].imshow(mask, vmin=0, vmax=4)
            axs[1].axis("off")
            axs[2].imshow(pred, vmin=0, vmax=4)
            axs[2].axis("off")

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )

            plt.close()

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics."""
        self.log("val_acc", self.val_accuracy.compute())
        self.log("val_iou", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_iou.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss)
        self.test_accuracy(y_hat_hard, y)
        self.test_iou(y_hat_hard, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics."""
        self.log("test_acc", self.test_accuracy.compute())
        self.log("test_iou", self.test_iou.compute())
        self.test_accuracy.reset()
        self.test_iou.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler."""
        optimizer: torch.optim.optimizer.Optimizer
        if self.hparams["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
            )
        elif self.hparams["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams["learning_rate"],
                momentum=0.9,
                weight_decay=1e-2,
            )
        else:
            raise ValueError(
                f"Optimizer choice '{self.hparams['optimizer']}' is not valid."
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
                "verbose": True,
            },
        }


class NAIPChesapeakeDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NAIP and Chesapeake datasets.

    Uses the train/val/test splits from the dataset.
    """

    # TODO: tune these hyperparams
    patch_size = 256
    length = 1000
    stride = 128

    def __init__(
        self,
        naip_root_dir: str,
        chesapeake_root_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for NAIP and Chesapeake based DataLoaders.

        Args:
            naip_root_dir: directory containing NAIP data
            chesapeake_root_dir: directory containing Chesapeake data
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.naip_root_dir = naip_root_dir
        self.chesapeake_root_dir = chesapeake_root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    # TODO: This needs to be converted to actual transforms instead of hacked
    def custom_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"] / 255.0
        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].long()

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        Chesapeake13(
            self.naip_root_dir,
            download=True,
            checksum=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        # TODO: these transforms will be applied independently, this won't work if we
        # add things like random horizontal flip
        chesapeake = Chesapeake13(
            self.chesapeake_root_dir, transforms=self.custom_transform
        )
        naip = NAIP(
            self.naip_root_dir,
            chesapeake.crs,
            chesapeake.res,
            transforms=self.custom_transform,
        )
        self.dataset = chesapeake + naip

        # TODO: figure out better train/val/test split
        roi = self.dataset.bounds
        midx = roi.minx + (roi.maxx - roi.minx) / 2
        midy = roi.miny + (roi.maxy - roi.miny) / 2
        train_roi = BoundingBox(roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt)
        val_roi = BoundingBox(midx, roi.maxx, roi.miny, midy, roi.mint, roi.maxt)
        test_roi = BoundingBox(roi.minx, roi.maxx, midy, roi.maxy, roi.mint, roi.maxt)

        self.train_sampler = RandomBatchGeoSampler(
            naip.index, self.patch_size, self.batch_size, self.length, train_roi
        )
        self.val_sampler = GridGeoSampler(
            naip.index, self.patch_size, self.stride, val_roi
        )
        self.test_sampler = GridGeoSampler(
            naip.index, self.patch_size, self.stride, test_roi
        )

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Return a DataLoader for training."""
        return DataLoader(
            self.dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Return a DataLoader for validation."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
        )
