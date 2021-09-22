# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Landcover.ai trainer."""

from typing import Any, Dict, Optional, cast

import kornia.augmentation as K
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
from torchmetrics import Accuracy, IoU, MetricCollection

from ..datasets import LandCoverAI
from ..models import FCN

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"


class LandcoverAISegmentationTask(pl.LightningModule):
    """LightningModule for training models on the Landcover.AI Dataset.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hparams["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=3,
                classes=6,
            )
        elif self.hparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                encoder_output_stride=self.hparams["encoder_output_stride"],
                in_channels=3,
                classes=6,
            )
        elif self.hparams["segmentation_model"] == "fcn":
            self.model = FCN(in_channels=3, classes=6, num_filters=256)
        else:
            raise ValueError(
                f"Model type '{self.hparams['segmentation_model']}' is not valid."
            )

        if self.hparams["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss(  # type: ignore[attr-defined]
                ignore_index=0
            )
        elif self.hparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass", classes=range(1, 6))
        elif self.hparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                "multiclass", ignore_index=0, normalized=True
            )
        else:
            raise ValueError(f"Loss type '{self.hparams['loss']}' is not valid.")

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

        self.config_task()

        self.train_augmentations = K.AugmentationSequential(
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
            K.ColorJitter(p=0.5, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            data_keys=["input", "mask"],
        )

        self.train_metrics = MetricCollection(
            [
                Accuracy(num_classes=6, ignore_index=0),
                IoU(num_classes=6, ignore_index=0),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU.

        Args:
            batch: Current batch
            batch_idx: Index of current batch

        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["mask"]
        with torch.no_grad():
            x, y = self.train_augmentations(x, y)
        y = y.long().squeeze()

        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU.

        Logs the first 10 validation samples to tensorboard as images with 3 subplots
        showing the image, mask, and predictions.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"].long().squeeze()
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if batch_idx < 10:
            # Render the image, ground truth mask, and predicted mask for the first
            # image in the batch
            img = np.rollaxis(  # convert image to channels last format
                x[0].cpu().numpy(), 0, 3
            )
            mask = y[0].cpu().numpy()
            pred = y_hat_hard[0].cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img)
            axs[0].axis("off")
            axs[1].imshow(mask, vmin=0, vmax=5)
            axs[1].axis("off")
            axs[2].imshow(pred, vmin=0, vmax=5)
            axs[2].axis("off")

            # the SummaryWriter is a tensorboard object, see:
            # https://pytorch.org/docs/stable/tensorboard.html#
            summary_writer: SummaryWriter = self.logger.experiment
            summary_writer.add_figure(
                f"image/{batch_idx}", fig, global_step=self.global_step
            )

            plt.close()

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"].long().squeeze()
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["learning_rate"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
            },
        }


class LandcoverAIDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the Landcover.AI dataset.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Landcover.AI based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the Landcover.AI Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"] / 255.0

        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].float().unsqueeze(0) + 1

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        _ = LandCoverAI(
            self.root_dir,
            download=True,
            checksum=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        train_transforms = self.preprocess
        val_test_transforms = self.preprocess

        self.train_dataset = LandCoverAI(
            self.root_dir,
            split="train",
            transforms=train_transforms,
        )

        self.val_dataset = LandCoverAI(
            self.root_dir,
            split="val",
            transforms=val_test_transforms,
        )

        self.test_dataset = LandCoverAI(
            self.root_dir,
            split="test",
            transforms=val_test_transforms,
        )

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
