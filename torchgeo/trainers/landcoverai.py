"""Landcover.ai trainer."""

from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
from torchmetrics import Accuracy, IoU  # type: ignore[attr-defined]

from ..datasets import LandCoverAI

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"


class LandcoverAISegmentationTask(pl.LightningModule):
    """LightningModule for training models on the Landcover.AI Dataset.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    def __init__(
        self,
        model: Module,
        loss: Module = nn.CrossEntropyLoss(),  # type: ignore[attr-defined]
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Args:
            model: A model (specifically, a ``nn.Module``) instance to be trained.
            loss: A semantic segmentation loss function to use (e.g. pixel-wise
                crossentropy)
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs
        self.model = model
        self.loss = loss

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.train_iou = IoU(num_classes=5)
        self.val_iou = IoU(num_classes=5)
        self.test_iou = IoU(num_classes=5)

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

    # TODO: This needs to be converted to actual transforms instead of hacked
    def custom_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"] / 255.0
        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].long()

        return sample

    def prepare_data(self) -> None:
        """Initialize the main ``Dataset`` objects."""
        self.train_dataset = LandCoverAI(
            self.root_dir,
            split="train",
            transforms=self.custom_transform,
        )

        self.val_dataset = LandCoverAI(
            self.root_dir,
            split="val",
            transforms=self.custom_transform,
        )

        self.test_dataset = LandCoverAI(
            self.root_dir,
            split="test",
            transforms=self.custom_transform,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
