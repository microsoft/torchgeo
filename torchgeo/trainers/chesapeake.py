# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for the Chesapeake datasets."""

from typing import Any, Dict, Optional, cast, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore[attr-defined]
from torchmetrics import Accuracy, IoU
from torchvision.transforms import Compose

from ..datasets import Chesapeake7, ChesapeakeCVPR
from ..samplers import RandomBatchGeoSampler, GridGeoSampler

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"

CMAP = matplotlib.colors.ListedColormap(
    [np.array(Chesapeake7.cmap[i + 1]) / 255.0 for i in range(6)]
)


class ChesapeakeCVPRSegmentationTask(LightningModule):
    """LightningModule for training models on the Chesapeake CVPR Land Cover dataset.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hparams["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=4,
                classes=7,
            )
        elif self.hparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=4,
                classes=7,
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['segmentation_model']}' is not valid."
            )

        if self.hparams["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss(  # type: ignore[attr-defined]
                ignore_index=7
            )
        elif self.hparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(mode="multiclass")
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
            loss: Name of the loss function

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.train_iou = IoU(num_classes=7)
        self.val_iou = IoU(num_classes=7)
        self.test_iou = IoU(num_classes=7)

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
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
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log("train_acc", self.train_accuracy.compute())
        self.log("train_iou", self.train_iou.compute())
        self.train_accuracy.reset()
        self.train_iou.reset()

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
            axs[0].imshow(img[:, :, :3])
            axs[0].axis("off")
            axs[1].imshow(mask, vmin=0, vmax=6, cmap=CMAP, interpolation="none")
            axs[1].axis("off")
            axs[2].imshow(pred, vmin=0, vmax=6, cmap=CMAP, interpolation="none")
            axs[2].axis("off")
            plt.tight_layout()

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
        self.log("val_acc", self.val_accuracy.compute())
        self.log("val_iou", self.val_iou.compute())
        self.val_accuracy.reset()
        self.val_iou.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
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
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        self.log("test_acc", self.test_accuracy.compute())
        self.log("test_iou", self.test_iou.compute())
        self.test_accuracy.reset()
        self.test_iou.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation
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
                "verbose": True,
            },
        }


class ChesapeakeCVPRDataModule(LightningDataModule):
    """LightningDataModule implementation for the Chesapeake CVPR Land Cover dataset.

    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """

    def __init__(
        self,
        root_dir: str,
        train_state: str,
        patches_per_tile: int = 200,
        patch_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Chesapeake CVPR based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the ChesapeakeCVPR Dataset
                classes
            train_state: The state code to use to train the model, e.g. "ny"
            patches_per_tile: The number of patches per tile to sample
            patch_size: The size of each patch in pixels (test patches will be 1.5 times
                this size)
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        assert train_state in ["md", "de", "ny", "pa", "va", "wv"]

        self.root_dir = root_dir
        self.train_state = train_state
        self.layers = ["naip-new", "lc"]
        self.patches_per_tile = patches_per_tile
        self.patch_size = patch_size
        # This is a rough estimate of how large of a patch we will need to sample in
        # EPSG:3857 in order to garuntee a large enough patch in the local CRS.
        self.original_patch_size = int(patch_size * 1.5)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def pad_to(
        self, size: int = 512, image_value: int = 0, mask_value: int = 0
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a padding transform on a single sample."""

        def pad_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape
            assert height <= size and width <= size

            height_pad = size - height
            width_pad = size - width

            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            # for a description of the format of the padding tuple
            sample["image"] = F.pad(
                sample["image"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=image_value,
            )
            sample["mask"] = F.pad(
                sample["mask"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=mask_value,
            )
            return sample
        return pad_inner

    def center_crop(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a center crop transform on a single sample."""

        def center_crop_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape
            assert height >= size and width >= size

            _, height, width = sample["image"].shape

            y1 = (height - size) // 2
            x1 = (width - size) // 2
            sample["image"] = sample["image"][:, y1 : y1 + size, x1 : x1 + size]
            sample["mask"] = sample["mask"][:, y1 : y1 + size, x1 : x1 + size]

            return sample
        return center_crop_inner

    def preprocess(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Preprocesses a single sample."""
        sample["image"] = sample["image"] / 255.0
        sample["mask"] = sample["mask"] - 1

        sample["image"] = sample["image"].float()
        sample["mask"] = sample["mask"].long()

        return sample

    def prepare_data(self) -> None:
        """Confirms that the dataset is downloaded on the local node.

        This method is called once per node, while :func:`setup` is called once per GPU.
        """
        ChesapeakeCVPR(
            self.root_dir,
            split=f"{self.train_state}-train",
            layers=self.layers,
            transforms=None,
            download=True,
            checksum=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.
        """
        train_transforms = Compose([
            self.center_crop(self.patch_size),
            self.preprocess
        ])
        val_transforms = Compose([
            self.center_crop(self.patch_size),
            self.preprocess
        ])
        test_transforms = Compose([
            self.pad_to(self.original_patch_size, image_value=0, mask_value=7),
            self.preprocess
        ])

        self.train_dataset = ChesapeakeCVPR(
            self.root_dir,
            split=f"{self.train_state}-train",
            layers=self.layers,
            transforms=train_transforms,
            download=False,
            checksum=False,
        )
        self.val_dataset = ChesapeakeCVPR(
            self.root_dir,
            split=f"{self.train_state}-val",
            layers=self.layers,
            transforms=val_transforms,
            download=False,
            checksum=False,
        )
        self.test_dataset = ChesapeakeCVPR(
            self.root_dir,
            split=f"{self.train_state}-test",
            layers=self.layers,
            transforms=test_transforms,
            download=False,
            checksum=False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        sampler = RandomBatchGeoSampler(
            self.train_dataset.index,
            size=self.original_patch_size,
            batch_size=self.batch_size,
            length=self.patches_per_tile * self.train_dataset.index.get_size(),
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,  # type: ignore[arg-type]
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        sampler = RandomBatchGeoSampler(
            self.val_dataset.index,
            size=self.original_patch_size,
            batch_size=self.batch_size,
            length=self.patches_per_tile * self.val_dataset.index.get_size(),
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,  # type: ignore[arg-type]
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        sampler = GridGeoSampler(
            self.test_dataset.index,
            size=self.original_patch_size,
            stride=self.original_patch_size,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=sampler,   # type: ignore[arg-type]
            num_workers=self.num_workers
        )
