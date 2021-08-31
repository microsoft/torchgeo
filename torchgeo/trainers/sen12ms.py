# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SEN12MS trainer."""

from typing import Any, Dict, Optional, cast

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy

from ..datasets import SEN12MS

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"


class SEN12MSSegmentationTask(pl.LightningModule):
    """LightningModule for training models on the SEN12MS Dataset.

    This allows using arbitrary models and losses from the
    ``pytorch_segmentation_models`` package.
    """

    def config_task(self, kwargs: Dict[str, Any]) -> None:
        """Configures the task based on kwargs parameters."""
        if kwargs["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=kwargs["encoder_name"],
                encoder_weights=kwargs["encoder_weights"],
                in_channels=15,  # TODO: set number of input channels based on task
                classes=11,
            )
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
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            loss: Name of the loss function
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task(kwargs)

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

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

        self.log("train_loss", loss)  # logging to TensorBoard
        self.log("train_acc_step", self.train_accuracy(y_hat_hard, y))

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics."""
        self.log("train_acc_epoch", self.train_accuracy.compute())

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss)
        self.log("val_acc_step", self.val_accuracy(y_hat_hard, y))

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics."""
        self.log("val_acc_epoch", self.val_accuracy.compute())

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step."""
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc_step", self.test_accuracy(y_hat_hard, y))

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics."""
        self.log("test_acc_epoch", self.test_accuracy.compute())

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
                "verbose": True,
            },
        }


class SEN12MSDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the SEN12MS dataset.

    Implements 80/20 geographic train/val splits and uses the test split from the
    classification dataset definitions. See :func:`setup` for more details.

    Uses the Simplified IGBP scheme defined in the 2020 Data Fusion Competition. See
    https://arxiv.org/abs/2002.08254.
    """

    # Mapping from the IGBP class definitions to the DFC2020, taken from the dataloader
    # here https://github.com/lukasliebel/dfc2020_baseline.
    DFC2020_CLASS_MAPPING = torch.tensor(  # type: ignore[attr-defined]
        [
            0,  # maps 0s to 0
            1,  # maps 1s to 1
            1,  # maps 2s to 1
            1,  # ...
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            5,
            6,
            7,
            6,
            8,
            9,
            10,
        ]
    )

    def __init__(
        self,
        root_dir: str,
        seed: int,
        band_set: str = "all",
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> None:
        """Initialize a LightningDataModule for SEN12MS based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the SEN12MS Dataset classes
            seed: The seed value to use when doing the sklearn based ShuffleSplit
            band_set: The subset of S1/S2 bands to use. Options are: "all",
                "s1", "s2-all", and "s2-reduced" where the "s2-reduced" set includes:
                B2, B3, B4, B8, B11, and B12.
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        assert band_set in SEN12MS.BAND_SETS.keys()

        self.root_dir = root_dir
        self.seed = seed
        self.band_set = band_set
        self.band_indices = SEN12MS.BAND_SETS[band_set]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def custom_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"].float()

        if self.band_set == "all":
            sample["image"][:2] = sample["image"][:2].clip(-25, 0) / -25
            sample["image"][2:] = sample["image"][2:].clip(0, 10000) / 10000
        elif self.band_set == "s1":
            sample["image"][:2] = sample["image"][:2].clip(-25, 0) / -25
        else:
            sample["image"][:] = sample["image"][:].clip(0, 10000) / 10000

        sample["mask"] = sample["mask"][0, :, :].long()
        sample["mask"] = torch.take(  # type: ignore[attr-defined]
            self.DFC2020_CLASS_MAPPING, sample["mask"]
        )

        return sample

    def prepare_data(self) -> None:
        """Initialize the main ``Dataset`` objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """
        self.all_train_dataset = SEN12MS(
            self.root_dir,
            split="train",
            bands=self.band_indices,
            transforms=self.custom_transform,
            checksum=False,
        )

        self.all_test_dataset = SEN12MS(
            self.root_dir,
            split="test",
            bands=self.band_indices,
            transforms=self.custom_transform,
            checksum=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.

        We split samples between train and val geographically with proportions of 80/20.
        This mimics the geographic test set split.
        """
        season_to_int = {
            "winter": 0,
            "spring": 1000,
            "summer": 2000,
            "fall": 3000,
        }

        # A patch is a filename like: "ROIs{num}_{season}_s2_{scene_id}_p{patch_id}.tif"
        # This patch will belong to the scene that is uniquelly identified by its
        # (season, scene_id) tuple. Because the largest scene_id is 149, we can simply
        # give each season a large number and representing a `unique_scene_id` as
        # `season_id + scene_id`.
        scenes = []
        for scene_fn in self.all_train_dataset.ids:
            parts = scene_fn.split("_")
            season_id = season_to_int[parts[1]]
            scene_id = int(parts[3])
            scenes.append(season_id + scene_id)

        train_indices, val_indices = next(
            GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=self.seed).split(
                scenes, groups=scenes
            )
        )

        self.train_dataset = Subset(self.all_train_dataset, train_indices)
        self.val_dataset = Subset(self.all_train_dataset, val_indices)
        self.test_dataset = Subset(
            self.all_test_dataset, range(len(self.all_test_dataset))
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
