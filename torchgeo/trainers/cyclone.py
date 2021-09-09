# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NASA Cyclone dataset trainer."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import models

from ..datasets import TropicalCycloneWindEstimation

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"


class CycloneSimpleRegressionTask(pl.LightningModule):
    """LightningModule for training models on the NASA Cyclone Dataset using MSE loss.

    This does not take into account other per-sample features available in this dataset.
    """

    def config_task(self, kwargs: Dict[str, Any]) -> None:
        """Configures the task based on kwargs parameters."""
        if kwargs["model"] == "resnet18":
            self.model = models.resnet18(pretrained=False, num_classes=1)
        else:
            raise ValueError(f"Model type '{kwargs['model']}' is not valid.")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LightningModule for training simple regression models.

        Keyword Args:
            model: Name of the model to use.
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task(kwargs)

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        return self.model(x)

    # NOTE: See https://github.com/PyTorchLightning/pytorch-lightning/issues/5023 for
    # why we need to tell mypy to ignore a bunch of things
    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step with an MSE loss. Reports MSE and RMSE."""
        x = batch["image"]
        y = batch["wind_speed"].view(-1, 1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", loss)  # logging to TensorBoard

        rmse = torch.sqrt(loss)  # type: ignore[attr-defined]
        self.log("train_rmse", rmse)

        return loss

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports MSE and RMSE."""
        x = batch["image"]
        y = batch["wind_speed"].view(-1, 1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)

        rmse = torch.sqrt(loss)  # type: ignore[attr-defined]
        self.log("val_rmse", rmse)

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step. Reports MSE and RMSE."""
        x = batch["image"]
        y = batch["wind_speed"].view(-1, 1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

        rmse = torch.sqrt(loss)  # type: ignore[attr-defined]
        self.log("test_rmse", rmse)

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


class CycloneDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NASA Cyclone dataset.

    Implements 80/20 train/val splits based on hurricane storm ids.
    See :func:`setup` for more details.
    """

    def __init__(
        self,
        root_dir: str,
        seed: int,
        batch_size: int = 64,
        num_workers: int = 4,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for NASA Cyclone based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the
                TropicalCycloneWindEstimation Datasets classes
            seed: The seed value to use when doing the sklearn based GroupShuffleSplit
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            api_key: The RadiantEarth MLHub API key to use if the dataset needs to be
                downloaded
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.api_key = api_key

    def custom_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"] / 255.0  # scale to [0,1]
        sample["image"] = (
            sample["image"].unsqueeze(0).repeat(3, 1, 1)
        )  # convert to 3 channel
        sample["wind_speed"] = torch.as_tensor(  # type: ignore[attr-defined]
            sample["wind_speed"]
        ).float()

        return sample

    def prepare_data(self) -> None:
        """Initialize the main ``Dataset`` objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """
        do_download = self.api_key is not None
        _ = TropicalCycloneWindEstimation(
            self.root_dir,
            split="train",
            transforms=self.custom_transform,
            download=do_download,
            api_key=self.api_key,
        )

        _ = TropicalCycloneWindEstimation(
            self.root_dir,
            split="test",
            transforms=self.custom_transform,
            download=do_download,
            api_key=self.api_key,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.

        We split samples between train/val by the ``storm_id`` property. I.e. all
        samples with the same ``storm_id`` value will be either in the train or the val
        split. This is important to test one type of generalizability -- given a new
        storm, can we predict its windspeed. The test set, however, contains *some*
        storms from the training set (specifically, the latter parts of the storms) as
        well as some novel storms.
        """
        self.all_train_dataset = TropicalCycloneWindEstimation(
            self.root_dir,
            split="train",
            transforms=self.custom_transform,
            download=False,
        )

        self.all_test_dataset = TropicalCycloneWindEstimation(
            self.root_dir,
            split="test",
            transforms=self.custom_transform,
            download=False,
        )

        storm_ids = []
        for item in self.all_train_dataset.collection:
            storm_id = item["href"].split("/")[0].split("_")[-2]
            storm_ids.append(storm_id)

        train_indices, val_indices = next(
            GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=self.seed).split(
                storm_ids, groups=storm_ids
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
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )
