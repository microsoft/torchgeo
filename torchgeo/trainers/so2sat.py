# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""So2Sat trainer."""

from typing import Any, Dict, Optional, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, IoU, MetricCollection
from torchvision.transforms import Compose

from ..datasets import So2Sat
from ..transforms import RandomHorizontalFlip, RandomVerticalFlip

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"

IN_CHANNELS = 18
NUM_CLASSES = 17


class So2SatClassificationTask(pl.LightningModule):
    """LightningModule for training models on the So2Sat Dataset."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hparams["classification_model"] == "resnet18":
            self.model = torchvision.models.resnet18(
                pretrained=False, num_classes=NUM_CLASSES
            )
            self.model.conv1 = nn.Conv2d(  # type: ignore[attr-defined]
                IN_CHANNELS,
                self.model.inplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['segmentation_model']}' is not valid."
            )

        if self.hparams["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss()  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Loss type '{self.hparams['loss']}' is not valid.")

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            classification_model: Name of the classification model use
            loss: Name of the loss function
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

        self.train_metrics = MetricCollection(
            [
                Accuracy(num_classes=NUM_CLASSES),
                IoU(num_classes=NUM_CLASSES),
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
        y = batch["label"]
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

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

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
        y = batch["label"]
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
            },
        }


class So2SatDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the So2Sat dataset.

    Uses the train/val/test splits from the dataset.
    """

    band_means = torch.tensor(  # type: ignore[attr-defined]
        [
            -3.591224256609313e-05,
            -7.658561276843396e-06,
            5.9373857475971184e-05,
            2.5166231537121083e-05,
            0.04420110659759328,
            0.25761027084996196,
            0.0007556743372573258,
            0.0013503466830024448,
            0.12375696117681859,
            0.1092774636368323,
            0.1010855203267882,
            0.1142398616114001,
            0.1592656692023089,
            0.18147236008771792,
            0.1745740312291377,
            0.19501607349635292,
            0.15428468872076637,
            0.10905050699570007,
        ]
    )

    band_stds = torch.tensor(  # type: ignore[attr-defined]
        [
            0.17555201137417686,
            0.17556463274968204,
            0.45998793417834255,
            0.455988755730148,
            2.8559909213125763,
            8.324800606439833,
            2.4498757382563103,
            1.4647352984509094,
            0.03958795985905458,
            0.047778262752410296,
            0.06636616706371974,
            0.06358874912497474,
            0.07744387147984592,
            0.09101635085921553,
            0.09218466562387101,
            0.10164581233948201,
            0.09991773043519253,
            0.08780632509122865,
        ]
    )

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for So2Sat based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the So2Sat Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = (sample["image"] - self.band_means) / self.band_stds
        sample["image"] = sample["image"].float()
        sample["label"] = sample["label"].long()

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        _ = So2Sat(
            self.root_dir,
            checksum=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        train_transforms = Compose(
            [
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                self.preprocess,
            ]
        )
        val_test_transforms = self.preprocess

        self.train_dataset = So2Sat(
            self.root_dir,
            split="train",
            transforms=train_transforms,
        )

        self.val_dataset = So2Sat(
            self.root_dir,
            split="val",
            transforms=val_test_transforms,
        )

        self.test_dataset = So2Sat(
            self.root_dir,
            split="test",
            transforms=val_test_transforms,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
