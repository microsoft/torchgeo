# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 trainer."""

from typing import Any, Dict, Optional, cast

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models
from torch import Tensor
from torch.nn.modules import Conv2d, Linear, Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MetricCollection
from torchvision.transforms import Compose, Normalize

from ..datasets import RESISC45
from ..datasets.utils import dataset_split

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"
Module.__module__ = "torch.nn"
Conv2d.__module__ = "nn.Conv2d"
Linear.__module__ = "nn.Linear"

IN_CHANNELS = 3
NUM_CLASSES = 45


class RESISC45ClassificationTask(pl.LightningModule):
    """LightningModule for training models on the RESISC45 Dataset."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        pretrained = "imagenet" in self.hparams["weights"]

        if "resnet" in self.hparams["classification_model"]:
            self.model = getattr(
                torchvision.models.resnet, self.hparams["classification_model"]
            )(pretrained=pretrained)
            in_features = self.model.fc.in_features
            self.model.fc = Linear(in_features, out_features=NUM_CLASSES)
        else:
            raise ValueError(
                f"Model type '{self.hparams['classification_model']}' is not valid."
            )

        if "resnet" in self.hparams["classification_model"]:

            if self.hparams["weights"] in ["imagenet_only", "random"]:
                pass
            else:
                raise ValueError(
                    f"Weight type '{self.hparams['weights']}' is not valid."
                )
        else:
            pass  # stub for initializing the weights of other models

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
            weights: Either "random", "imagenet_only", "imagenet_and_random", or
                "random_rgb"
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(num_classes=NUM_CLASSES, average="micro"),
                "AverageAccuracy": Accuracy(num_classes=NUM_CLASSES, average="macro"),
            },
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
        """Logs epoch-level training metrics.

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


class RESISC45DataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the RESISC45 dataset.

    Uses the train/val/test splits from the dataset.
    """

    band_means = torch.tensor(  # type: ignore[attr-defined]
        [0.36801773, 0.38097873, 0.343583]
    )

    band_stds = torch.tensor(  # type: ignore[attr-defined]
        [0.14540215, 0.13558227, 0.13203649]
    )

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        weights: str = "random",
        unsupervised_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for RESISC45 based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the RESISC45 Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            weights: Either "random", "imagenet_only", "imagenet_and_random", or
                "random_rgb"
            unsupervised_mode: Makes the train dataloader return imagery from the train,
                val, and test sets
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weights = weights
        self.unsupervised_mode = unsupervised_mode

        self.norm = Normalize(self.band_means, self.band_stds)
        self.transforms = K.AugmentationSequential(
            K.RandomAffine(degrees=30),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            data_keys=["input"],
        )

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        sample["image"] = self.norm(sample["image"])
        return sample

    def kornia_pipeline(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset with Kornia."""
        sample["image"] = self.transforms(sample["image"]).squeeze()
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        RESISC45(self.root_dir, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        transforms = Compose([self.preprocess])

        if not self.unsupervised_mode:

            dataset = RESISC45(
                self.root_dir,
                transforms=transforms,
            )
            self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
                dataset, val_pct=0.2, test_pct=0.2
            )
        else:

            self.train_dataset = RESISC45(
                self.root_dir,
                transforms=transforms,
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
