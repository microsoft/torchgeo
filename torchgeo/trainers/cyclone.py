from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor
from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset

from ..datasets import TropicalCycloneWindEstimation


class CycloneSimpleRegressionTask(pl.LightningModule):
    def __init__(self, model: Module, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs
        self.model = model

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        y = self.model(x)
        return y

    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/5023 for
    # why we need to tell mypy to ignore a bunch of things
    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        x = batch["image"]
        y = batch["wind_speed"].view(-1, 1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", loss)  # Logging to TensorBoard

        rmse = torch.sqrt(loss)  # type: ignore[attr-defined]
        self.log("train_rmse", rmse)

        return loss

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
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
        x = batch["image"]
        y = batch["wind_speed"].view(-1, 1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

        rmse = torch.sqrt(loss)  # type: ignore[attr-defined]
        self.log("test_rmse", rmse)

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore[union-attr]
        )
        return optimizer


class CycloneDataModule(pl.LightningDataModule):
    def __init__(
        self, root_dir: str, seed: int, batch_size: int = 64, num_workers: int = 4
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

    def custom_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transforms a single sample from the Dataset"""
        sample["image"] = sample["image"] / 255.0  # scale to [0,1]
        sample["image"] = (
            sample["image"].unsqueeze(0).repeat(3, 1, 1)
        )  # convert to 3 channel
        sample["wind_speed"] = torch.as_tensor(  # type: ignore[attr-defined]
            sample["wind_speed"]
        ).float()

        return sample

    def setup(self, stage: Optional[str] = None) -> None:

        all_train_dataset = TropicalCycloneWindEstimation(
            self.root_dir,
            split="train",
            transforms=self.custom_transform,
            download=False,
        )

        all_test_dataset = TropicalCycloneWindEstimation(
            self.root_dir,
            split="test",
            transforms=self.custom_transform,
            download=False,
        )

        storm_ids = []
        for item in all_train_dataset.collection:
            storm_id = item["href"].split("/")[0].split("_")[-2]
            storm_ids.append(storm_id)

        train_indices, val_indices = next(
            GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=self.seed).split(
                storm_ids, groups=storm_ids
            )
        )

        self.train_dataset = Subset(all_train_dataset, train_indices)
        self.val_dataset = Subset(all_train_dataset, val_indices)
        self.test_dataset = Subset(all_test_dataset, range(len(all_test_dataset)))

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
