# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all :mod:`torchgeo` data modules."""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule

# TODO: import from lightning_lite instead
from pytorch_lightning.utilities.exceptions import (  # type: ignore[attr-defined]
    MisconfigurationException,
)
from torch import Tensor
from torch.nn import Identity, Module
from torch.utils.data import DataLoader, Dataset


class NonGeoDataModule(LightningDataModule):
    """Base class for data modules lacking geospatial information."""

    #: Training dataset
    train_dataset: Optional[Dataset[Dict[str, Tensor]]] = None

    #: Validation dataset
    val_dataset: Optional[Dataset[Dict[str, Tensor]]] = None

    #: Testing dataset
    test_dataset: Optional[Dataset[Dict[str, Tensor]]] = None

    #: Prediction dataset
    predict_dataset: Optional[Dataset[Dict[str, Tensor]]] = None

    # DataLoader arguments
    batch_size: Optional[int] = None
    train_batch_size: Optional[int] = None
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None
    predict_batch_size: Optional[int] = None
    num_workers = 0

    # Data augmentation
    aug: Optional[Module] = None
    train_aug: Optional[Module] = None
    val_aug: Optional[Module] = None
    test_aug: Optional[Module] = None
    predict_aug: Optional[Module] = None

    def train_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for training.

        Returns:
            A collection of data loaders specifying training samples.

        Raises:
            MisconfigurationException: If :attr:`train_dataset` is not defined.
        """
        if self.train_dataset is not None:
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.train_batch_size or self.batch_size or 1,
                shuffle=True,
                num_workers=self.num_workers,
            )
        else:
            msg = f"{self.__class__.__name__} does not define a 'train_dataset'"
            raise MisconfigurationException(msg)

    def val_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for validation.

        Returns:
            A collection of data loaders specifying validation samples.

        Raises:
            MisconfigurationException: If :attr:`val_dataset` is not defined.
        """
        if self.val_dataset is not None:
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=self.val_batch_size or self.batch_size or 1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        else:
            msg = f"{self.__class__.__name__} does not define a 'val_dataset'"
            raise MisconfigurationException(msg)

    def test_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for testing.

        Returns:
            A collection of data loaders specifying testing samples.

        Raises:
            MisconfigurationException: If :attr:`test_dataset` is not defined.
        """
        if self.test_dataset is not None:
            return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.test_batch_size or self.batch_size or 1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        else:
            msg = f"{self.__class__.__name__} does not define a 'test_dataset'"
            raise MisconfigurationException(msg)

    def predict_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for prediction.

        Returns:
            A collection of data loaders specifying prediction samples.

        Raises:
            MisconfigurationException: If :attr:`predict_dataset` is not defined.
        """
        if self.predict_dataset is not None:
            return DataLoader(
                dataset=self.predict_dataset,
                batch_size=self.predict_batch_size or self.batch_size or 1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        else:
            msg = f"{self.__class__.__name__} does not define a 'predict_dataset'"
            raise MisconfigurationException(msg)

    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug or self.aug or Identity()
            elif self.trainer.validating:
                aug = self.val_aug or self.aug or Identity()
            elif self.trainer.testing:
                aug = self.test_aug or self.aug or Identity()
            elif self.trainer.predicting:
                aug = self.predict_aug or self.aug or Identity()

            batch = aug(batch)

        return batch

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run the plot method of the dataset if one exists.

        Args:
            *args: Arguments passed to plot method.
            **kwargs: Keyword arguments passed to plot method.

        Returns:
            a matplotlib Figure with the image, ground truth, and predictions
        """
        if self.train_dataset is not None:
            if hasattr(self.train_dataset, "plot"):
                return self.train_dataset.plot(*args, **kwargs)
