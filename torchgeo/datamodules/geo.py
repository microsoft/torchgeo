# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all :mod:`torchgeo` data modules."""

from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, default_collate

from ..datasets import GeoDataset, NonGeoDataset, stack_samples
from ..samplers import (
    BatchGeoSampler,
    GeoSampler,
    GridGeoSampler,
    RandomBatchGeoSampler,
)
from ..transforms import AugmentationSequential
from .utils import MisconfigurationException


class GeoDataModule(LightningDataModule):  # type: ignore[misc]
    """Base class for data modules containing geospatial information.

    .. versionadded:: 0.4
    """

    mean = torch.tensor(0)
    std = torch.tensor(255)

    def __init__(
        self,
        dataset_class: Type[GeoDataset],
        batch_size: int = 1,
        patch_size: Union[int, Tuple[int, int]] = 64,
        length: int = 1000,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new GeoDataModule instance.

        Args:
            dataset_class: Class used to instantiate a new dataset.
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__()

        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.length = length
        self.num_workers = num_workers
        self.kwargs = kwargs

        # Datasets
        self.dataset: Optional[Dataset[Dict[str, Tensor]]] = None
        self.train_dataset: Optional[Dataset[Dict[str, Tensor]]] = None
        self.val_dataset: Optional[Dataset[Dict[str, Tensor]]] = None
        self.test_dataset: Optional[Dataset[Dict[str, Tensor]]] = None
        self.predict_dataset: Optional[Dataset[Dict[str, Tensor]]] = None

        # Samplers
        self.sampler: Optional[GeoSampler] = None
        self.train_sampler: Optional[GeoSampler] = None
        self.val_sampler: Optional[GeoSampler] = None
        self.test_sampler: Optional[GeoSampler] = None
        self.predict_sampler: Optional[GeoSampler] = None

        # Batch samplers
        self.batch_sampler: Optional[BatchGeoSampler] = None
        self.train_batch_sampler: Optional[BatchGeoSampler] = None
        self.val_batch_sampler: Optional[BatchGeoSampler] = None
        self.test_batch_sampler: Optional[BatchGeoSampler] = None
        self.predict_batch_sampler: Optional[BatchGeoSampler] = None

        # Data loaders
        self.train_batch_size: Optional[int] = None
        self.val_batch_size: Optional[int] = None
        self.test_batch_size: Optional[int] = None
        self.predict_batch_size: Optional[int] = None

        # Collation
        self.collate_fn = stack_samples

        # Data augmentation
        Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]
        self.aug: Transform = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image"]
        )
        self.train_aug: Optional[Transform] = None
        self.val_aug: Optional[Transform] = None
        self.test_aug: Optional[Transform] = None
        self.predict_aug: Optional[Transform] = None

    def prepare_data(self) -> None:
        """Download and prepare data.

        During distributed training, this method is called only within a single process
        to avoid corrupted data. This method should not set state since it is not called
        on every device, use :meth:`setup` instead.
        """
        if self.kwargs.get("download", False):
            self.dataset_class(**self.kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="train", **self.kwargs
            )
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="val", **self.kwargs
            )
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="test", **self.kwargs
            )
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

    def train_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for training.

        Returns:
            A collection of data loaders specifying training samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'train_dataset'.
        """
        dataset = self.train_dataset or self.dataset
        sampler = self.train_sampler or self.sampler
        batch_sampler = self.train_batch_sampler or self.batch_sampler
        if dataset is not None and (sampler or batch_sampler) is not None:
            batch_size = self.train_batch_size or self.batch_size
            if batch_sampler is not None:
                batch_size = 1
                sampler = None
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'train_dataset'"
            raise MisconfigurationException(msg)

    def val_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for validation.

        Returns:
            A collection of data loaders specifying validation samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'val_dataset'.
        """
        dataset = self.val_dataset or self.dataset
        sampler = self.val_sampler or self.sampler
        batch_sampler = self.val_batch_sampler or self.batch_sampler
        if dataset is not None and (sampler or batch_sampler) is not None:
            batch_size = self.val_batch_size or self.batch_size
            if batch_sampler is not None:
                batch_size = 1
                sampler = None
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'val_dataset'"
            raise MisconfigurationException(msg)

    def test_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for testing.

        Returns:
            A collection of data loaders specifying testing samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'test_dataset'.
        """
        dataset = self.test_dataset or self.dataset
        sampler = self.test_sampler or self.sampler
        batch_sampler = self.test_batch_sampler or self.batch_sampler
        if dataset is not None and (sampler or batch_sampler) is not None:
            batch_size = self.test_batch_size or self.batch_size
            if batch_sampler is not None:
                batch_size = 1
                sampler = None
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'test_dataset'"
            raise MisconfigurationException(msg)

    def predict_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for prediction.

        Returns:
            A collection of data loaders specifying prediction samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'predict_dataset'.
        """
        dataset = self.predict_dataset or self.dataset
        sampler = self.predict_sampler or self.sampler
        batch_sampler = self.predict_batch_sampler or self.batch_sampler
        if dataset is not None and (sampler or batch_sampler) is not None:
            batch_size = self.predict_batch_size or self.batch_size
            if batch_sampler is not None:
                batch_size = 1
                sampler = None
            return DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'predict_dataset'"
            raise MisconfigurationException(msg)

    def transfer_batch_to_device(
        self, batch: Dict[str, Tensor], device: torch.device, dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Transfer batch to device.

        Defines how custom data types are moved to the target device.

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A reference to the data on the new device.
        """
        # Non-Tensor values cannot be moved to a device
        del batch["crs"]
        del batch["bbox"]

        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

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
                aug = self.train_aug or self.aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug or self.aug
            elif self.trainer.testing:
                aug = self.test_aug or self.aug
            elif self.trainer.predicting:
                aug = self.predict_aug or self.aug

            batch = aug(batch)

        return batch

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run the plot method of the validation dataset if one exists.

        Should only be called during 'fit' or 'validate' stages as ``val_dataset``
        may not exist during other stages.

        Args:
            *args: Arguments passed to plot method.
            **kwargs: Keyword arguments passed to plot method.

        Returns:
            A matplotlib Figure with the image, ground truth, and predictions.
        """
        dataset = self.val_dataset or self.dataset
        if dataset is not None:
            if hasattr(dataset, "plot"):
                return dataset.plot(*args, **kwargs)


class NonGeoDataModule(LightningDataModule):  # type: ignore[misc]
    """Base class for data modules lacking geospatial information.

    .. versionadded:: 0.4
    """

    mean = torch.tensor(0)
    std = torch.tensor(255)

    def __init__(
        self,
        dataset_class: Type[NonGeoDataset],
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new NonGeoDataModule instance.

        Args:
            dataset_class: Class used to instantiate a new dataset.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to ``dataset_class``
        """
        super().__init__()

        self.dataset_class = dataset_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        # Datasets
        self.dataset: Optional[Dataset[Dict[str, Tensor]]] = None
        self.train_dataset: Optional[Dataset[Dict[str, Tensor]]] = None
        self.val_dataset: Optional[Dataset[Dict[str, Tensor]]] = None
        self.test_dataset: Optional[Dataset[Dict[str, Tensor]]] = None
        self.predict_dataset: Optional[Dataset[Dict[str, Tensor]]] = None

        # Data loaders
        self.train_batch_size: Optional[int] = None
        self.val_batch_size: Optional[int] = None
        self.test_batch_size: Optional[int] = None
        self.predict_batch_size: Optional[int] = None

        # Collation
        self.collate_fn = default_collate

        # Data augmentation
        Transform = Callable[[Dict[str, Tensor]], Dict[str, Tensor]]
        self.aug: Transform = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image"]
        )
        self.train_aug: Optional[Transform] = None
        self.val_aug: Optional[Transform] = None
        self.test_aug: Optional[Transform] = None
        self.predict_aug: Optional[Transform] = None

    def prepare_data(self) -> None:
        """Download and prepare data.

        During distributed training, this method is called only within a single process
        to avoid corrupted data. This method should not set state since it is not called
        on every device, use :meth:`setup` instead.
        """
        if self.kwargs.get("download", False):
            self.dataset_class(**self.kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="train", **self.kwargs
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="val", **self.kwargs
            )
        if stage in ["test"]:
            self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                split="test", **self.kwargs
            )

    def train_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for training.

        Returns:
            A collection of data loaders specifying training samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'train_dataset'.
        """
        dataset = self.train_dataset or self.dataset
        if dataset is not None:
            return DataLoader(
                dataset=dataset,
                batch_size=self.train_batch_size or self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'train_dataset'"
            raise MisconfigurationException(msg)

    def val_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for validation.

        Returns:
            A collection of data loaders specifying validation samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'val_dataset'.
        """
        dataset = self.val_dataset or self.dataset
        if dataset is not None:
            return DataLoader(
                dataset=dataset,
                batch_size=self.val_batch_size or self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'val_dataset'"
            raise MisconfigurationException(msg)

    def test_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for testing.

        Returns:
            A collection of data loaders specifying testing samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'test_dataset'.
        """
        dataset = self.test_dataset or self.dataset
        if dataset is not None:
            return DataLoader(
                dataset=dataset,
                batch_size=self.test_batch_size or self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'test_dataset'"
            raise MisconfigurationException(msg)

    def predict_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for prediction.

        Returns:
            A collection of data loaders specifying prediction samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                'predict_dataset'.
        """
        dataset = self.predict_dataset or self.dataset
        if dataset is not None:
            return DataLoader(
                dataset=dataset,
                batch_size=self.predict_batch_size or self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        else:
            msg = f"{self.__class__.__name__}.setup does not define a 'predict_dataset'"
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
                aug = self.train_aug or self.aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug or self.aug
            elif self.trainer.testing:
                aug = self.test_aug or self.aug
            elif self.trainer.predicting:
                aug = self.predict_aug or self.aug

            batch = aug(batch)

        return batch

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run the plot method of the validation dataset if one exists.

        Should only be called during 'fit' or 'validate' stages as ``val_dataset``
        may not exist during other stages.

        Args:
            *args: Arguments passed to plot method.
            **kwargs: Keyword arguments passed to plot method.

        Returns:
            A matplotlib Figure with the image, ground truth, and predictions.
        """
        dataset = self.dataset or self.val_dataset
        if dataset is not None:
            if hasattr(dataset, "plot"):
                return dataset.plot(*args, **kwargs)
