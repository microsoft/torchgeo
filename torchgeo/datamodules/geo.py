# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all :mod:`torchgeo` data modules."""

from collections.abc import Callable
from typing import Any, cast

import kornia.augmentation as K
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


class BaseDataModule(LightningDataModule):
    """Base class for all TorchGeo data modules.

    .. versionadded:: 0.5
    """

    mean = torch.tensor(0)
    std = torch.tensor(255)

    def __init__(
        self,
        dataset_class: type[Dataset[dict[str, Tensor]]],
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new BaseDataModule instance.

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
        self.dataset: Dataset[dict[str, Tensor]] | None = None
        self.train_dataset: Dataset[dict[str, Tensor]] | None = None
        self.val_dataset: Dataset[dict[str, Tensor]] | None = None
        self.test_dataset: Dataset[dict[str, Tensor]] | None = None
        self.predict_dataset: Dataset[dict[str, Tensor]] | None = None

        # Data loaders
        self.train_batch_size: int | None = None
        self.val_batch_size: int | None = None
        self.test_batch_size: int | None = None
        self.predict_batch_size: int | None = None

        # Data augmentation
        Transform = Callable[[dict[str, Tensor]], dict[str, Tensor]]
        self.aug: Transform = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=['image']
        )
        self.train_aug: Transform | None = None
        self.val_aug: Transform | None = None
        self.test_aug: Transform | None = None
        self.predict_aug: Transform | None = None

    def prepare_data(self) -> None:
        """Download and prepare data.

        During distributed training, this method is called only within a single process
        to avoid corrupted data. This method should not set state since it is not called
        on every device, use ``setup`` instead.
        """
        if self.kwargs.get('download', False):
            self.dataset_class(**self.kwargs)

    def _valid_attribute(self, *args: str) -> Any:
        """Find a valid attribute with length > 0.

        Args:
            args: One or more names of attributes to check.

        Returns:
            The first valid attribute found.

        Raises:
            MisconfigurationException: If no attribute is defined, or has length 0.
        """
        for arg in args:
            obj = getattr(self, arg)

            if obj is None:
                continue

            if not obj:
                msg = f'{self.__class__.__name__}.{arg} has length 0.'
                raise MisconfigurationException(msg)

            return obj

        msg = f'{self.__class__.__name__}.setup must define one of {args}.'
        raise MisconfigurationException(msg)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                split = 'train'
            elif self.trainer.validating or self.trainer.sanity_checking:
                split = 'val'
            elif self.trainer.testing:
                split = 'test'
            elif self.trainer.predicting:
                split = 'predict'

            aug = self._valid_attribute(f'{split}_aug', 'aug')
            batch = aug(batch)

        return batch


class GeoDataModule(BaseDataModule):
    """Base class for data modules containing geospatial information.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        dataset_class: type[GeoDataset],
        batch_size: int = 1,
        patch_size: int | tuple[int, int] = 64,
        length: int | None = None,
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
        super().__init__(dataset_class, batch_size, num_workers, **kwargs)

        self.patch_size = patch_size
        self.length = length

        # Collation
        self.collate_fn = stack_samples

        # Samplers
        self.sampler: GeoSampler | None = None
        self.train_sampler: GeoSampler | None = None
        self.val_sampler: GeoSampler | None = None
        self.test_sampler: GeoSampler | None = None
        self.predict_sampler: GeoSampler | None = None

        # Batch samplers
        self.batch_sampler: BatchGeoSampler | None = None
        self.train_batch_sampler: BatchGeoSampler | None = None
        self.val_batch_sampler: BatchGeoSampler | None = None
        self.test_batch_sampler: BatchGeoSampler | None = None
        self.predict_batch_sampler: BatchGeoSampler | None = None

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = cast(
                GeoDataset,
                self.dataset_class(  # type: ignore[call-arg]
                    split='train', **self.kwargs
                ),
            )
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = cast(
                GeoDataset,
                self.dataset_class(  # type: ignore[call-arg]
                    split='val', **self.kwargs
                ),
            )
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ['test']:
            self.test_dataset = cast(
                GeoDataset,
                self.dataset_class(  # type: ignore[call-arg]
                    split='test', **self.kwargs
                ),
            )
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f'{split}_dataset', 'dataset')
        sampler = self._valid_attribute(
            f'{split}_batch_sampler', f'{split}_sampler', 'batch_sampler', 'sampler'
        )
        batch_size = self._valid_attribute(f'{split}_batch_size', 'batch_size')

        if isinstance(sampler, BatchGeoSampler):
            batch_size = 1
            batch_sampler = sampler
            sampler = None
        else:
            batch_sampler = None

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for training.

        Returns:
            A collection of data loaders specifying training samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        return self._dataloader_factory('train')

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for validation.

        Returns:
            A collection of data loaders specifying validation samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        return self._dataloader_factory('val')

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for testing.

        Returns:
            A collection of data loaders specifying testing samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        return self._dataloader_factory('test')

    def predict_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for prediction.

        Returns:
            A collection of data loaders specifying prediction samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        return self._dataloader_factory('predict')

    def transfer_batch_to_device(
        self, batch: dict[str, Tensor], device: torch.device, dataloader_idx: int
    ) -> dict[str, Tensor]:
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
        del batch['crs']
        del batch['bbox']

        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch


class NonGeoDataModule(BaseDataModule):
    """Base class for data modules lacking geospatial information.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        dataset_class: type[NonGeoDataset],
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
        super().__init__(dataset_class, batch_size, num_workers, **kwargs)

        # Collation
        self.collate_fn = default_collate

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Called at the beginning of fit, validate, test, or predict. During distributed
        training, this method is called from every process across all the nodes. Setting
        state here is recommended.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='train', **self.kwargs
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='val', **self.kwargs
            )
        if stage in ['test']:
            self.test_dataset = self.dataset_class(  # type: ignore[call-arg]
                split='test', **self.kwargs
            )

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f'{split}_dataset', 'dataset')
        batch_size = self._valid_attribute(f'{split}_batch_size', 'batch_size')
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == 'train',
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for training.

        Returns:
            A collection of data loaders specifying training samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset, or if the dataset has length 0.
        """
        return self._dataloader_factory('train')

    def val_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for validation.

        Returns:
            A collection of data loaders specifying validation samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset, or if the dataset has length 0.
        """
        return self._dataloader_factory('val')

    def test_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for testing.

        Returns:
            A collection of data loaders specifying testing samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset, or if the dataset has length 0.
        """
        return self._dataloader_factory('test')

    def predict_dataloader(self) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders for prediction.

        Returns:
            A collection of data loaders specifying prediction samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset, or if the dataset has length 0.
        """
        return self._dataloader_factory('predict')
