# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""SSL4EO datamodule."""

from collections.abc import Callable
from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import SSL4EOL, SSL4EOS12
from ..datasets.utils import Sample
from .geo import NonGeoDataModule


def _normalize_ssl4eo_batch(
    batch: Sample,
    trainer: Any,
    image_aug: Callable[[Sample], Sample],
    video_aug: Callable[[Sample], Sample],
) -> Sample:
    """Normalize single-view samples as images and multi-view samples as videos.

    Args:
        batch: A batch of data that needs to be normalized.
        trainer: The active trainer, if any.
        image_aug: Augmentation pipeline for image batches.
        video_aug: Augmentation pipeline for video batches.

    Returns:
        A normalized batch of data.

    Raises:
        ValueError: If ``batch['image']`` has an unsupported shape.
    """
    if trainer:
        image = batch['image']
        if image.ndim == 4:
            batch = image_aug(batch)
        else:
            batch = video_aug(batch)

    return batch


class SSL4EOLDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SSL4EO-L dataset.

    .. versionadded:: 0.5
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SSL4EOLDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SSL4EOL`.
        """
        super().__init__(SSL4EOL, batch_size, num_workers, **kwargs)

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        )
        self.video_aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = SSL4EOL(**self.kwargs)

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Apply batch augmentations after transfer to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        return _normalize_ssl4eo_batch(batch, self.trainer, self.aug, self.video_aug)


class SSL4EOS12DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SSL4EO-S12 dataset.

    .. versionadded:: 0.5
    """

    # https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/datasets/EuroSat/eurosat_dataset.py#L97
    mean = torch.tensor(0)
    std = torch.tensor(10000)

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SSL4EOS12DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SSL4EOS12`.
        """
        super().__init__(SSL4EOS12, batch_size, num_workers, **kwargs)

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        )
        self.video_aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = SSL4EOS12(**self.kwargs)

    def on_after_batch_transfer(self, batch: Sample, dataloader_idx: int) -> Sample:
        """Apply batch augmentations after transfer to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        return _normalize_ssl4eo_batch(batch, self.trainer, self.aug, self.video_aug)
