# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CaFFe datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import CaFFe
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class CaFFeDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the CaFFe dataset.

    Implements the default splits that come with the dataset.

    .. versionadded:: 0.7
    """

    mean = torch.Tensor([0.5517])
    std = torch.Tensor([11.8478])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, size: int = 256, **kwargs: Any
    ) -> None:
        """Initialize a new CaFFeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            size: resize images of input size 512x512 to size x size
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CaFFe`.
        """
        super().__init__(CaFFe, batch_size, num_workers, **kwargs)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=['image', 'mask'],
        )

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size),
            data_keys=['image', 'mask'],
        )

        self.size = size
