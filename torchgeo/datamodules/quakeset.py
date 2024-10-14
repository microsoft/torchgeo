# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""QuakeSet datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import QuakeSet
from .geo import NonGeoDataModule


class QuakeSetDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the QuakeSet dataset.

    .. versionadded:: 0.6
    """

    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new QuakeSetDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.QuakeSet`.
        """
        super().__init__(QuakeSet, batch_size, num_workers, **kwargs)
        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )
        # https://github.com/kornia/kornia/issues/2848
        self.train_aug.keepdim = True
