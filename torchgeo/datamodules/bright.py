# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BRIGHT datamodule."""


from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import BRIGHTDFC2025
from .geo import NonGeoDataModule

class BRIGHTDFC2025DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the BRIGHT_DFC25 dataset.
    
    Implements the default splits that come with the dataset. Note
    that the test split does not have any targets.

    Implements Normalization and RandomCrop for training and Normalization for validation and testing.

    .. versionadded:: 0.7
    """

    # pre image normalization is also used for post_image
    # https://github.com/ChenHongruixuan/BRIGHT/blob/11b1ffafa4d30d2df2081189b56864b0de4e3ed7/dfc25_benchmark/dataset/imutils.py#L5
    mean = torch.Tensor([123.675, 116.28, 103.53])
    std = torch.Tensor([58.395, 57.12, 57.375])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, size: int = 512, **kwargs: Any
    ) -> None:
        """Initialize a new BRIGHT_DFCDataModule instance.
        
        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            size: resize images of input size 512x512 to size x size
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.BRIGHTDFC2025`.
        """
        super().__init__(BRIGHTDFC2025, batch_size, num_workers, **kwargs)

        self.size = size

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomCrop(size),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=None,
            keepdim=True,
        )