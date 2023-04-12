# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""UC Merced datamodule."""

from typing import Any, Dict

import kornia.augmentation as K
import torch
import torch.nn.functional as F

from ..datasets import UCMerced
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class UCMercedDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the UC Merced dataset.

    Uses random train/val/test splits.
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new UCMercedDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.UCMerced`.
        """

        def default_transform(
            sample: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            sample["image"] = F.interpolate(sample["image"], size=256)
            return sample

        kwargs["transforms"] = kwargs.get("transforms", default_transform)
        super().__init__(UCMerced, batch_size, num_workers, **kwargs)

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size=256),
            data_keys=["image"],
        )
