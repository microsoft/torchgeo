# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OSCD datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import OSCD
from ..samplers.utils import _to_tuple
from ..transforms.transforms import _RandomNCrop
from .geo import NonGeoDataModule

MEAN = {
    'B01': 1565.696044921875,
    'B02': 1351.3319091796875,
    'B03': 1257.1082763671875,
    'B04': 1254.932861328125,
    'B05': 1388.689208984375,
    'B06': 1827.6710205078125,
    'B07': 2050.2744140625,
    'B08': 1963.4619140625,
    'B8A': 2182.680908203125,
    'B09': 629.837646484375,
    'B10': 14.855598449707031,
    'B11': 1909.8394775390625,
    'B12': 1379.6024169921875,
}

STD = {
    'B01': 263.7977600097656,
    'B02': 394.5567321777344,
    'B03': 508.9673767089844,
    'B04': 726.4053344726562,
    'B05': 686.6111450195312,
    'B06': 730.0204467773438,
    'B07': 822.0133056640625,
    'B08': 842.5917358398438,
    'B8A': 895.7645263671875,
    'B09': 314.8407287597656,
    'B10': 9.417905807495117,
    'B11': 984.9249267578125,
    'B12': 844.7711181640625,
}


class OSCDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the OSCD dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: tuple[int, int] | int = 64,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new OSCDDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.OSCD`.
        """
        super().__init__(OSCD, 1, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.bands = kwargs.get('bands', OSCD.all_bands)
        self.mean = torch.tensor([MEAN[b] for b in self.bands])
        self.std = torch.tensor([STD[b] for b in self.bands])

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _RandomNCrop(self.patch_size, batch_size),
            data_keys=None,
            keepdim=True,
        )
        # https://github.com/kornia/kornia/issues/2848
        self.aug.keepdim = True

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            self.dataset = OSCD(split='train', **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [1 - self.val_split_pct, self.val_split_pct], generator
            )
        if stage in ['test']:
            self.test_dataset = OSCD(split='test', **self.kwargs)
