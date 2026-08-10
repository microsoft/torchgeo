# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""OSCD datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import OSCD, OSCD100
from ..samplers.utils import _to_tuple
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


MEAN100 = {
    'B01': 1699.1854248046875,
    'B02': 1519.08349609375,
    'B03': 1481.2452392578125,
    'B04': 1542.990234375,
    'B05': 1708.63134765625,
    'B06': 2195.550048828125,
    'B07': 2444.430419921875,
    'B08': 2349.7373046875,
    'B8A': 2592.41845703125,
    'B09': 815.1251831054688,
    'B10': 16.794158935546875,
    'B11': 2331.2841796875,
    'B12': 1734.558837890625,
}

STD100 = {
    'B01': 361.23687744140625,
    'B02': 517.9739379882812,
    'B03': 668.9505004882812,
    'B04': 953.0056762695312,
    'B05': 899.5523071289062,
    'B06': 812.8031616210938,
    'B07': 859.7351684570312,
    'B08': 868.78369140625,
    'B8A': 898.2400512695312,
    'B09': 382.8373718261719,
    'B10': 8.891886711120605,
    'B11': 1158.5733642578125,
    'B12': 1069.1888427734375,
}


class OSCDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the OSCD dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        batch_size: int = 32,
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
        super().__init__(OSCD, batch_size=batch_size, num_workers=num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.bands = kwargs.get('bands', OSCD.all_bands)
        self.mean = torch.tensor([MEAN[b] for b in self.bands])
        self.std = torch.tensor([STD[b] for b in self.bands])

        self.aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        transforms = K.AugmentationSequential(
            K.VideoSequential(K.RandomCrop(self.patch_size)),
            data_keys=None,
            keepdim=True,
        )
        if stage in ['fit', 'validate']:
            self.dataset = OSCD(split='train', transforms=transforms, **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [1 - self.val_split_pct, self.val_split_pct], generator
            )
        if stage in ['test']:
            self.test_dataset = OSCD(split='test', transforms=transforms, **self.kwargs)


class OSCD100DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the OSCD100 dataset.

    Intended for tutorials and demonstrations, not benchmarking.

    .. versionadded:: 0.9
    """

    def __init__(
        self,
        batch_size: int = 8,
        patch_size: tuple[int, int] | int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new OSCD100DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.OSCD100`.
        """
        super().__init__(
            OSCD100, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.patch_size = _to_tuple(patch_size)

        self.bands = kwargs.get('bands', OSCD.all_bands)
        self.mean = torch.tensor([MEAN100[b] for b in self.bands])
        self.std = torch.tensor([STD100[b] for b in self.bands])

        self.aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        transforms = K.AugmentationSequential(
            K.VideoSequential(K.RandomCrop(self.patch_size)),
            data_keys=None,
            keepdim=True,
        )
        if stage in ['fit']:
            self.train_dataset = OSCD100(
                split='train', transforms=transforms, **self.kwargs
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = OSCD100(
                split='val', transforms=transforms, **self.kwargs
            )
        if stage in ['test']:
            self.test_dataset = OSCD100(
                split='test', transforms=transforms, **self.kwargs
            )
