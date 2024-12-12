# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TreeSatAI datamodules."""

from typing import Any

import kornia.augmentation as K
import torch
from torch import Tensor
from torch.utils.data import random_split

from ..datasets import TreeSatAI
from ..samplers.utils import _to_tuple
from .geo import NonGeoDataModule

# https://git.tu-berlin.de/rsim/treesat_benchmark/-/blob/master/configs/multimodal/AllModes_Xformer_ResnetScratch_v8.json
means = {
    'aerial': [
        151.26809261440323,
        93.1159469148246,
        85.05016794624635,
        81.0471576353153,
    ],
    's1': [-6.933713050794077, -12.628564056094067, 0.47448312147709354],
    's2': [
        231.43385024546893,
        376.94788434611434,
        241.03688288984037,
        2809.8421354087955,
        616.5578221193639,
        2104.3826773960823,
        2695.083864757169,
        2969.868417923599,
        1306.0814241837832,
        587.0608264363341,
        249.1888624097736,
        2950.2294375352285,
    ],
}
stds = {
    'aerial': [
        48.70879149145466,
        33.59622314610158,
        28.000497087051126,
        33.683983599997724,
    ],
    's1': [87.8762246957811, 47.03070478433704, 1.297291303623673],
    's2': [
        123.16515044781909,
        139.78991338362886,
        140.6154081184225,
        786.4508872594147,
        202.51268536579394,
        530.7255451201194,
        710.2650071967689,
        777.4421400779165,
        424.30312334282684,
        247.21468849049668,
        122.80062680549261,
        702.7404237034002,
    ],
}


class TreeSatAIDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the TreeSatAI dataset.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int | tuple[int, int] = 304,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new TreeSatAIDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.TreeSatAI`.
        """
        super().__init__(TreeSatAI, batch_size, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        self.sensors = kwargs.get('sensors', TreeSatAI.all_sensors)

        self.train_aug = K.AugmentationSequential(
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            K.Resize(self.patch_size),
            data_keys=None,
            keepdim=True,
        )
        self.aug = K.AugmentationSequential(
            K.Resize(self.patch_size), data_keys=None, keepdim=True
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        # Convert 90-10 train-test split to 80-10-10 train-val-test split
        train_val_dataset = TreeSatAI(split='train', **self.kwargs)
        self.test_dataset = TreeSatAI(split='test', **self.kwargs)
        generator = torch.Generator().manual_seed(0)
        self.train_dataset, self.val_dataset = random_split(
            train_val_dataset,
            [len(train_val_dataset) - len(self.test_dataset), len(self.test_dataset)],
            generator=generator,
        )

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
        batch = super().on_after_batch_transfer(batch, dataloader_idx)

        images = []
        for sensor in self.sensors:
            aug = K.Normalize(mean=means[sensor], std=stds[sensor], keepdim=True)
            batch[f'image_{sensor}'] = aug(batch[f'image_{sensor}'])
            images.append(batch[f'image_{sensor}'])

        batch['image'] = torch.cat(images, dim=1)

        return batch
