# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ReforesTree datamodule."""

from typing import Any

from torch.utils.data import Subset
import kornia.augmentation as K

from ..datasets import ReforesTree
from .geo import NonGeoDataModule
from .utils import group_shuffle_split


class ReforesTreeDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the ReforesTree dataset.

    Implements 80/20 train/val splits.

    .. versionchanged:: 0.1
        
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, size: int = 512, **kwargs: Any
    ) -> None:
        """Initialize a new ReforesTreeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            size: resize images of input size 512x512 to size x size
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ReforesTree`.
        """
        super().__init__(ReforesTree, batch_size, num_workers, **kwargs)

        self.train_aug = K.AugmentationSequential(
            K.Resize(size),
            K.Normalize(self.mean, self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size),
            data_keys=None,
            keepdim=True,
        )

        self.size = size

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            dataset = ReforesTree(split='train', **self.kwargs)
            grouping_paths = [os.path.dirname(path) for path in dataset.file_list]
            train_indices, val_indices = group_shuffle_split(
                grouping_paths, test_size=0.2, random_state=0
            )
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
        if stage in ['test']:
            self.test_dataset = ReforesTree(split='test', **self.kwargs)
