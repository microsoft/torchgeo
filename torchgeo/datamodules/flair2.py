"""This module contains the FLAIR2DataModule class for loading the FLAIR2 dataset.

The FLAIR dataset is released under open license 2.0:
- https://www.etalab.gouv.fr/wp-content/uploads/2018/11/open-licence.pdf
- https://ignf.github.io/FLAIR/#FLAIR2

Code for loading dataset licensed under the MIT License.
"""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import FLAIR2, FLAIR2Toy
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class FLAIR2DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the FLAIR2 dataset.

    Uses the train/test splits from the dataset.
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: tuple[int, int] | int = 64,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        use_toy: bool = False,
        augs: AugmentationSequential | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new FLAIR2DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            use_toy: Whether to use the toy version of the dataset.
            augs: Optional augmentations to apply to the dataset.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.FLAIR2`.
            
            ..versionadded:: 0.7
        """
        self.ds_class = FLAIR2 if not use_toy else FLAIR2Toy
        
        super().__init__(self.ds_class, batch_size, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.aug: AugmentationSequential = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=['image', 'mask'],
        )

        self.augs = augs if augs is not None else self.aug

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            self.dataset = self.ds_class(split="train", **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [1 - self.val_split_pct, self.val_split_pct], generator
            )
        if stage in ['test']:
            self.test_dataset = self.ds_class(split="test", **self.kwargs)