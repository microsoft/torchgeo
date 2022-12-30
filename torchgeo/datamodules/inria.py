# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""InriaAerialImageLabeling datamodule."""

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from kornia.augmentation import Normalize, RandomHorizontalFlip, RandomVerticalFlip
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets import InriaAerialImageLabeling
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from ..transforms.transforms import _ExtractTensorPatches, _RandomNCrop
from .utils import dataset_split


class InriaAerialImageLabelingDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the InriaAerialImageLabeling dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        num_tiles_per_batch: int = 16,
        num_patches_per_tile: int = 16,
        patch_size: Union[Tuple[int, int], int] = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize a new LightningDataModule instance.

        The Inria Aerial Image Labeling dataset contains images that are too large to
        pass directly through a model. Instead, we randomly sample patches from image
        tiles during training and chop up image tiles into patch grids during
        evaluation. During training, the effective batch size is equal to
        ``num_tiles_per_batch`` x ``num_patches_per_tile``.

        Args:
            num_tiles_per_batch: The number of image tiles to sample from during
                training
            num_patches_per_tile: The number of patches to randomly sample from each
                image tile during training
            patch_size: The size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.InriaAerialImageLabeling`
        """
        super().__init__()

        self.num_tiles_per_batch = num_tiles_per_batch
        self.num_patches_per_tile = num_patches_per_tile
        self.patch_size = _to_tuple(patch_size)
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.kwargs = kwargs

        self.train_transform = AugmentationSequential(
            Normalize(mean=0, std=255),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            _RandomNCrop(self.patch_size, self.num_patches_per_tile),
            data_keys=["image", "mask"],
        )
        self.test_transform = AugmentationSequential(
            Normalize(mean=0, std=255),
            _ExtractTensorPatches(self.patch_size),
            data_keys=["image", "mask"],
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        dataset = InriaAerialImageLabeling(split="train", **self.kwargs)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            dataset, self.val_split_pct, self.test_split_pct
        )
        self.predict_dataset = InriaAerialImageLabeling(split="test", **self.kwargs)

    def train_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.num_patches_per_tile,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    def predict_dataloader(self) -> DataLoader[Dict[str, Tensor]]:
        """Return a DataLoader for prediction.

        Returns:
            prediction data loader
        """
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Apply augmentations to batch after transferring to GPU.

        Args:
            batch: A batch of data that needs to be altered or augmented
            dataloader_idx: The index of the dataloader to which the batch belongs

        Returns:
            A batch of data
        """
        if self.trainer:
            if self.trainer.training:
                batch = self.train_transform(batch)
            elif (
                self.trainer.validating
                or self.trainer.testing
                or self.trainer.predicting
            ):
                batch = self.test_transform(batch)

        return batch

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.InriaAerialImageLabeling.plot`.

        .. versionadded:: 0.4
        """
        return self.dataset.plot(*args, **kwargs)
