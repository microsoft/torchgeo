# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Potsdam datamodule."""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import kornia.augmentation as K
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from ..datasets import Potsdam2D
from ..datasets.utils import collate_patches_per_tile
from ..samplers.utils import _to_tuple
from .utils import dataset_split


class Potsdam2DDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the Potsdam2D dataset.

    Uses the train/test splits from the dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        patch_size: Union[Tuple[int, int], int] = (64, 64),
        num_tiles_per_batch: int = 16,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Potsdam2D based DataLoaders.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            patch_size: Size of random patch from image and mask (height, width), should
                be a multiple of 32 for most segmentation architectures
            num_tiles_per_batch: number of random tiles to consider sampling patches
                from per sample, should evenly divide batch_size and be less than
                or equal to batch_size
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.Potsdam2D`

        .. versionchanged:: 0.4
            'patch_size' and 'num_tiles_per_batch' introduced in order to randomly
            crop the variable size images during training
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct
        self.kwargs = kwargs

        assert (
            self.batch_size >= num_tiles_per_batch
        ), "num_tiles_per_batch should be less than or equal to batch_size."

        self.num_patches_per_tile = self.batch_size // num_tiles_per_batch
        self.num_tiles_per_batch = num_tiles_per_batch

        if (self.num_patches_per_tile % 2) != 0 and (
            self.num_patches_per_tile != num_tiles_per_batch
        ):
            warnings.warn(
                "The effective batch size"
                f" will differ from the specified {batch_size}"
                f" and be {self.num_patches_per_tile * num_tiles_per_batch} instead."
                " To match the batch_size exactly, ensure that"
                " num_tiles_per_batch evenly divides batch_size"
            )

        self.rcrop = K.AugmentationSequential(
            K.RandomCrop(self.patch_size), data_keys=["input", "mask"]
        )

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up

        .. versionchanged:: 0.4
            Add functionality to randomly crop patches from a tile during
            training and pad validation and test samples to next multiple of 32
        """

        def n_random_crop(sample: Dict[str, Any]) -> Dict[str, Any]:
            """Construct 'num_patches_per_tile' random patches of input tile.

            Args:
                sample: contains image and mask tile from dataset

            Returns:
                stacked randomly cropped patches from input tile
            """
            images, masks = [], []
            for i in range(self.num_patches_per_tile):
                image, mask = self.rcrop(sample["image"], sample["mask"].float())
                images.append(image.squeeze(0))
                masks.append(mask.squeeze().long())

            sample["image"] = torch.stack(images)
            sample["mask"] = torch.stack(masks)
            return sample

        def pad_to(sample: Dict[str, Any]) -> Dict[str, Any]:
            """Pad image and mask to next multiple of 32.

            Args:
                sample: contains image and mask sample from dataset

            Returns:
                padded image and mask
            """
            h, w = sample["image"].shape[1], sample["image"].shape[2]
            new_h = int(32 * ((h // 32) + 1))
            new_w = int(32 * ((w // 32) + 1))

            padto = K.PadTo((new_h, new_w))

            sample["image"] = padto(sample["image"])[0]
            sample["mask"] = padto(sample["mask"].float()).long()[0, 0]
            return sample

        train_transforms = Compose([self.preprocess, n_random_crop])
        # for testing and validation we pad all inputs to next larger multiple of 32
        # to avoid issues with upsampling paths in encoder-decoder architectures
        test_transforms = Compose([self.preprocess, pad_to])

        train_dataset = Potsdam2D(
            split="train", transforms=train_transforms, **self.kwargs
        )

        self.train_dataset: Dataset[Any]
        self.val_dataset: Dataset[Any]

        if self.val_split_pct > 0.0:
            val_dataset = Potsdam2D(
                split="train", transforms=test_transforms, **self.kwargs
            )
            self.train_dataset, self.val_dataset, _ = dataset_split(
                train_dataset, val_pct=self.val_split_pct, test_pct=0.0
            )
            self.val_dataset.dataset = val_dataset
        else:
            self.train_dataset = train_dataset
            self.val_dataset = train_dataset

        self.test_dataset = Potsdam2D(
            split="test", transforms=test_transforms, **self.kwargs
        )

    def train_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.num_tiles_per_batch,
            num_workers=self.num_workers,
            collate_fn=collate_patches_per_tile,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        if self.val_split_pct > 0.0:
            return DataLoader(
                self.val_dataset,
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=collate_patches_per_tile,
            )

    def test_dataloader(self) -> DataLoader[Dict[str, Any]]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.Potsdam2D.plot`.

        .. versionadded:: 0.4
        """
        return self.test_dataset.plot(*args, **kwargs)
