# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Vaihingen datamodule."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import kornia.augmentation as K
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import Compose

from torchgeo.samplers.utils import _to_tuple

from ..datasets import Vaihingen2D
from .utils import dataset_split


class Vaihingen2DDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the Vaihingen2D dataset.

    Uses the train/test splits from the dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        train_batch_size: int = 32,
        num_workers: int = 0,
        patch_size: Union[Tuple[int, int], int] = (64, 64),
        num_patches_per_tile: int = 32,
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Vaihingen2D based DataLoaders.

        Args:
            train_batch_size: The batch size used in the train DataLoader
                (val_batch_size == test_batch_size == 1). The effective batch size
                will be 'train_batch_size' * 'num_patches_per_tile'
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            patch_size: Size of random patch from image and mask (height, width), should
                be a multiple of 32 for most segmentation architectures
            num_patches_per_tile: number of random patches per sample
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.Vaihingen2D`

        .. versionchanged:: 0.4
            'batch_size' is renamed to 'train_batch_size', 'patch_size' and
            'num_patches_per_tile' introduced in order to randomly crop the
            variable size images during training
        """
        super().__init__()

        self.train_batch_size = train_batch_size
        self.num_workers = num_workers
        self.patch_size = _to_tuple(patch_size)
        self.num_patches_per_tile = num_patches_per_tile
        self.val_split_pct = val_split_pct
        self.kwargs = kwargs

        self.rcrop = K.AugmentationSequential(
            K.RandomCrop(self.patch_size, align_corners=False), data_keys=["input", "mask"]
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

        train_dataset = Vaihingen2D(
            split="train", transforms=train_transforms, **self.kwargs
        )

        self.train_dataset: Dataset[Any]
        self.val_dataset: Dataset[Any]

        if self.val_split_pct > 0.0:
            val_dataset = Vaihingen2D(
                split="train", transforms=test_transforms, **self.kwargs
            )
            self.train_dataset, self.val_dataset, _ = dataset_split(
                train_dataset, val_pct=self.val_split_pct, test_pct=0.0
            )
            self.val_dataset.dataset = val_dataset
        else:
            self.train_dataset = train_dataset
            self.val_dataset = train_dataset

        self.test_dataset = Vaihingen2D(
            split="test", transforms=test_transforms, **self.kwargs
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """

        def collate_wrapper(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Define collate function to combine patches per tile and batch size.

            Args:
                batch: sample batch from dataloader containing image and mask

            Returns:
                sample batch where the batch dimension is
                'train_batch_size' * 'num_patches_per_tile'
            """
            r_batch: Dict[str, Any] = default_collate(  # type: ignore[no-untyped-call]
                batch
            )
            r_batch["image"] = rearrange(r_batch["image"], "b t c h w -> (b t) c h w")
            r_batch["mask"] = rearrange(r_batch["mask"], "b t h w -> (b t) h w")
            return r_batch

        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            test data loader
        """
        return DataLoader(
            self.test_dataset, batch_size=1, num_workers=self.num_workers, shuffle=False
        )

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.Vaihingen2D.plot`.

        .. versionadded:: 0.4
        """
        return self.test_dataset.plot(*args, **kwargs)
