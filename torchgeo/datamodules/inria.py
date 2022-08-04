# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""InriaAerialImageLabeling datamodule."""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import kornia.augmentation as K
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from einops import rearrange
from kornia.contrib import compute_padding, extract_tensor_patches
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from ..datasets import InriaAerialImageLabeling
from ..samplers.utils import _to_tuple
from .utils import dataset_split


def collate_wrapper(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten wrapper."""
    r_batch: Dict[str, Any] = default_collate(batch)  # type: ignore[no-untyped-call]
    r_batch["image"] = torch.flatten(r_batch["image"], 0, 1)
    if "mask" in r_batch:
        r_batch["mask"] = torch.flatten(r_batch["mask"], 0, 1)

    return r_batch


class InriaAerialImageLabelingDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the InriaAerialImageLabeling dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.3
    """

    h, w = 5000, 5000

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.1,
        patch_size: Union[int, Tuple[int, int]] = 512,
        num_patches_per_tile: int = 32,
        predict_on: str = "test",
    ) -> None:
        """Initialize a LightningDataModule for InriaAerialImageLabeling.

        Args:
            root_dir: The ``root`` arugment to pass to the InriaAerialImageLabeling
                Dataset classes
            batch_size: The batch size used in the train DataLoader
                (val_batch_size == test_batch_size == 1)
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
            patch_size: Size of random patch from image and mask (height, width)
            num_patches_per_tile: Number of random patches per sample
            predict_on: Directory/Dataset of images to run inference on
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.patch_size = cast(Tuple[int, int], _to_tuple(patch_size))
        self.num_patches_per_tile = num_patches_per_tile
        self.augmentations = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
        )
        self.predict_on = predict_on
        self.random_crop = K.AugmentationSequential(
            K.RandomCrop(self.patch_size, p=1.0, keepdim=False),
            data_keys=["input", "mask"],
        )

    def patch_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract patches from single sample."""
        assert sample["image"].ndim == 3
        _, h, w = sample["image"].shape

        padding = compute_padding((h, w), self.patch_size)
        sample["original_shape"] = (h, w)
        sample["patch_shape"] = self.patch_size
        sample["padding"] = padding
        sample["image"] = extract_tensor_patches(
            sample["image"].unsqueeze(0),
            self.patch_size,
            self.patch_size,
            padding=padding,
        )
        sample["image"] = rearrange(sample["image"], "() t c h w -> t () c h w")
        return sample

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        sample["image"] = torch.clip(sample["image"], min=0.0, max=1.0)

        if "mask" in sample:
            sample["mask"] = rearrange(sample["mask"], "h w -> () h w")

        return sample

    def n_random_crop(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get n random crops."""
        images, masks = [], []
        for _ in range(self.num_patches_per_tile):
            image, mask = sample["image"], sample["mask"]
            # RandomCrop needs image and mask to be in float
            mask = mask.to(torch.float)
            image, mask = self.random_crop(image, mask)
            images.append(image.squeeze())
            masks.append(mask.squeeze(0).long())
        sample["image"] = torch.stack(images)  # (t,c,h,w)
        sample["mask"] = torch.stack(masks)  # (t, 1, h, w)
        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        train_transforms = T.Compose([self.preprocess, self.n_random_crop])
        test_transforms = T.Compose([self.preprocess, self.patch_sample])

        train_dataset = InriaAerialImageLabeling(
            self.root_dir, split="train", transforms=train_transforms
        )

        self.train_dataset: Dataset[Any]
        self.val_dataset: Dataset[Any]
        self.test_dataset: Dataset[Any]

        if self.val_split_pct > 0.0:
            if self.test_split_pct > 0.0:
                self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
                    train_dataset,
                    val_pct=self.val_split_pct,
                    test_pct=self.test_split_pct,
                )
            else:
                self.train_dataset, self.val_dataset = dataset_split(
                    train_dataset, val_pct=self.val_split_pct
                )
                self.test_dataset = self.val_dataset
        else:
            self.train_dataset = train_dataset
            self.val_dataset = train_dataset
            self.test_dataset = train_dataset

        assert self.predict_on == "test"
        self.predict_dataset = InriaAerialImageLabeling(
            self.root_dir, self.predict_on, transforms=test_transforms
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for prediction."""
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=collate_wrapper,
            shuffle=False,
        )

    def on_after_batch_transfer(
        self, batch: Dict[str, Any], dataloader_idx: int
    ) -> Dict[str, Any]:
        """Apply augmentations to batch after transferring to GPU.

        Args:
            batch (dict): A batch of data that needs to be altered or augmented.
            dataloader_idx (int): The index of the dataloader to which the batch
            belongs.

        Returns:
            dict: A batch of data
        """
        # Training
        if (
            hasattr(self, "trainer")
            and self.trainer is not None
            and hasattr(self.trainer, "training")
            and self.trainer.training
            and self.augmentations is not None
        ):
            batch["mask"] = batch["mask"].to(torch.float)
            batch["image"], batch["mask"] = self.augmentations(
                batch["image"], batch["mask"]
            )
            batch["mask"] = batch["mask"].to(torch.long)

        # Validation
        if "mask" in batch:
            batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch
