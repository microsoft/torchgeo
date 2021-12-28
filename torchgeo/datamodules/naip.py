# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""National Agriculture Imagery Program (NAIP) datamodule."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..datasets import NAIP, BoundingBox, Chesapeake13, stack_samples
from ..samplers.batch import RandomBatchGeoSampler
from ..samplers.single import GridGeoSampler

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class NAIPChesapeakeDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NAIP and Chesapeake datasets.

    Uses the train/val/test splits from the dataset.
    """

    # TODO: tune these hyperparams
    length = 1000
    stride = 128

    def __init__(
        self,
        naip_root_dir: str,
        chesapeake_root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        patch_size: int = 256,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for NAIP and Chesapeake based DataLoaders.

        Args:
            naip_root_dir: directory containing NAIP data
            chesapeake_root_dir: directory containing Chesapeake data
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            patch_size: size of patches to sample
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.naip_root_dir = naip_root_dir
        self.chesapeake_root_dir = chesapeake_root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size

    def naip_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the NAIP Dataset.

        Args:
            sample: NAIP image dictionary

        Returns:
            preprocessed NAIP data
        """
        sample["image"] = sample["image"] / 255.0
        sample["image"] = sample["image"].float()

        del sample["bbox"]

        return sample

    def chesapeake_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Chesapeake Dataset.

        Args:
            sample: Chesapeake mask dictionary

        Returns:
            preprocessed Chesapeake data
        """
        sample["mask"] = sample["mask"].long()[0]

        del sample["bbox"]

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        Chesapeake13(self.chesapeake_root_dir, download=False, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: state to set up
        """
        # TODO: these transforms will be applied independently, this won't work if we
        # add things like random horizontal flip
        chesapeake = Chesapeake13(
            self.chesapeake_root_dir, transforms=self.chesapeake_transform
        )
        naip = NAIP(
            self.naip_root_dir,
            chesapeake.crs,
            chesapeake.res,
            transforms=self.naip_transform,
        )
        self.dataset = chesapeake & naip

        # TODO: figure out better train/val/test split
        roi = self.dataset.bounds
        midx = roi.minx + (roi.maxx - roi.minx) / 2
        midy = roi.miny + (roi.maxy - roi.miny) / 2
        train_roi = BoundingBox(roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt)
        val_roi = BoundingBox(midx, roi.maxx, roi.miny, midy, roi.mint, roi.maxt)
        test_roi = BoundingBox(roi.minx, roi.maxx, midy, roi.maxy, roi.mint, roi.maxt)

        self.train_sampler = RandomBatchGeoSampler(
            naip, self.patch_size, self.batch_size, self.length, train_roi
        )
        self.val_sampler = GridGeoSampler(naip, self.patch_size, self.stride, val_roi)
        self.test_sampler = GridGeoSampler(naip, self.patch_size, self.stride, test_roi)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )
