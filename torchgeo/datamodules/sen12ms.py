# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SEN12MS datamodule."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from ..datasets import SEN12MS

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class SEN12MSDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the SEN12MS dataset.

    Implements 80/20 geographic train/val splits and uses the test split from the
    classification dataset definitions. See :func:`setup` for more details.

    Uses the Simplified IGBP scheme defined in the 2020 Data Fusion Competition. See
    https://arxiv.org/abs/2002.08254.
    """

    #: Mapping from the IGBP class definitions to the DFC2020, taken from the dataloader
    #: here https://github.com/lukasliebel/dfc2020_baseline.
    DFC2020_CLASS_MAPPING = torch.tensor(
        [
            0,  # maps 0s to 0
            1,  # maps 1s to 1
            1,  # maps 2s to 1
            1,  # ...
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            5,
            6,
            7,
            6,
            8,
            9,
            10,
        ]
    )

    def __init__(
        self,
        root_dir: str,
        seed: int,
        band_set: str = "all",
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for SEN12MS based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the SEN12MS Dataset classes
            seed: The seed value to use when doing the sklearn based ShuffleSplit
            band_set: The subset of S1/S2 bands to use. Options are: "all",
                "s1", "s2-all", and "s2-reduced" where the "s2-reduced" set includes:
                B2, B3, B4, B8, B11, and B12.
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()
        assert band_set in SEN12MS.BAND_SETS.keys()

        self.root_dir = root_dir
        self.seed = seed
        self.band_set = band_set
        self.band_indices = SEN12MS.BAND_SETS[band_set]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image and mask

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()

        if self.band_set == "all":
            sample["image"][:2] = sample["image"][:2].clamp(-25, 0) / -25
            sample["image"][2:] = sample["image"][2:].clamp(0, 10000) / 10000
        elif self.band_set == "s1":
            sample["image"][:2] = sample["image"][:2].clamp(-25, 0) / -25
        else:
            sample["image"][:] = sample["image"][:].clamp(0, 10000) / 10000

        if "mask" in sample:
            sample["mask"] = sample["mask"][0, :, :].long()
            sample["mask"] = torch.take(self.DFC2020_CLASS_MAPPING, sample["mask"])

        return sample

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.

        We split samples between train and val geographically with proportions of 80/20.
        This mimics the geographic test set split.

        Args:
            stage: stage to set up
        """
        season_to_int = {"winter": 0, "spring": 1000, "summer": 2000, "fall": 3000}

        self.all_train_dataset = SEN12MS(
            self.root_dir,
            split="train",
            bands=self.band_indices,
            transforms=self.preprocess,
            checksum=False,
        )

        self.all_test_dataset = SEN12MS(
            self.root_dir,
            split="test",
            bands=self.band_indices,
            transforms=self.preprocess,
            checksum=False,
        )

        # A patch is a filename like: "ROIs{num}_{season}_s2_{scene_id}_p{patch_id}.tif"
        # This patch will belong to the scene that is uniquelly identified by its
        # (season, scene_id) tuple. Because the largest scene_id is 149, we can simply
        # give each season a large number and representing a `unique_scene_id` as
        # `season_id + scene_id`.
        scenes = []
        for scene_fn in self.all_train_dataset.ids:
            parts = scene_fn.split("_")
            season_id = season_to_int[parts[1]]
            scene_id = int(parts[3])
            scenes.append(season_id + scene_id)

        train_indices, val_indices = next(
            GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=self.seed).split(
                scenes, groups=scenes
            )
        )

        self.train_dataset = Subset(self.all_train_dataset, train_indices)
        self.val_dataset = Subset(self.all_train_dataset, val_indices)
        self.test_dataset = Subset(
            self.all_test_dataset, range(len(self.all_test_dataset))
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
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
