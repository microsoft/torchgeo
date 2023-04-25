# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SEN12MS datamodule."""

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Subset

from ..datasets import SEN12MS
from .geo import NonGeoDataModule
from .utils import group_shuffle_split


class SEN12MSDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the SEN12MS dataset.

    Implements 80/20 geographic train/val splits and uses the test split from the
    classification dataset definitions.

    Uses the Simplified IGBP scheme defined in the 2020 Data Fusion Competition. See
    https://arxiv.org/abs/2002.08254.
    """

    #: Mapping from the IGBP class definitions to the DFC2020, taken from the dataloader
    #: here: https://github.com/lukasliebel/dfc2020_baseline.
    DFC2020_CLASS_MAPPING = torch.tensor(
        [0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10]
    )

    std = torch.tensor(
        [-25, -25, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4]
    )

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        band_set: str = "all",
        **kwargs: Any,
    ) -> None:
        """Initialize a new SEN12MSDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            band_set: Subset of S1/S2 bands to use. Options are: "all",
                "s1", "s2-all", and "s2-reduced" where the "s2-reduced" set includes:
                B2, B3, B4, B8, B11, and B12.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SEN12MS`.
        """
        kwargs["bands"] = SEN12MS.BAND_SETS[band_set]

        if band_set == "s1":
            self.std = self.std[:2]
        elif band_set == "s2-all":
            self.std = self.std[2:]
        elif band_set == "s2-reduced":
            self.std = self.std[torch.tensor([3, 4, 5, 9, 12, 13])]

        super().__init__(SEN12MS, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            season_to_int = {"winter": 0, "spring": 1000, "summer": 2000, "fall": 3000}

            self.dataset = SEN12MS(split="train", **self.kwargs)

            # A patch is a filename like:
            #     "ROIs{num}_{season}_s2_{scene_id}_p{patch_id}.tif"
            # This patch will belong to the scene that is uniquely identified by its
            # (season, scene_id) tuple. Because the largest scene_id is 149, we can
            # simply give each season a large number and representing a unique_scene_id
            # as (season_id + scene_id).
            scenes = []
            for scene_fn in self.dataset.ids:
                parts = scene_fn.split("_")
                season_id = season_to_int[parts[1]]
                scene_id = int(parts[3])
                scenes.append(season_id + scene_id)

            train_indices, val_indices = group_shuffle_split(
                scenes, test_size=0.2, random_state=0
            )

            self.train_dataset = Subset(self.dataset, train_indices)
            self.val_dataset = Subset(self.dataset, val_indices)
        if stage in ["test"]:
            self.test_dataset = SEN12MS(split="test", **self.kwargs)

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
        batch["mask"] = torch.take(self.DFC2020_CLASS_MAPPING, batch["mask"])

        return super().on_after_batch_transfer(batch, dataloader_idx)
