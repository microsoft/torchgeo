# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Chesapeake Bay High-Resolution Land Cover Project datamodule."""

from typing import Any, Optional

import kornia.augmentation as K
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ..datasets import ChesapeakeCVPR
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from ..transforms import AugmentationSequential
from .geo import GeoDataModule


class _Transform(nn.Module):
    """Version of AugmentationSequential designed for samples, not batches."""

    def __init__(self, aug: nn.Module) -> None:
        """Initialize a new _Transform instance.

        Args:
            aug: Augmentation to apply.
        """
        super().__init__()
        self.aug = aug

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply the augmentation.

        Args:
            sample: Input sample.

        Returns:
            Augmented sample.
        """
        for key in ["image", "mask"]:
            dtype = sample[key].dtype
            # All inputs must be float
            sample[key] = sample[key].float()
            sample[key] = self.aug(sample[key])
            sample[key] = sample[key].to(dtype)
            # Kornia adds batch dimension
            sample[key] = rearrange(sample[key], "() c h w -> c h w")
        return sample


class ChesapeakeCVPRDataModule(GeoDataModule):
    """LightningDataModule implementation for the Chesapeake CVPR Land Cover dataset.

    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """

    def __init__(
        self,
        train_splits: list[str],
        val_splits: list[str],
        test_splits: list[str],
        batch_size: int = 64,
        patch_size: int = 256,
        length: Optional[int] = None,
        num_workers: int = 0,
        class_set: int = 7,
        use_prior_labels: bool = False,
        prior_smoothing_constant: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """Initialize a new ChesapeakeCVPRDataModule instance.

        Args:
            train_splits: Splits used to train the model, e.g., ["ny-train"].
            val_splits: Splits used to validate the model, e.g., ["ny-val"].
            test_splits: Splits used to test the model, e.g., ["ny-test"].
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            class_set: The high-resolution land cover class set to use (5 or 7).
            use_prior_labels: Flag for using a prior over high-resolution classes
                instead of the high-resolution labels themselves.
            prior_smoothing_constant: Additive smoothing to add when using prior labels.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ChesapeakeCVPR`.

        Raises:
            ValueError: If ``use_prior_labels=True`` is used with ``class_set=7``.
        """
        # This is a rough estimate of how large of a patch we will need to sample in
        # EPSG:3857 in order to guarantee a large enough patch in the local CRS.
        self.original_patch_size = patch_size * 3
        kwargs["transforms"] = _Transform(K.CenterCrop(patch_size))

        super().__init__(
            ChesapeakeCVPR, batch_size, patch_size, length, num_workers, **kwargs
        )

        assert class_set in [5, 7]
        if use_prior_labels and class_set == 7:
            raise ValueError(
                "The pre-generated prior labels are only valid for the 5"
                + " class set of labels"
            )

        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits
        self.class_set = class_set
        self.use_prior_labels = use_prior_labels
        self.prior_smoothing_constant = prior_smoothing_constant

        if self.use_prior_labels:
            self.layers = [
                "naip-new",
                "prior_from_cooccurrences_101_31_no_osm_no_buildings",
            ]
        else:
            self.layers = ["naip-new", "lc"]

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        )

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            self.train_dataset = ChesapeakeCVPR(
                splits=self.train_splits, layers=self.layers, **self.kwargs
            )
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset,
                self.original_patch_size,
                self.batch_size,
                self.length,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = ChesapeakeCVPR(
                splits=self.val_splits, layers=self.layers, **self.kwargs
            )
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.original_patch_size, self.original_patch_size
            )
        if stage in ["test"]:
            self.test_dataset = ChesapeakeCVPR(
                splits=self.test_splits, layers=self.layers, **self.kwargs
            )
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.original_patch_size, self.original_patch_size
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
        if self.use_prior_labels:
            batch["mask"] = F.normalize(batch["mask"].float(), p=1, dim=1)
            batch["mask"] = F.normalize(
                batch["mask"] + self.prior_smoothing_constant, p=1, dim=1
            ).long()
        else:
            if self.class_set == 5:
                batch["mask"][batch["mask"] == 5] = 4
                batch["mask"][batch["mask"] == 6] = 4

        return super().on_after_batch_transfer(batch, dataloader_idx)
