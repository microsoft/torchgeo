# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Chesapeake Bay High-Resolution Land Cover Project datamodule."""

from typing import Any, List

import kornia.augmentation as K

from ..datasets import ChesapeakeCVPR
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from ..transforms import AugmentationSequential
from .geo import GeoDataModule


class ChesapeakeCVPRDataModule(GeoDataModule):
    """LightningDataModule implementation for the Chesapeake CVPR Land Cover dataset.

    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """

    def __init__(
        self,
        train_splits: List[str],
        val_splits: List[str],
        test_splits: List[str],
        num_tiles_per_batch: int = 64,
        num_patches_per_tile: int = 200,
        patch_size: int = 256,
        length: int = 1000,
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
            num_tiles_per_batch: Number of image tiles to sample from during training.
            num_patches_per_tile: Number of patches to randomly sample from each image
                tile during training
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
        super().__init__(ChesapeakeCVPR, 1, patch_size, length, num_workers, **kwargs)

        for state in train_splits + val_splits + test_splits:
            assert state in ChesapeakeCVPR.splits
        assert class_set in [5, 7]
        if use_prior_labels and class_set != 5:
            raise ValueError(
                "The pre-generated prior labels are only valid for the 5"
                + " class set of labels"
            )

        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits
        self.train_batch_size = num_tiles_per_batch
        self.num_patches_per_tile = num_patches_per_tile
        # This is a rough estimate of how large of a patch we will need to sample in
        # EPSG:3857 in order to guarantee a large enough patch in the local CRS.
        self.original_patch_size = patch_size * 2
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
            K.CenterCrop(patch_size),
            K.Normalize(mean=0.0, std=255.0),
            data_keys=["image", "mask"],
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
                self.train_batch_size or 1,
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
