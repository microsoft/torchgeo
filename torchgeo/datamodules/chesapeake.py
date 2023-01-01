# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Chesapeake Bay High-Resolution Land Cover Project datamodule."""

from typing import Any, Dict, List, Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets import ChesapeakeCVPR, stack_samples
from ..samplers.batch import RandomBatchGeoSampler
from ..samplers.single import GridGeoSampler
from ..transforms import AugmentationSequential


class ChesapeakeCVPRDataModule(LightningDataModule):
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
        num_workers: int = 0,
        class_set: int = 7,
        use_prior_labels: bool = False,
        prior_smoothing_constant: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """Initialize a new LightningDataModule instance.

        Args:
            train_splits: The splits used to train the model, e.g. ["ny-train"]
            val_splits: The splits used to validate the model, e.g. ["ny-val"]
            test_splits: The splits used to test the model, e.g. ["ny-test"]
            num_tiles_per_batch: The number of image tiles to sample from during
                training
            num_patches_per_tile: The number of patches to randomly sample from each
                image tile during training
            patch_size: The size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures
            num_workers: The number of workers to use in all created DataLoaders
            class_set: The high-resolution land cover class set to use - 5 or 7
            use_prior_labels: Flag for using a prior over high-resolution classes
                instead of the high-resolution labels themselves
            prior_smoothing_constant: additive smoothing to add when using prior labels
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ChesapeakeCVPR`

        Raises:
            ValueError: if ``use_prior_labels`` is used with ``class_set==7``
        """
        super().__init__()
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
        self.patch_size = patch_size
        # This is a rough estimate of how large of a patch we will need to sample in
        # EPSG:3857 in order to guarantee a large enough patch in the local CRS.
        self.original_patch_size = patch_size * 2
        self.num_workers = num_workers
        self.class_set = class_set
        self.use_prior_labels = use_prior_labels
        self.prior_smoothing_constant = prior_smoothing_constant
        self.kwargs = kwargs

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

    def prepare_data(self) -> None:
        """Confirms that the dataset is downloaded on the local node.

        This method is called once per node, while :func:`setup` is called once per GPU.
        """
        if self.kwargs.get("download", False):
            ChesapeakeCVPR(splits=self.train_splits, layers=self.layers, **self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.train_dataset = ChesapeakeCVPR(
            splits=self.train_splits, layers=self.layers, **self.kwargs
        )
        self.val_dataset = ChesapeakeCVPR(
            splits=self.val_splits, layers=self.layers, **self.kwargs
        )
        self.test_dataset = ChesapeakeCVPR(
            splits=self.test_splits, layers=self.layers, **self.kwargs
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        sampler = RandomBatchGeoSampler(
            self.train_dataset,
            size=self.original_patch_size,
            batch_size=self.train_batch_size,
            length=self.patches_per_tile * len(self.train_dataset),
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        sampler = GridGeoSampler(
            self.val_dataset,
            size=self.original_patch_size,
            stride=self.original_patch_size,
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        sampler = GridGeoSampler(
            self.test_dataset,
            size=self.original_patch_size,
            stride=self.original_patch_size,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=self.train_batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
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
        batch = self.aug(batch)
        return batch

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.ChesapeakeCVPR.plot`.

        .. versionadded:: 0.4
        """
        return self.test_dataset.plot(*args, **kwargs)
