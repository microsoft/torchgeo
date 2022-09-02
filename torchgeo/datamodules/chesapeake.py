# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Chesapeake Bay High-Resolution Land Cover Project datamodule."""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets import ChesapeakeCVPR, stack_samples
from ..samplers.batch import RandomBatchGeoSampler
from ..samplers.single import GridGeoSampler

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class ChesapeakeCVPRDataModule(LightningDataModule):
    """LightningDataModule implementation for the Chesapeake CVPR Land Cover dataset.

    Uses the random splits defined per state to partition tiles into train, val,
    and test sets.
    """

    def __init__(
        self,
        root_dir: str,
        train_splits: List[str],
        val_splits: List[str],
        test_splits: List[str],
        patches_per_tile: int = 200,
        patch_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 0,
        class_set: int = 7,
        use_prior_labels: bool = False,
        prior_smoothing_constant: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for Chesapeake CVPR based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the ChesapeakeCVPR Dataset
                classes
            train_splits: The splits used to train the model, e.g. ["ny-train"]
            val_splits: The splits used to validate the model, e.g. ["ny-val"]
            test_splits: The splits used to test the model, e.g. ["ny-test"]
            patches_per_tile: The number of patches per tile to sample
            patch_size: The size of each patch in pixels (test patches will be 1.5 times
                this size)
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            class_set: The high-resolution land cover class set to use - 5 or 7
            use_prior_labels: Flag for using a prior over high-resolution classes
                instead of the high-resolution labels themselves
            prior_smoothing_constant: additive smoothing to add when using prior labels

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

        self.root_dir = root_dir
        self.train_splits = train_splits
        self.val_splits = val_splits
        self.test_splits = test_splits
        self.patches_per_tile = patches_per_tile
        self.patch_size = patch_size
        # This is a rough estimate of how large of a patch we will need to sample in
        # EPSG:3857 in order to guarantee a large enough patch in the local CRS.
        self.original_patch_size = patch_size * 2
        self.batch_size = batch_size
        self.num_workers = num_workers
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

    def pad_to(
        self, size: int = 512, image_value: int = 0, mask_value: int = 0
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a padding transform on a single sample.

        Args:
            size: output image size
            image_value: value to pad image with
            mask_value: value to pad mask with

        Returns:
            function to perform padding
        """

        def pad_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape
            assert height <= size and width <= size

            height_pad = size - height
            width_pad = size - width

            # See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
            # for a description of the format of the padding tuple
            sample["image"] = F.pad(
                sample["image"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=image_value,
            )
            sample["mask"] = F.pad(
                sample["mask"],
                (0, width_pad, 0, height_pad),
                mode="constant",
                value=mask_value,
            )
            return sample

        return pad_inner

    def center_crop(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to perform a center crop transform on a single sample.

        Args:
            size: output image size

        Returns:
            function to perform center crop
        """

        def center_crop_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            _, height, width = sample["image"].shape

            y1 = round((height - size) / 2)
            x1 = round((width - size) / 2)
            sample["image"] = sample["image"][:, y1 : y1 + size, x1 : x1 + size]
            sample["mask"] = sample["mask"][:, y1 : y1 + size, x1 : x1 + size]

            return sample

        return center_crop_inner

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocesses a single sample.

        Args:
            sample: sample dictionary containing image and mask

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0

        if "mask" in sample:
            sample["mask"] = sample["mask"].squeeze()
            if self.use_prior_labels:
                sample["mask"] = F.normalize(sample["mask"].float(), p=1, dim=0)
                sample["mask"] = F.normalize(
                    sample["mask"] + self.prior_smoothing_constant, p=1, dim=0
                )
            else:
                if self.class_set == 5:
                    sample["mask"][sample["mask"] == 5] = 4
                    sample["mask"][sample["mask"] == 6] = 4
                sample["mask"] = sample["mask"].long()

        return sample

    def remove_bbox(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Removes the bounding box property from a sample.

        Args:
            sample: dictionary with geographic metadata

        Returns
            sample without the bbox property
        """
        del sample["bbox"]
        return sample

    def nodata_check(
        self, size: int = 512
    ) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
        """Returns a function to check for nodata or mis-sized input.

        Args:
            size: output image size

        Returns:
            function to check for nodata values
        """

        def nodata_check_inner(sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
            num_channels, height, width = sample["image"].shape

            if height < size or width < size:
                sample["image"] = torch.zeros((num_channels, size, size))
                sample["mask"] = torch.zeros((size, size))

            return sample

        return nodata_check_inner

    def prepare_data(self) -> None:
        """Confirms that the dataset is downloaded on the local node.

        This method is called once per node, while :func:`setup` is called once per GPU.
        """
        ChesapeakeCVPR(
            self.root_dir,
            splits=self.train_splits,
            layers=self.layers,
            transforms=None,
            download=False,
            checksum=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.

        Args:
            stage: stage to set up
        """
        train_transforms = Compose(
            [
                self.center_crop(self.patch_size),
                self.nodata_check(self.patch_size),
                self.preprocess,
                self.remove_bbox,
            ]
        )
        val_transforms = Compose(
            [
                self.center_crop(self.patch_size),
                self.nodata_check(self.patch_size),
                self.preprocess,
                self.remove_bbox,
            ]
        )
        test_transforms = Compose(
            [
                self.pad_to(self.original_patch_size, image_value=0, mask_value=0),
                self.preprocess,
                self.remove_bbox,
            ]
        )

        self.train_dataset = ChesapeakeCVPR(
            self.root_dir,
            splits=self.train_splits,
            layers=self.layers,
            transforms=train_transforms,
            download=False,
            checksum=False,
        )
        self.val_dataset = ChesapeakeCVPR(
            self.root_dir,
            splits=self.val_splits,
            layers=self.layers,
            transforms=val_transforms,
            download=False,
            checksum=False,
        )
        self.test_dataset = ChesapeakeCVPR(
            self.root_dir,
            splits=self.test_splits,
            layers=self.layers,
            transforms=test_transforms,
            download=False,
            checksum=False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        sampler = RandomBatchGeoSampler(
            self.train_dataset,
            size=self.original_patch_size,
            batch_size=self.batch_size,
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
            batch_size=self.batch_size,
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
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )
