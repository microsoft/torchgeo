# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""National Agriculture Imagery Program (NAIP) datamodule."""

from typing import Any, Dict, Optional, Tuple

import kornia.augmentation as K
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets import NAIP, BoundingBox, Chesapeake13, stack_samples
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from ..transforms import AugmentationSequential


class NAIPChesapeakeDataModule(LightningDataModule):
    """LightningDataModule implementation for the NAIP and Chesapeake datasets.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        patch_size: int = 256,
        stride: int = 128,
        length: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for NAIP and Chesapeake based DataLoaders.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            patch_size: size of patches to sample
            stride: stride of grid sampler
            length: epoch size
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.NAIP` (prefix keys with ``naip_``) and
                :class:`~torchgeo.datasets.Chesapeake13`
                (prefix keys with ``chesapeake_``)
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.stride = stride
        self.length = length

        self.naip_kwargs = {}
        self.chesapeake_kwargs = {}
        for key, val in kwargs.items():
            if key.startswith("naip_"):
                self.naip_kwargs[key[5:]] = val
            elif key.startswith("chesapeake_"):
                self.chesapeake_kwargs[key[11:]] = val

        self.aug = AugmentationSequential(
            K.Normalize(mean=0.0, std=255.0), data_keys=["image", "mask"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.chesapeake_kwargs.get("download", False):
            Chesapeake13(**self.chesapeake_kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: state to set up
        """
        self.chesapeake = Chesapeake13(**self.chesapeake_kwargs)
        self.naip = NAIP(**self.naip_kwargs)
        self.dataset = self.chesapeake & self.naip

        # TODO: figure out better train/val/test split
        roi = self.dataset.bounds
        midx = roi.minx + (roi.maxx - roi.minx) / 2
        midy = roi.miny + (roi.maxy - roi.miny) / 2
        train_roi = BoundingBox(roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt)
        val_roi = BoundingBox(midx, roi.maxx, roi.miny, midy, roi.mint, roi.maxt)
        test_roi = BoundingBox(roi.minx, roi.maxx, midy, roi.maxy, roi.mint, roi.maxt)

        self.train_sampler = RandomBatchGeoSampler(
            self.naip, self.patch_size, self.batch_size, self.length, train_roi
        )
        self.val_sampler = GridGeoSampler(
            self.naip, self.patch_size, self.stride, val_roi
        )
        self.test_sampler = GridGeoSampler(
            self.naip, self.patch_size, self.stride, test_roi
        )

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

    def plot(self, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, plt.Figure]:
        """Run NAIP and Chesapeake plot methods.

        See :meth:`torchgeo.datasets.NAIP.plot` and
        :meth:`torchgeo.datasets.Chesapeake.plot`.

        .. versionadded:: 0.4
        """
        image = self.naip.plot(*args, **kwargs)
        label = self.chesapeake.plot(*args, **kwargs)
        return image, label
