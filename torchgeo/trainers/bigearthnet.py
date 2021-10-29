# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet trainer."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets import BigEarthNet
from ..datasets.utils import dataset_split
from .tasks import MultiLabelClassificationTask

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class BigEarthNetClassificationTask(MultiLabelClassificationTask):
    """LightningModule for training models on the BigEarthNet Dataset."""

    num_classes = 43


class BigEarthNetDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the BigEarthNet dataset.

    Uses the train/val/test splits from the dataset.
    """

    # (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    # min/max band statistics computed on 83k random samples
    band_mins_raw = torch.tensor(  # type: ignore[attr-defined]
        [-70.0, -72.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )
    band_maxs_raw = torch.tensor(  # type: ignore[attr-defined]
        [
            31.0,
            35.0,
            18556.0,
            20528.0,
            18903.0,
            17846.0,
            16593.0,
            16512.0,
            16394.0,
            16575.0,
            16124.0,
            16097.0,
            15336.0,
            15203.0,
        ]
    )

    # min/max band statistics computed by percentile clipping the
    # above to samples to [2, 98]
    band_mins = torch.tensor(  # type: ignore[attr-defined]
        [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    band_maxs = torch.tensor(  # type: ignore[attr-defined]
        [
            6.0,
            16.0,
            9859.0,
            12874.1,
            13160.1,
            14437.3,
            12479.0,
            12564.3,
            12282.2,
            15605.0,
            12186.0,
            9453.1,
            5896.0,
            5533.0,
        ]
    )

    def __init__(
        self,
        root_dir: str,
        bands: str = "all",
        batch_size: int = 64,
        num_workers: int = 4,
        unsupervised_mode: bool = False,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for BigEarthNet based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the BigEarthNet Dataset classes
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            unsupervised_mode: Makes the train dataloader return imagery from the train,
                val, and test sets
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.bands = bands
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unsupervised_mode = unsupervised_mode

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

        if bands == "all":
            self.mins = self.band_mins[:, None, None]
            self.maxs = self.band_maxs[:, None, None]
        elif bands == "s1":
            self.mins = self.band_mins[:2, None, None]
            self.maxs = self.band_maxs[:2, None, None]
        else:
            self.mins = self.band_mins[2:, None, None]
            self.maxs = self.band_maxs[2:, None, None]

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"].float()
        sample["image"] = (sample["image"] - self.mins) / (self.maxs - self.mins)
        sample["image"] = torch.clip(  # type: ignore[attr-defined]
            sample["image"], min=0.0, max=1.0
        )
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        BigEarthNet(self.root_dir, bands=self.bands, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        transforms = Compose([self.preprocess])

        if not self.unsupervised_mode:

            dataset = BigEarthNet(
                self.root_dir, bands=self.bands, transforms=transforms
            )
            self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
                dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
            )
        else:
            self.train_dataset = BigEarthNet(  # type: ignore[assignment]
                self.root_dir, bands=self.bands, transforms=transforms
            )
            self.val_dataset, self.test_dataset = None, None  # type: ignore[assignment]

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        if self.unsupervised_mode or self.val_split_pct == 0:
            return self.train_dataloader()
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        if self.unsupervised_mode or self.test_split_pct == 0:
            return self.train_dataloader()
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
