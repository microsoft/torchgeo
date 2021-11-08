# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 dataset."""

import os
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from .geo import VisionClassificationDataset
from .utils import download_url, extract_archive

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class RESISC45(VisionClassificationDataset):
    """RESISC45 dataset.

    The `RESISC45 <http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html>`_
    dataset is a dataset for remote sensing image scene classification.

    Dataset features:

    * 31,500 images with 0.2-30 m per pixel resolution (256x256 px)
    * three spectral bands - RGB
    * 45 scene classes, 700 images per class
    * images extracted from Google Earth from over 100 countries
    * images conditions with high variability (resolution, weather, illumination)

    Dataset format:

    * images are three-channel jpgs

    Dataset classes:

    0. airplane
    1. airport
    2. baseball_diamond
    3. basketball_court
    4. beach
    5. bridge
    6. chaparral
    7. church
    8. circular_farmland
    9. cloud
    10. commercial_area
    11. dense_residential
    12. desert
    13. forest
    14. freeway
    15. golf_course
    16. ground_track_field
    17. harbor
    18. industrial_area
    19. intersection
    20. island
    21. lake
    22. meadow
    23. medium_residential
    24. mobile_home_park
    25. mountain
    26. overpass
    27. palace
    28. parking_lot
    29. railway
    30. railway_station
    31. rectangular_farmland
    32. river
    33. roundabout
    34. runway
    35. sea_ice
    36. ship
    37. snowberg
    38. sparse_residential
    39. stadium
    40. storage_tank
    41. tennis_court
    42. terrace
    43. thermal_power_station
    44. wetland

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/jproc.2017.2675998

    """

    url = "https://drive.google.com/file/d/1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv"
    md5 = "d824acb73957502b00efd559fc6cfbbb"
    filename = "NWPU-RESISC45.rar"
    directory = "NWPU-RESISC45"

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt",  # noqa: E501
        "val": "https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt",  # noqa: E501
        "test": "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt",  # noqa: E501
    }
    split_md5s = {
        "train": "b5a4c05a37de15e4ca886696a85c403e",
        "val": "a0770cee4c5ca20b8c32bbd61e114805",
        "test": "3dda9e4988b47eb1de9f07993653eb08",
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new RESISC45 dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in self.splits
        self.root = root
        self.download = download
        self.checksum = checksum
        self._verify()

        valid_fns = set()
        with open(os.path.join(self.root, f"resisc45-{split}.txt"), "r") as f:
            for fn in f:
                valid_fns.add(fn.strip())
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.directory),
            transforms=transforms,
            is_valid_file=is_in_split,
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        filepath = os.path.join(self.root, self.directory)
        if os.path.exists(filepath):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automaticaly download the dataset."
            )

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
        for split in self.splits:
            download_url(
                self.split_urls[split],
                self.root,
                filename=f"resisc45-{split}.txt",
                md5=self.split_md5s[split] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)


class RESISC45DataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the RESISC45 dataset.

    Uses the train/val/test splits from the dataset.
    """

    band_means = torch.tensor(  # type: ignore[attr-defined]
        [0.36801773, 0.38097873, 0.343583]
    )

    band_stds = torch.tensor(  # type: ignore[attr-defined]
        [0.14540215, 0.13558227, 0.13203649]
    )

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for RESISC45 based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the RESISC45 Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.norm = Normalize(self.band_means, self.band_stds)

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        sample["image"] = self.norm(sample["image"])
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        RESISC45(self.root_dir, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        self.train_dataset = RESISC45(self.root_dir, "train", transforms=transforms)
        self.val_dataset = RESISC45(self.root_dir, "val", transforms=transforms)
        self.test_dataset = RESISC45(self.root_dir, "test", transforms=transforms)

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
