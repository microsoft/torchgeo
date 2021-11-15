# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT dataset."""

import os
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from .geo import VisionClassificationDataset
from .utils import check_integrity, download_url, extract_archive, rasterio_loader


class EuroSAT(VisionClassificationDataset):
    """EuroSAT dataset.

    The `EuroSAT <https://github.com/phelber/EuroSAT>`_ dataset is based on Sentinel-2
    satellite images covering 13 spectral bands and consists of 10 target classes with
    a total of 27,000 labeled and geo-referenced images.

    Dataset format:

    * rasters are 13-channel GeoTiffs
    * labels are values in the range [0,9]

    Dataset classes:

    * Industrial Buildings
    * Residential Buildings
    * Annual Crop
    * Permanent Crop
    * River
    * Sea and Lake
    * Herbaceous Vegetation
    * Highway
    * Pasture
    * Forest

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following papers:

    * https://ieeexplore.ieee.org/document/8736785
    * https://ieeexplore.ieee.org/document/8519248
    """

    url = "http://madm.dfki.de/files/sentinel/EuroSATallBands.zip"  # 2.0 GB download
    filename = "EuroSATallBands.zip"
    md5 = "5ac12b3b2557aa56e1826e981e8e200e"

    # For some reason the class directories are actually nested in this directory
    base_dir = os.path.join(
        "ds", "images", "remote_sensing", "otherDatasets", "sentinel_2", "tif"
    )

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/eurosat-train.txt",  # noqa: E501
        "val": "https://storage.googleapis.com/remote_sensing_representations/eurosat-val.txt",  # noqa: E501
        "test": "https://storage.googleapis.com/remote_sensing_representations/eurosat-test.txt",  # noqa: E501
    }
    split_md5s = {
        "train": "908f142e73d6acdf3f482c5e80d851b1",
        "val": "95de90f2aa998f70a3b2416bfe0687b4",
        "test": "7ae5ab94471417b6e315763121e67c5f",
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new EuroSAT dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self._verify()

        valid_fns = set()
        with open(os.path.join(self.root, f"eurosat-{split}.txt"), "r") as f:
            for fn in f:
                valid_fns.add(fn.strip().replace(".jpg", ".tif"))
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.base_dir),
            transforms=transforms,
            loader=rasterio_loader,
            is_valid_file=is_in_split,
        )

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.filename), self.md5 if self.checksum else None
        )
        return integrity

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        filepath = os.path.join(self.root, self.base_dir)
        if os.path.exists(filepath):
            return

        # Check if zip file already exists (if so then extract)
        if self._check_integrity():
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
                filename=f"eurosat-{split}.txt",
                md5=self.split_md5s[split] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)


class EuroSATDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the EuroSAT dataset.

    Uses the train/val/test splits from the dataset.
    """

    band_means = torch.tensor(  # type: ignore[attr-defined]
        [
            1354.40546513,
            1118.24399958,
            1042.92983953,
            947.62620298,
            1199.47283961,
            1999.79090914,
            2369.22292565,
            2296.82608323,
            732.08340178,
            12.11327804,
            1819.01027855,
            1118.92391149,
            2594.14080798,
        ]
    )

    band_stds = torch.tensor(  # type: ignore[attr-defined]
        [
            245.71762908,
            333.00778264,
            395.09249139,
            593.75055589,
            566.4170017,
            861.18399006,
            1086.63139075,
            1117.98170791,
            404.91978886,
            4.77584468,
            1002.58768311,
            761.30323499,
            1231.58581042,
        ]
    )

    def __init__(
        self, root_dir: str, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a LightningDataModule for EuroSAT based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the EuroSAT Dataset classes
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
        sample["image"] = self.norm(sample["image"])
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        EuroSAT(self.root_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        self.train_dataset = EuroSAT(self.root_dir, "train", transforms=transforms)
        self.val_dataset = EuroSAT(self.root_dir, "val", transforms=transforms)
        self.test_dataset = EuroSAT(self.root_dir, "test", transforms=transforms)

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
