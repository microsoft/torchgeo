# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""UC Merced dataset."""
import os
from typing import Any, Callable, Dict, Optional

import pytorch_lightning as pl
import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from .geo import VisionClassificationDataset
from .utils import check_integrity, download_url, extract_archive

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class UCMerced(VisionClassificationDataset):
    """UC Merced dataset.

    The `UC Merced <http://weegee.vision.ucmerced.edu/datasets/landuse.html>`_
    dataset is a land use classification dataset of 2.1k 256x256 1ft resolution RGB
    images of urban locations around the U.S. extracted from the USGS National Map Urban
    Area Imagery collection with 21 land use classes (100 images per class).

    Dataset features:

    * land use class labels from around the U.S.
    * three spectral bands - RGB
    * 21 classes

    Dataset classes:

    * agricultural
    * airplane
    * baseballdiamond
    * beach
    * buildings
    * chaparral
    * denseresidential
    * forest
    * freeway
    * golfcourse
    * harbor
    * intersection
    * mediumresidential
    * mobilehomepark
    * overpass
    * parkinglot
    * river
    * runway
    * sparseresidential
    * storagetanks
    * tenniscourt

    This dataset uses the train/val/test splits defined in the "In-domain representation
    learning for remote sensing" paper:

    * https://arxiv.org/abs/1911.06721

    If you use this dataset in your research, please cite the following paper:

    * https://dl.acm.org/doi/10.1145/1869790.1869829
    """

    url = "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"  # 318 MB
    filename = "UCMerced_LandUse.zip"
    md5 = "5b7ec56793786b6dc8a908e8854ac0e4"

    base_dir = os.path.join("UCMerced_LandUse", "Images")
    classes = [
        "agricultural",
        "airplane",
        "baseballdiamond",
        "beach",
        "buildings",
        "chaparral",
        "denseresidential",
        "forest",
        "freeway",
        "golfcourse",
        "harbor",
        "intersection",
        "mediumresidential",
        "mobilehomepark",
        "overpass",
        "parkinglot",
        "river",
        "runway",
        "sparseresidential",
        "storagetanks",
        "tenniscourt",
    ]

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/uc_merced-train.txt",  # noqa: E501
        "val": "https://storage.googleapis.com/remote_sensing_representations/uc_merced-val.txt",  # noqa: E501
        "test": "https://storage.googleapis.com/remote_sensing_representations/uc_merced-test.txt",  # noqa: E501
    }
    split_md5s = {
        "train": "f2fb12eb2210cfb53f93f063a35ff374",
        "val": "11ecabfc52782e5ea6a9c7c0d263aca0",
        "test": "046aff88472d8fc07c4678d03749e28d",
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new UC Merced dataset instance.

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
        assert split in self.splits
        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self._verify()

        valid_fns = set()
        with open(os.path.join(self.root, f"uc_merced-{split}.txt"), "r") as f:
            for fn in f:
                valid_fns.add(fn.strip())
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.base_dir),
            transforms=transforms,
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
                filename=f"uc_merced-{split}.txt",
                md5=self.split_md5s[split] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)


class UCMercedDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the UC Merced dataset.

    Uses random train/val/test splits.
    """

    band_means = torch.tensor([0, 0, 0])  # type: ignore[attr-defined]

    band_stds = torch.tensor([1, 1, 1])  # type: ignore[attr-defined]

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for UCMerced based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the UCMerced Dataset classes
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
            sample: dictionary containing image

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        c, h, w = sample["image"].shape
        if h != 256 or w != 256:
            sample["image"] = torchvision.transforms.functional.resize(
                sample["image"], size=(256, 256)
            )
        sample["image"] = self.norm(sample["image"])
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        UCMerced(self.root_dir, download=False, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        self.train_dataset = UCMerced(self.root_dir, "train", transforms=transforms)
        self.val_dataset = UCMerced(self.root_dir, "val", transforms=transforms)
        self.test_dataset = UCMerced(self.root_dir, "test", transforms=transforms)

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
