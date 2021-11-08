# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tropical Cyclone Wind Estimation Competition dataset."""

import json
import os
from functools import lru_cache
from typing import Any, Callable, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from .geo import VisionDataset
from .utils import check_integrity, download_radiant_mlhub_dataset, extract_archive

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class TropicalCycloneWindEstimation(VisionDataset):
    """Tropical Cyclone Wind Estimation Competition dataset.

    A collection of tropical storms in the Atlantic and East Pacific Oceans from 2000 to
    2019 with corresponding maximum sustained surface wind speed. This dataset is split
    into training and test categories for the purpose of a competition.

    See https://www.drivendata.org/competitions/72/predict-wind-speeds/ for more
    information about the competition.

    If you use this dataset in your research, please cite the following paper:

    * http://doi.org/10.1109/JSTARS.2020.3011907

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub
    """

    collection_id = "nasa_tropical_storm_competition"
    md5s = {
        "train": {
            "source": "97e913667a398704ea8d28196d91dad6",
            "labels": "97d02608b74c82ffe7496a9404a30413",
        },
        "test": {
            "source": "8d88099e4b310feb7781d776a6e1dcef",
            "labels": "d910c430f90153c1f78a99cbc08e7bd0",
        },
    }
    size = 366

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Tropical Cyclone Wind Estimation Competition Dataset.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        assert split in self.md5s

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download(api_key)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        output_dir = "_".join([self.collection_id, split, "source"])
        filename = os.path.join(root, output_dir, "collection.json")
        with open(filename) as f:
            self.collection = json.load(f)["links"]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, field ids, and metadata at that index
        """
        source_id = os.path.split(self.collection[index]["href"])[0]
        directory = os.path.join(
            self.root,
            "_".join([self.collection_id, self.split, "{0}"]),
            source_id.replace("source", "{0}"),
        )

        sample: Dict[str, Any] = {"image": self._load_image(directory)}
        sample.update(self._load_features(directory))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.collection)

    @lru_cache()
    def _load_image(self, directory: str) -> Tensor:
        """Load a single image.

        Args:
            directory: directory containing image

        Returns:
            the image
        """
        filename = os.path.join(directory.format("source"), "image.jpg")
        with Image.open(filename) as img:
            if img.height != self.size or img.width != self.size:
                img = img.resize(size=(self.size, self.size), resample=Image.BILINEAR)
            array = np.array(img)
            if len(array.shape) == 3:
                array = array[:, :, 0]
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
            return tensor

    def _load_features(self, directory: str) -> Dict[str, Any]:
        """Load features for a single image.

        Args:
            directory: directory containing image

        Returns:
            the features
        """
        filename = os.path.join(directory.format("source"), "features.json")
        with open(filename) as f:
            features: Dict[str, Any] = json.load(f)

        filename = os.path.join(directory.format("labels"), "labels.json")
        with open(filename) as f:
            features.update(json.load(f))

        features["relative_time"] = int(features["relative_time"])
        features["ocean"] = int(features["ocean"])
        features["label"] = int(features["wind_speed"])

        return features

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        for split, resources in self.md5s.items():
            for resource_type, md5 in resources.items():
                filename = "_".join([self.collection_id, split, resource_type])
                filename = os.path.join(self.root, filename + ".tar.gz")
                if not check_integrity(filename, md5 if self.checksum else None):
                    return False
        return True

    def _download(self, api_key: Optional[str] = None) -> None:
        """Download the dataset and extract it.

        Args:
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_radiant_mlhub_dataset(self.collection_id, self.root, api_key)

        for split, resources in self.md5s.items():
            for resource_type in resources:
                filename = "_".join([self.collection_id, split, resource_type])
                filename = os.path.join(self.root, filename) + ".tar.gz"
                extract_archive(filename, self.root)


class CycloneDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NASA Cyclone dataset.

    Implements 80/20 train/val splits based on hurricane storm ids.
    See :func:`setup` for more details.
    """

    def __init__(
        self,
        root_dir: str,
        seed: int,
        batch_size: int = 64,
        num_workers: int = 0,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for NASA Cyclone based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the
                TropicalCycloneWindEstimation Datasets classes
            seed: The seed value to use when doing the sklearn based GroupShuffleSplit
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            api_key: The RadiantEarth MLHub API key to use if the dataset needs to be
                downloaded
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.api_key = api_key

    def custom_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image and target

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"] / 255.0  # scale to [0,1]
        sample["image"] = (
            sample["image"].unsqueeze(0).repeat(3, 1, 1)
        )  # convert to 3 channel
        sample["label"] = torch.as_tensor(  # type: ignore[attr-defined]
            sample["label"]
        ).float()

        return sample

    def prepare_data(self) -> None:
        """Initialize the main ``Dataset`` objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """
        TropicalCycloneWindEstimation(
            self.root_dir,
            split="train",
            transforms=self.custom_transform,
            download=self.api_key is not None,
            api_key=self.api_key,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.

        We split samples between train/val by the ``storm_id`` property. I.e. all
        samples with the same ``storm_id`` value will be either in the train or the val
        split. This is important to test one type of generalizability -- given a new
        storm, can we predict its windspeed. The test set, however, contains *some*
        storms from the training set (specifically, the latter parts of the storms) as
        well as some novel storms.

        Args:
            stage: stage to set up
        """
        self.all_train_dataset = TropicalCycloneWindEstimation(
            self.root_dir,
            split="train",
            transforms=self.custom_transform,
            download=False,
        )

        self.all_test_dataset = TropicalCycloneWindEstimation(
            self.root_dir,
            split="test",
            transforms=self.custom_transform,
            download=False,
        )

        storm_ids = []
        for item in self.all_train_dataset.collection:
            storm_id = item["href"].split("/")[0].split("_")[-2]
            storm_ids.append(storm_id)

        train_indices, val_indices = next(
            GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=self.seed).split(
                storm_ids, groups=storm_ids
            )
        )

        self.train_dataset = Subset(self.all_train_dataset, train_indices)
        self.val_dataset = Subset(self.all_train_dataset, val_indices)
        self.test_dataset = Subset(
            self.all_test_dataset, range(len(self.all_test_dataset))
        )

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
