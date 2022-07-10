# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tropical Cyclone Wind Estimation Competition dataset."""

import json
import os
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, download_radiant_mlhub_dataset, extract_archive


class TropicalCycloneWindEstimation(NonGeoDataset):
    """Tropical Cyclone Wind Estimation Competition dataset.

    A collection of tropical storms in the Atlantic and East Pacific Oceans from 2000 to
    2019 with corresponding maximum sustained surface wind speed. This dataset is split
    into training and test categories for the purpose of a competition.

    See https://www.drivendata.org/competitions/72/predict-wind-speeds/ for more
    information about the competition.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/JSTARS.2020.3011907

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
                # Moved in PIL 9.1.0
                try:
                    resample = Image.Resampling.BILINEAR
                except AttributeError:
                    resample = Image.BILINEAR
                img = img.resize(size=(self.size, self.size), resample=resample)
            array: "np.typing.NDArray[np.int_]" = np.array(img)
            if len(array.shape) == 3:
                array = array[:, :, 0]
            tensor = torch.from_numpy(array)
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

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns;
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.2
        """
        image, label = sample["image"], sample["label"]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = sample["prediction"].item()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image, cmap="gray")
        ax.axis("off")

        if show_titles:
            title = f"Label: {label}"
            if showing_predictions:
                title += f"\nPrediction: {cast(str, prediction)}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
