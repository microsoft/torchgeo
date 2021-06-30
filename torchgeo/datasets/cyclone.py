import json
import os
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision.datasets.utils import check_integrity, extract_archive

from .geo import VisionDataset


class TropicalCycloneWindEstimation(VisionDataset):
    """`Tropical Cyclone Wind Estimation Competition
    <https://www.drivendata.org/competitions/72/predict-wind-speeds/>`_ Dataset.

    A collection of tropical storms in the Atlantic and East Pacific Oceans from 2000 to
    2019 with corresponding maximum sustained surface wind speed. This dataset is split
    into training and test categories for the purpose of a competition.

    If you use this dataset in your research, please cite the following paper:

    * http://doi.org/10.1109/JSTARS.2020.3011907

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub
    """

    base_folder = "cyclone"
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

        Parameters:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=True`` but ``api_key=None``, or
                ``download=False`` but dataset is missing or checksum fails
        """
        assert split in self.md5s

        self.root = root
        self.split = split
        self.transforms = transforms
        self.checksum = checksum

        if download:
            if api_key is None:
                raise RuntimeError(
                    "You must pass an MLHub API key if download=True. "
                    + "See https://www.mlhub.earth/ to register for API access."
                )
            else:
                self.download(api_key)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        output_dir = "_".join([self.collection_id, split, "source"])
        filename = os.path.join(root, self.base_folder, output_dir, "collection.json")
        with open(filename) as f:
            self.collection = json.load(f)["links"]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Parameters:
            index: index to return

        Returns:
            data, labels, field ids, and metadata at that index
        """
        source_id = os.path.split(self.collection[index]["href"])[0]
        directory = os.path.join(
            self.root,
            self.base_folder,
            "_".join([self.collection_id, self.split, "{0}"]),
            source_id.replace("source", "{0}"),
        )

        sample: Dict[str, Any] = {
            "image": self._load_image(directory),
        }
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

    def _load_image(self, directory: str) -> Tensor:
        """Load a single image.

        Parameters:
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

        Parameters:
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
        features["wind_speed"] = int(features["wind_speed"])

        return features

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        output_dir = os.path.join(self.root, self.base_folder)
        for split, resources in self.md5s.items():
            for resource_type, md5 in resources.items():
                filename = "_".join([self.collection_id, split, resource_type])
                filename = os.path.join(output_dir, filename + ".tar.gz")
                if not check_integrity(filename, md5 if self.checksum else None):
                    return False
        return True

    def _download(self, api_key: str) -> None:
        """Download the dataset and extract it.

        Parameters:
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        # Must be installed to download from MLHub
        import radiant_mlhub

        output_dir = os.path.join(self.root, self.base_folder)
        dataset = radiant_mlhub.Dataset.fetch(self.collection_id, api_key=api_key)
        dataset.download(output_dir, api_key=api_key)

        for split, resources in self.md5s.items():
            for resource_type in resources:
                filename = "_".join([self.collection_id, split, resource_type])
                filename = os.path.join(output_dir, filename) + ".tar.gz"
                extract_archive(filename, output_dir)
