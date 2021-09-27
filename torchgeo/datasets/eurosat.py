# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT dataset."""

import os
from functools import lru_cache
from typing import Callable, Dict, Optional

import numpy as np
import rasterio
import torch
from torch import Tensor

from .geo import VisionDataset
from .utils import check_integrity, download_and_extract_archive


class EuroSAT(VisionDataset):
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
    class_counts = {
        "AnnualCrop": 3000,
        "Forest": 3000,
        "HerbaceousVegetation": 3000,
        "Highway": 2500,
        "Industrial": 2500,
        "Pasture": 2000,
        "PermanentCrop": 2500,
        "Residential": 3000,
        "River": 2500,
        "SeaLake": 3000,
    }
    class_name_to_label_idx = {
        "AnnualCrop": 0,
        "Forest": 1,
        "HerbaceousVegetation": 2,
        "Highway": 3,
        "Industrial": 4,
        "Pasture": 5,
        "PermanentCrop": 6,
        "Residential": 7,
        "River": 8,
        "SeaLake": 9,
    }

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new EuroSAT dataset instance.

        Args:
            root: root directory where dataset can be found
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
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        self.fns = []
        self.labels = []
        for class_name, class_count in self.class_counts.items():
            for i in range(1, class_count + 1):
                self.fns.append(
                    os.path.join(
                        self.root, self.base_dir, class_name, f"{class_name}_{i}.tif"
                    )
                )
                self.labels.append(self.class_name_to_label_idx[class_name])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: Dict[str, Tensor] = {
            "image": self._load_image(index),
            "label": torch.tensor(self.labels[index]),  # type: ignore[attr-defined]
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.labels)

    @lru_cache()
    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the image
        """
        filename = self.fns[index]
        with rasterio.open(filename) as f:
            array = f.read().astype(np.int32)
            tensor: Tensor = torch.from_numpy(array)  # type: ignore[attr-defined]
        return tensor

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.filename),
            self.md5 if self.checksum else None,
        )

        return integrity

    def _download(self) -> None:
        """Download the dataset and extract it.

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
