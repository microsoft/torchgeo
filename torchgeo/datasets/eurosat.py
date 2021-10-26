# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT dataset."""

import os
from typing import Callable, Dict, Optional

from torch import Tensor

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
        self.download = download
        self.checksum = checksum
        self._verify()
        super().__init__(
            root=os.path.join(root, self.base_dir),
            transforms=transforms,
            loader=rasterio_loader,
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

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)
