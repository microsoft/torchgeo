# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""South Africa Crop Type Competition Dataset"""

import os
import re
import glob
from collections.abc import Iterable
from typing import Any, Callable, Optional, Union, cast

import matplotlib.pyplot as plt
import rasterio
from rasterio.crs import CRS
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import RasterDataset
from .utils import (
    BoundingBox,
    DatasetNotFoundError,
    RGBBandsMissingError,
    check_integrity,
    download_url,
)


class SouthAfricaCropType(RasterDataset):
    """South Africa Crop Type Challenge dataset.

    The `South Africa Crop Type Challenge
    <https://beta.source.coop/repositories/radiantearth/south-africa-crops-competition/description/>`__
    dataset includes satellite imagery from Sentinel-1 and Sentinel-2 and labels for
    crop type that were collected by aerial
    and vehicle survey from May 2017 to March 2018. Data was collected by the
    provided by the Western Cape Department of Agriculture and is
    is available via the Radiant Earth Foundation.
    Each chip is matched with a label.
    Each pixel in the label contains an integer field number and crop type class.

    Dataset format:

    * images are 2-band Sentinel 1 and 12-bands Sentinel-2 data with a cloud mask
    * masks are tiff image with unique values representing the class and field id.

    Dataset classes:

    0: No Data
    1: Lucerne/Medics
    2: Planted pastures (perennial)
    3: Fallow
    4: Wine grapes
    5: Weeds
    6: Small grain grazing
    7: Wheat
    8: Canola
    9: Rooibos

    If you use this dataset in your research, please cite the following dataset:
    Western Cape Department of Agriculture, Radiant Earth Foundation (2021)
    "Crop Type Classification Dataset for Western Cape, South Africa", 
    Version 1.0, Radiant MLHub, https://doi.org/10.34911/rdnt.j0co8q

    .. versionadded:: 0.6
    """

    url = "https://beta.source.coop/repositories/radiantearth/south-africa-crops-competition/download/"

    #1_2017_04_01_B01_10m.tif
    filename_regex = r"""
        ^(?P<field_id>[0-9]*)_(?P<date>[0-9_]*)_(?P<band>[0-9A-Z]*)+_10m\.tif"""
    date_format = "%Y_%m_%d"
    separate_files = True

    rgb_bands = ["B04", "B03", "B02"]
    all_bands = (
        "VH",
        "VV"
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    )

    cmap = {
        0: (0, 0, 0, 255),
        1: (255, 211, 0, 255),
        2: (255, 37, 37, 255),
        3: (0, 168, 226, 255),
        4: (255, 158, 9, 255),
        5: (37, 111, 0, 255),
        6: (255, 255, 0, 255),
        8: (111, 166, 0, 255),
        9: (0, 175, 73, 255)
    }

    def __init__(
        self,
        root: str = "data",
        crs: CRS = CRS.from_epsg(4326),
        classes: list[int] = list(cmap.keys()),
        bands: tuple[str, ...] = all_bands,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        split: str = "train",
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new South Africa dataset instance.

        Args:
            root: root directory where dataset can be found
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            split: portion of dataset to load
        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert (
            set(classes) <= self.cmap.keys()
        ), f"Only the following classes are valid: {list(self.cmap.keys())}."
        assert 0 in classes, "Classes must include the background class: 0"

        self.root = root
        self.classes = classes
        self.checksum = checksum
        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=self.dtype)
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)

        # not downloading for now
        # if download:
        #     self._download()

        # not checking integrity for now
        # if not self._check_integrity():
        #     raise DatasetNotFoundError(self)

        super().__init__(
            paths=root,
            crs=crs,
            bands=bands,
            transforms=transforms,
            cache=cache
        )

        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])


# download and checksum verification not implemented yet
# def _download(self) -> None:
#     """Download the dataset and extract it.

#     Raises:
#         RuntimeError: if download doesn't work correctly or checksums don't match
#     """
#     # not checking the integrity because no compressed files in the dataset
#     if self._check_integrity():
#         print("Files already downloaded and verified")
#         return
#     # make the data not auto-downloadable until azure-storage-blob is integrated

# def _check_integrity(self) -> bool:
#     """Check integrity of dataset.

#     Returns:
#         True if dataset files are found and/or MD5s match, else False
#     """
