# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NLCD dataset."""

import os
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import BoundingBox, download_url, extract_archive


class NLCD(RasterDataset):
    """National Land Cover Database (NLCD) dataset.

    The `NLCD dataset
    <https://www.usgs.gov/centers/eros/science/national-land-cover-database>`_
    is a land cover product that covers the United States and Puerto Rico. The current
    implementation supports maps for the continental United States only. The product is
    a joint effort between the United States Geological Survey
    (`USGS <https://www.usgs.gov/>`_) and the Multi-Resolution Land Characteristics
    Consortium (`MRLC <https://www.mrlc.gov/>`_) which released the first product
    in 2001 with new updates every five years since then.

    The dataset contains the following 17 classes:

    0. Background
    #. Open Water
    #. Perennial Ice/Snow
    #. Developed, Open Space
    #. Developed, Low Intensity
    #. Developed, Medium Intensity
    #. Developed, High Intensity
    #. Barren Land (Rock/Sand/Clay)
    #. Deciduous Forest
    #. Evergreen Forest
    #. Mixed Forest
    #. Shrub/Scrub
    #. Grassland/Herbaceous
    #. Pasture/Hay
    #. Cultivated Crops
    #. Woody Wetlands
    #. Emergent Herbaceous Wetlands

    Detailed descriptions of the classes can be found
    `here <https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description>`__.

    Dataset format:

    * single channel .img file with integer class labels

    If you use this dataset in your research, please use the corresponding citation:

    * 2001: https://doi.org/10.5066/P9MZGHLF
    * 2006: https://doi.org/10.5066/P9HBR9V3
    * 2011: https://doi.org/10.5066/P97S2IID
    * 2016: https://doi.org/10.5066/P96HHBIE
    * 2019: https://doi.org/10.5066/P9KZCM54

    .. versionadded:: 0.5
    """  # noqa: E501

    filename_glob = "nlcd_*_land_cover_l48_20210604.img"
    filename_regex = (
        r"nlcd_(?P<date>\d{4})_land_cover_l48_(?P<publication_date>\d{8})\.img"
    )
    zipfile_glob = "nlcd_*_land_cover_l48_20210604.zip"
    date_format = "%Y"
    is_image = False

    url = "https://s3-us-west-2.amazonaws.com/mrlc/nlcd_{}_land_cover_l48_20210604.zip"

    md5s = {
        2001: "538166a4d783204764e3df3b221fc4cd",
        2006: "67454e7874a00294adb9442374d0c309",
        2011: "ea524c835d173658eeb6fa3c8e6b917b",
        2016: "452726f6e3bd3f70d8ca2476723d238a",
        2019: "82851c3f8105763b01c83b4a9e6f3961",
    }

    ordinal_label_map = {
        0: 0,
        11: 1,
        12: 2,
        21: 3,
        22: 4,
        23: 5,
        24: 6,
        31: 7,
        41: 8,
        42: 9,
        43: 10,
        52: 11,
        71: 12,
        81: 13,
        82: 14,
        90: 15,
        95: 16,
    }

    cmap = {
        0: (0, 0, 0, 255),
        1: (70, 107, 159, 255),
        2: (209, 222, 248, 255),
        3: (222, 197, 197, 255),
        4: (217, 146, 130, 255),
        5: (235, 0, 0, 255),
        6: (171, 0, 0, 255),
        7: (179, 172, 159, 255),
        8: (104, 171, 95, 255),
        9: (28, 95, 44, 255),
        10: (181, 197, 143, 255),
        11: (204, 184, 121, 255),
        12: (223, 223, 194, 255),
        13: (220, 217, 57, 255),
        14: (171, 108, 40, 255),
        15: (184, 217, 235, 255),
        16: (108, 159, 184, 255),
    }

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        years: list[int] = [2019],
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            years: list of years for which to use nlcd layer
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
            AssertionError: if ``year`` is invalid
        """
        assert set(years).issubset(self.md5s.keys()), (
            "NLCD data product only exists for the following years: "
            f"{list(self.md5s.keys())}."
        )
        self.years = years
        self.root = root
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(root, crs, res, transforms=transforms, cache=cache)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        sample = super().__getitem__(query)

        mask = sample["mask"]
        for k, v in self.ordinal_label_map.items():
            mask[mask == k] = v

        sample["mask"] = mask

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        exists = []
        for year in self.years:
            filename_year = self.filename_glob.replace("*", str(year))
            dirname_year = filename_year.split(".")[0]
            pathname = os.path.join(self.root, dirname_year, filename_year)
            if os.path.exists(pathname):
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the zip files have already been downloaded
        exists = []
        for year in self.years:
            pathname = os.path.join(
                self.root, self.zipfile_glob.replace("*", str(year))
            )
            if os.path.exists(pathname):
                exists.append(True)
                self._extract()
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for year in self.years:
            download_url(
                self.url.format(year),
                self.root,
                md5=self.md5s[year] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        for year in self.years:
            zipfile_name = self.zipfile_glob.replace("*", str(year))
            pathname = os.path.join(self.root, zipfile_name)
            extract_archive(pathname, self.root)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        mask = sample["mask"].squeeze().numpy()
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze().numpy()
            ncols = 2

        kwargs = {
            "cmap": ListedColormap(np.array(list(self.cmap.values())) / 255),
            "vmin": 0,
            "vmax": len(self.cmap) - 1,
            "interpolation": "none",
        }

        fig, axs = plt.subplots(
            nrows=1, ncols=ncols, figsize=(ncols * 4, 4), squeeze=False
        )

        axs[0, 0].imshow(mask, **kwargs)
        axs[0, 0].axis("off")

        if show_titles:
            axs[0, 0].set_title("Mask")

        if showing_predictions:
            axs[0, 1].imshow(pred, **kwargs)
            axs[0, 1].axis("off")
            if show_titles:
                axs[0, 1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
