# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NLCD dataset."""

import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import download_url, extract_archive


class NLCD(RasterDataset):
    """National Land Cover Database (NLCD) dataset.

    The `NLCD dataset
    <https://www.usgs.gov/centers/eros/science/national-land-cover-database>`_
    is a land cover product that covers the United States and Puerto Rico. It is
    a joint effort between the United States Geological Survey
    (`USGS <https://www.usgs.gov/>`) and the Multi-Resolution Land Characteristics
    Consortium (`MRLC <https://www.mrlc.gov/>`_) which released the first product
    in 2001 with new updates every five years since then.

    The dataset contains the following 15 classes:

    * Open Water
    * Pereniial Ice/Snow
    * Developed, Open Space
    * Developed, Low Intensity
    * Developed, Medium Intensity
    * Developed, High Intensity
    * Barren Land (Rock/Sand/Clay)
    * Deciduous Forest
    * Evergreen Forest
    * Mixed Forest
    * Shruf/Scrub
    * Grassland/Herbaceous
    * Pasture/Hay
    * Cultivated Crops
    * Woody Wetlands
    * Emergent Herbaceous Wetlands

    With these additional classes available uniquely to Alaska:

    * Dwarf Scrub
    * Sedge/Herbaceous
    * Lichens
    * Moss

    Detailed descriptions of the classes can be found
    `here <https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description>`_.

    Dataset format:

    * single channel geotiff file with integer class labels

    If you use this dataset in your research, please cite:

    *

    .. versionadded:: 0.5
    """  # noqa: E501

    filename_glob = "nlcd_*_land_cover_l48_20210604.img"
    filename_regex = (
        r"""nlcd_(?P<date>\d{4})_land_cover_l48_(?P<publication_date>\d{8})\.img"""
    )
    zipfile_glob = "nlcd_*_land_cover_l48_20210604.zip"
    date_format = "%Y"
    is_image = False

    urls = {
        2001: "https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2001_land_cover_l48_20210604.zip",  # noqa: E501
        2006: "https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2006_land_cover_l48_20210604.zip",  # noqa: E501
        2011: "https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2011_land_cover_l48_20210604.zip",  # noqa: E501
        2016: "https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2016_land_cover_l48_20210604.zip",  # noqa: E501
        2019: "https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2019_land_cover_l48_20210604.zip",  # noqa: E501
    }

    md5s = {
        2001: "538166a4d783204764e3df3b221fc4cd",
        2006: "67454e7874a00294adb9442374d0c309",
        2011: "ea524c835d173658eeb6fa3c8e6b917b",
        2016: "452726f6e3bd3f70d8ca2476723d238a",
        2019: "82851c3f8105763b01c83b4a9e6f3961",
    }

    cmap = {
        0: (0, 0, 0, 255),
        11: (70, 107, 159, 255),
        12: (209, 222, 248, 255),
        21: (222, 197, 197, 255),
        22: (217, 146, 130, 255),
        23: (235, 0, 0, 255),
        24: (171, 0, 0, 255),
        31: (179, 172, 159, 255),
        41: (104, 171, 95, 255),
        42: (28, 95, 44, 255),
        43: (181, 197, 143, 255),
        52: (204, 184, 121, 255),
        71: (223, 223, 194, 255),
        81: (220, 217, 57, 255),
        82: (171, 108, 40, 255),
        90: (184, 217, 235, 255),
        95: (108, 159, 184, 255),
    }

    valid_years = [2001, 2006, 2011, 2016, 2019]

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        year: int = 2019,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
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
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
            RuntimeError: if ``year`` is invalid
        """
        assert (
            year in self.valid_years
        ), f"NLCD data product only exists for the following years: {self.valid_years}."
        self.year = year
        self.root = root
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(root, crs, res, transforms=transforms, cache=cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        filename_year = self.filename_glob.replace("*", str(self.year))
        dirname_year = filename_year.split(".")[0]
        pathname = os.path.join(self.root, dirname_year, filename_year)
        if os.path.exists(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(
            self.root, self.zipfile_glob.replace("*", str(self.year))
        )
        if os.path.exists(pathname):
            self._extract()
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
        download_url(
            self.urls[self.year],
            self.root,
            md5=self.md5s[self.year] if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        zipfile_name = self.zipfile_glob.replace("*", str(self.year))
        pathname = os.path.join(self.root, zipfile_name)
        extract_archive(pathname, os.path.join(self.root, zipfile_name.split(".")[0]))

    def plot(
        self,
        sample: Dict[str, Any],
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

        cmap: "np.typing.NDArray[np.int_]" = np.zeros((max(self.cmap) + 1, 4), np.int_)
        for idx, cmap_val in self.cmap.items():
            cmap[idx, :] = cmap_val

        mask = cmap[mask]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze().numpy()
            pred = cmap[pred]
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(pred)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(mask)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return
