# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""South America Soybean Dataset."""

import glob
import os
from collections.abc import Iterable
from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import BoundingBox, DatasetNotFoundError, download_url, extract_archive


class SouthAmericaSoybean(RasterDataset):
    """South America Soybean Dataset.

    This dataset produced annual 30-m soybean maps of South America from 2001 to 2021.

    Link: https://www.nature.com/articles/s41893-021-00729-z

    Dataset contains 2 classes:
    0: nodata
    1: soybean

    Dataset Format:

    * 21 .tif files


    If you use this dataset in your research, please use the corresponding citation:

    * https://doi.org/10.1038/s41893-021-00729-z

    .. versionadded:: 0.6
    """

    filename_glob = "South_America_Soybean_*.*"
    filename_regex = r"South_America_Soybean_(?P<year>\d{4})"
    zipfile_glob = "SouthAmericaSoybean.zip"

    date_format = "%Y"
    is_image = False
    url = "https://glad.umd.edu/projects/AnnualClassMapsV1/SouthAmerica_Soybean_"
    md5 = "7f1d06a57cc6c4ae6be3b3fb9464ddeb"

    def __init__(
        self,
        paths: Union[str, Iterable[str]] = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
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
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.paths = paths
        self.download = download
        self.checksum = checksum
        self._verify()

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

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

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        if self.files:
            return
        assert isinstance(self.paths, str)
        pathname = os.path.join(self.paths, "**", self.zipfile_glob)
        if glob.glob(pathname, recursive=True):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        filename = "SouthAmericaSoybean.zip"

        download_url(
            self.url, self.paths, filename, md5=self.md5 if self.checksum else None
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str)

        pathname = os.path.join(self.paths, "**", self.zipfile_glob)

        extract_archive(glob.glob(pathname, recursive=True)[0], self.paths)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.
        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle
        Returns:
            a matplotlib Figure with the rendered sample
        """
        mask = sample["mask"].squeeze()
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze()
            ncols = 2

        fig, axs = plt.subplots(
            nrows=1, ncols=ncols, figsize=(ncols * 4, 4), squeeze=False
        )

        axs[0, 0].imshow(mask, interpolation="none")
        axs[0, 0].axis("off")

        if show_titles:
            axs[0, 0].set_title("Mask")

        if showing_predictions:
            axs[0, 1].imshow(pred, interpolation="none")
            axs[0, 1].axis("off")
            if show_titles:
                axs[0, 1].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
