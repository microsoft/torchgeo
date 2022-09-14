# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Esri 2020 Land Cover Dataset."""

import glob
import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import download_url, extract_archive


class Esri2020(RasterDataset):
    """Esri 2020 Land Cover Dataset.

    The `Esri 2020 Land Cover dataset
    <https://www.arcgis.com/home/item.html?id=fc92d38533d440078f17678ebc20e8e2>`_
    consists of a global single band land use/land cover map derived from ESA
    Sentinel-2 imagery at 10m resolution with a total of 10 classes.
    It was published in July 2021 and used the Universal Transverse Mercator (UTM)
    projection. This dataset only contains labels, no raw satellite imagery.

    The 10 classes are:

    0. No Data
    1. Water
    2. Trees
    3. Grass
    4. Flooded Vegetation
    5. Crops
    6. Scrub/Shrub
    7. Built Area
    8. Bare Ground
    9. Snow/Ice
    10. Clouds

    A more detailed explanation of the invidual classes can be found
    `here <https://www.arcgis.com/home/item.html?id=fc92d38533d440078f17678ebc20e8e2>`_.

    If you use this dataset please cite the following paper:

    * https://ieeexplore.ieee.org/document/9553499

    .. versionadded:: 0.3
    """

    is_image = False
    filename_glob = "*_20200101-20210101.*"
    filename_regex = r"""^
        (?P<id>[0-9][0-9][A-Z])
        _(?P<date>\d{8})
        -(?P<processing_date>\d{8})
    """

    zipfile = "io-lulc-model-001-v01-composite-v03-supercell-v02-clip-v01.zip"
    md5 = "4932855fcd00735a34b74b1f87db3df0"

    url = (
        "https://ai4edataeuwest.blob.core.windows.net/io-lulc/"
        "io-lulc-model-001-v01-composite-v03-supercell-v02-clip-v01.zip"
    )

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
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
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
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
        # Check if the extracted file already exists
        pathname = os.path.join(self.root, "**", self.filename_glob)
        if glob.glob(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.zipfile)
        if glob.glob(pathname):
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
        download_url(self.url, self.root, filename=self.zipfile, md5=self.md5)

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.zipfile))

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
        mask = sample["mask"].squeeze()
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = sample["prediction"].squeeze()
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(prediction)
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

        return fig
