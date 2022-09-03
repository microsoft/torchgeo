# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Aboveground Live Woody Biomass Density dataset."""

import glob
import json
import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import download_url


class AbovegroundLiveWoodyBiomassDensity(RasterDataset):
    """Aboveground Live Woody Biomass Density dataset.

    The `Aboveground Live Woody Biomass Density dataset
    <https://data.globalforestwatch.org/datasets/gfw::aboveground-live-woody
    -biomass-density/about>`_
    is a global-scale, wall-to-wall map of aboveground biomass at ~30m resolution
    for the year 2000.

    Dataset features:

    * Masks with per pixel live woody biomass density estimates in megagrams
      biomass per hectare at ~30m resolution (~40,000x40,0000 px)

    Dataset format:

    * geojson file that contains download links to tif files
    * single-channel geotiffs with the pixel values representing biomass density

    If you use this dataset in your research, please give credit to:

    * `Global Forest Watch <https://data.globalforestwatch.org/>`_

    .. versionadded:: 0.3
    """

    is_image = False

    url = (
        "https://opendata.arcgis.com/api/v3/datasets/3e8736c8866b458687"
        "e00d40c9f00bce_0/downloads/data?format=geojson&spatialRefId=4326"
    )

    base_filename = "Aboveground_Live_Woody_Biomass_Density.geojson"

    filename_glob = "*N_*E.*"
    filename_regex = r"""^
        (?P<latitude>[0-9][0-9][A-Z])_
        (?P<longitude>[0-9][0-9][0-9][A-Z])*
    """

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        cache: bool = True,
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
            download: if True, download dataset and store it in the root directory
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        self.root = root
        self.download = download

        self._verify()

        super().__init__(root, crs, res, transforms=transforms, cache=cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, self.filename_glob)
        if glob.glob(pathname):
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

    def _download(self) -> None:
        """Download the dataset."""
        download_url(self.url, self.root, self.base_filename)

        with open(os.path.join(self.root, self.base_filename)) as f:
            content = json.load(f)

        for item in content["features"]:
            download_url(
                item["properties"]["download"],
                self.root,
                item["properties"]["tile_id"] + ".tif",
            )

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
            pred = sample["prediction"].squeeze()
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

        return fig
