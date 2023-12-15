# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Aster Global Digital Elevation Model dataset."""

from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import DatasetNotFoundError


class AsterGDEM(RasterDataset):
    """Aster Global Digital Elevation Model Dataset.

    The `Aster Global Digital Elevation Model
    <https://lpdaac.usgs.gov/products/astgtmv003/>`_
    dataset is a Digital Elevation Model (DEM) on a global scale.
    The dataset can be downloaded from the
    `Earth Data website <https://search.earthdata.nasa.gov/search/>`_
    after making an account.

    Dataset features:

    * DEMs at 30 m per pixel spatial resolution (3601x3601 px)
    * data collected from the `Aster
      <https://terra.nasa.gov/about/terra-instruments/aster>`_ instrument

    Dataset format:

    * DEMs are single-channel tif files

    .. versionadded:: 0.3
    """

    is_image = True
    all_bands = ["elevation"]
    filename_glob = "ASTGTMV003_*_dem*"
    filename_regex = r"""
        (?P<name>[ASTGTMV003]{10})
        _(?P<id>[A-Z0-9]{7})
        _(?P<data>[a-z]{3})*
    """

    def __init__(
        self,
        paths: Union[str, list[str]] = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load, here
                the collection of individual zip files for each tile should be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            DatasetNotFoundError: If dataset is not found.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        self.paths = paths

        self._verify()
        bands = self.all_bands

        super().__init__(
            paths, crs, res, bands=bands, transforms=transforms, cache=cache
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exists
        if self.files:
            return

        raise DatasetNotFoundError(self)

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
        image = sample["image"][0]

        image = torch.clamp(image, min=0, max=1)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Elevation")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
