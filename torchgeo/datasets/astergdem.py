# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Aster Global Digital Evaluation Model dataset."""

import glob
import os
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
from rasterio.crs import CRS

from .geo import RasterDataset


class AsterGDEM(RasterDataset):
    """Aster Global Digital Evaluation Model Dataset.

    The `Aster Global Digital Evaluation Model
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

    is_image = False
    filename_glob = "ASTGTMV003_*_dem*"
    filename_regex = r"""
        (?P<name>[ASTGTMV003]{10})
        _(?P<id>[A-Z0-9]{7})
        _(?P<data>[a-z]{3})*
    """

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found, here the collection of
                individual zip files for each tile should be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if dataset is missing
        """
        self.root = root

        self._verify()

        super().__init__(root, crs, res, transforms=transforms, cache=cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing
        """
        # Check if the extracted files already exists
        pathname = os.path.join(self.root, self.filename_glob)
        if glob.glob(pathname):
            return

        raise RuntimeError(
            f"Dataset not found in `root={self.root}` "
            "either specify a different `root` directory or make sure you "
            "have manually downloaded dataset tiles as suggested in the documentation."
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
