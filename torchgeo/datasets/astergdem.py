# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Aster Global Digital Evaluation Model dataset."""

import glob
import os
from typing import Any, Callable, Dict, Optional

from rasterio.crs import CRS

from .geo import RasterDataset


class AsterGDEM(RasterDataset):
    """AsterGDEM Dataset.

    The `AsterGDEM
    <https://lpdaac.usgs.gov/products/astgtmv003/>`_
    dataset is a Digital Elevation Model of reference on a global scale.
    The dataset can be downloaded from the `Earth Data website
    <https://search.earthdata.nasa.gov/search/`_ after making an account.

    Dataset features:
    * DEMs at 30 m per pixel spatial resolution (3601x3601 px)
    * data collected from `Aster
    <https://terra.nasa.gov/about/terra-instruments/aster>`_ instrument

    Dataset format:
    * DEMs are single-channel tif files

    .. versionadded:: 0.3
    """

    is_image = False
    filename_glob = "ASTGTMV003_*_dem*"
    filename_regex = (
        r"""(?P<name>[ASTGTMV003]{10})_(?P<id>[A-Z0-9]{7}_(?P<data>[a-z]{3})*)"""
    )

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

        super().__init__(root, crs, res, transforms, cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if dataset is missing
        """
        # Check if the extracted file already exists
        pathname = os.path.join(self.root, self.filename_glob)
        if glob.glob(pathname):
            return

        raise RuntimeError(
            f"Dataset not found in `root={self.root}` "
            "either specify a different `root` directory or make sure you "
            "have manually downloaded dataset tiles as suggested in the documentation."
        )
