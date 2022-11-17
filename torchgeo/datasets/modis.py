# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Modis dataset."""

import functools
import glob
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray as rxr
import torch
from rasterio.crs import CRS
from rioxarray.merge import merge_datasets
from torch import Tensor
from xarray import Dataset as xrDataset

from torchgeo.datasets.geo import GeoDataset

from .utils import BoundingBox, disambiguate_timestamp


class Modis(GeoDataset):
    """Modis Dataset.

    Moderate Resolution Imaging Spectroradiometer (`Modis
    <https://modis.gsfc.nasa.gov/data/>`_) data measures up to 36 spectral
    bands at different resolution for earth and climate monitoring.
    Datasets can be downloaded from the `USGS Earth Explorer website
    <https://earthexplorer.usgs.gov/>`_ after making an account.

    Dataset features:

    * Multispectral data at various resolutions (250m, 500m, and 1km)
    * data collected from `Terra <https://terra.nasa.gov/about/terra-instruments>`_
      or `Aqua <https://aqua.nasa.gov/content/instruments>`_ instrument

    Dataset format

    * .hdf files containing data variables and meta data

    Since there are many different Modis data products, you might
    have to adapt the *filename_glob* and *filename_regex* to build
    your dataset. This defautl implementation is for the USGS
    `MOD09GA` product.

    .. note::
        This dataset requires the following additional installations:

        * `rioxarray <https://corteva.github.io/rioxarray/stable/>`_ to load
          the modis files
        * a full GDAL installation from source for rasterio, for information see
          `rasterio documentation
          <https://rasterio.readthedocs.io/en/latest/installation.html>`_.


    .. versionadded:: 0.4
    """

    filename_glob = "MOD09GA.*.hdf"
    filename_regex = r"""^
        (?P<product_name>[MOD09GA]{7})
        .(?P<julian_date>[A0-9]{8})
        .(?P<tile_id>[h0-9v0-9]{6})
        .(?P<collection_ver>[0-9]{3})
        .(?P<julian_data_aquisition>[0-9]{13})
    """

    all_bands = ("B01", "B02", "B03", "B04", "B05", "B06", "B07")

    rgb_bands = ("B01", "B04", "B03")

    date_format = "A%Y%j"

    is_image = True

    def __init__(
        self,
        root: str,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        checksum: bool = False,
    ) -> None:
        """Initialize a new instance of Modis dataset."""
        super().__init__(transforms=transforms)
        self.root = root
        self.checksum = checksum
        self.cache = cache

        bands = bands or self.all_bands
        self._validate_bands(bands)
        self.bands = bands

        # specify band names as they are used in the data product
        self.modis_band_names = [f"sur_refl_{band.lower()}_1" for band in self.bands]

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            match = re.match(filename_regex, os.path.basename(filepath))

            if match is not None:
                try:
                    with rxr.open_rasterio(
                        filepath, variable=self.modis_band_names
                    ) as src:
                        if crs is None:
                            crs = src.rio.crs

                        if res is None:
                            res = src.rio.resolution()[0]

                        warped = src.rio.clip_box(*src.rio.bounds(), crs=crs)
                        minx, miny, maxx, maxy = warped.rio.bounds()

                # not sure what the right exception to raise is here
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue

                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "julian_date" in match.groupdict():
                        date = match.group("julian_date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)

                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        band_indexes = [self.all_bands.index(i) + 1 for i in bands]
        assert len(band_indexes) == len(self.bands)
        self.band_indexes = band_indexes

        # need to specify band indices
        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(List[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data = self._merge_files(filepaths, query)

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _merge_files(self, filepaths: Sequence[str], query: BoundingBox) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            image/mask at that index
        """
        if self.cache:
            rxr_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            rxr_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        dest = (
            merge_datasets(rxr_fhs, bounds, res=self.res)
            .to_array()
            .to_numpy()
            .squeeze(1)
        )

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.from_numpy(dest)
        return tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> xrDataset:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> xrDataset:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped riox Dataset
        """
        src = rxr.open_rasterio(filepath, variable=self.modis_band_names)

        # Only warp if necessary
        if src.rio.crs != self.crs:
            rxr_warped = src.rio.clip_box(*src.rio.bounds(), crs=self.crs)
            return rxr_warped
        else:
            return src

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided sequence of bands to load

        Raises:
            AssertionError: if ``bands`` is not a sequence
            ValueError: if an invalid band name is provided
        """
        assert isinstance(bands, tuple), "'bands' must be a sequence"
        for band in bands:
            if band not in self.all_bands:
                raise ValueError(f"'{band}' is an invalid band name.")

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

        Raises:
            ValueError: if the RGB bands are not included in ``self.bands``
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 2000, min=0, max=1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
