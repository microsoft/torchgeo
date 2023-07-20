# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""In-memory geographical xarray.DataArray."""

import glob
import os
import re
import sys
from typing import Any, Callable, Optional, cast

import numpy as np
import torch
import xarray as xr
from rasterio.crs import CRS
from rtree.index import Index, Property

from .geo import GeoDataset
from .utils import BoundingBox


class RioXarrayDataset(GeoDataset):
    """Wrapper for geographical datasets stored as an xarray.DataArray.

    Relies on rioxarray.
    """

    filename_glob = "*os_new.nc"
    filename_regex = ".*"

    def __init__(
        self,
        root: str,
        # xr_dataarray: xr.DataArray,
        data_variables: list[str],
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: directory with nc files
            data_variables: data variables that should be gathered from the xr_datasets
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of dataarray)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the dataarray)
            transforms: a function/transform that takes an input sample
                and returns a transformed version

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(transforms)

        self.root = root
        self.data_variables = data_variables
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            match = re.match(filename_regex, os.path.basename(filepath))
            print(filepath)
            if match is not None:
                with xr.open_dataset(filepath, decode_times=False) as ds:
                    if crs is None:
                        crs = ds.rio.crs
                    if res is None:
                        res = ds.rio.resolution()[0]

                    (minx, miny, maxx, maxy) = ds.rio.bounds()

                if hasattr(ds, "time"):
                    mint = int(ds.time.min().data)
                    maxt = int(ds.time.max().data)
                else:
                    mint = 0
                    maxt = sys.maxsize
                coords = (minx, maxx, miny, maxy, mint, maxt)
                self.index.insert(i, coords, filepath)
                i += 1

        if i == 0:
            msg = f"No {self.__class__.__name__} data was found in `root='{self.root}'`"
            raise FileNotFoundError(msg)

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        items = [hit.object for hit in hits]

        if not items:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data_arrays: list[np.ndarray] = []
        for item in items:
            with xr.open_dataset(item) as ds:
                ds.rio.write_crs("EPSG:4326", inplace=True)
                clipped = ds.rio.clip_box(
                    minx=query.minx, miny=query.miny, maxx=query.maxx, maxy=query.maxy
                )
                for variable in self.data_variables:
                    if hasattr(clipped, variable):
                        data_arrays.append(clipped[variable].data.squeeze())

        sample = {"image": torch.from_numpy(np.stack(data_arrays)), "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
