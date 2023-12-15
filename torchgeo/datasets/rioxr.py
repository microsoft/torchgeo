# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""In-memory geographical xarray.DataArray and xarray.Dataset."""

import glob
import os
import re
import sys
from datetime import datetime
from typing import Any, Callable, Optional, cast

import numpy as np
import torch
import xarray as xr
from rasterio.crs import CRS
from rioxarray.merge import merge_arrays
from rtree.index import Index, Property

from .geo import GeoDataset
from .utils import BoundingBox


class RioXarrayDataset(GeoDataset):
    """Wrapper for geographical datasets stored as Xarray Datasets.

    Relies on rioxarray.

    .. versionadded:: 5.0
    """

    filename_glob = "*"
    filename_regex = ".*"

    is_image = True

    spatial_x_name = "x"
    spatial_y_name = "y"

    transform = None

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the dataset (overrides the dtype of the data file via a cast).

        Returns:
            the dtype of the dataset
        """
        if self.is_image:
            return torch.float32
        else:
            return torch.long

    def harmonize_format(self, ds):
        """Convert the dataset to the standard format.

        Args:
            ds: dataset or array to harmonize

        Returns:
            the harmonized dataset or array
        """
        # rioxarray expects spatial dimensions to be named x and y
        ds.rio.set_spatial_dims(self.spatial_x_name, self.spatial_y_name, inplace=True)

        # if x coords go from 0 to 360, convert to -180 to 180
        if ds[self.spatial_x_name].min() > 180:
            ds = ds.assign_coords(
                {self.spatial_x_name: ds[self.spatial_x_name] % 360 - 180}
            )

        # if y coords go from 0 to 180, convert to -90 to 90
        if ds[self.spatial_x_name].min() > 90:
            ds = ds.assign_coords(
                {self.spatial_y_name: ds[self.spatial_y_name] % 180 - 90}
            )
        # expect asceding coordinate values
        ds = ds.sortby(self.spatial_x_name, ascending=True)
        ds = ds.sortby(self.spatial_y_name, ascending=True)
        return ds

    def __init__(
        self,
        root: str,
        data_variables: Optional[list[str]] = None,
        crs: Optional[CRS] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: directory with files to be opened with xarray
            data_variables: data variables that should be gathered from the collection
                of xarray datasets
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of dataarray)
            transforms: a function/transform that takes an input sample
                and returns a transformed version

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(transforms)

        self.root = root

        if data_variables:
            self.data_variables = data_variables
        else:
            data_variables_to_collect: list[str] = []

        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                with xr.open_dataset(filepath, decode_times=True) as ds:
                    ds = self.harmonize_format(ds)

                    if crs is None:
                        crs = ds.rio.crs

                    try:
                        (minx, miny, maxx, maxy) = ds.rio.bounds()
                    except AttributeError:
                        # or take the shape of the data variable?
                        continue

                if hasattr(ds, "time"):
                    try:
                        indices = ds.indexes["time"].to_datetimeindex()
                    except AttributeError:
                        indices = ds.indexes["time"]

                    mint = indices.min().to_pydatetime().timestamp()
                    maxt = indices.max().to_pydatetime().timestamp()
                else:
                    mint = 0
                    maxt = sys.maxsize
                coords = (minx, maxx, miny, maxy, mint, maxt)
                self.index.insert(i, coords, filepath)
                i += 1

                # collect all possible data variables if self.data_variables is None
                if not data_variables:
                    data_variables_to_collect.extend(list(ds.data_vars))

        if i == 0:
            import pdb

            pdb.set_trace()
            msg = f"No {self.__class__.__name__} data was found in `root='{self.root}'`"
            raise FileNotFoundError(msg)

        if not data_variables:
            self.data_variables = list(set(data_variables_to_collect))

        if not crs:
            self._crs = "EPSG:4326"
        else:
            self._crs = cast(CRS, crs)
        self.res = 1.0

    def _infer_spatial_coordinate_names(self, ds) -> tuple[str]:
        """Infer the names of the spatial coordinates.

        Args:
            ds: Dataset or DataArray of which to infer the spatial coordinates

        Returns:
            x and y coordinate names
        """
        x_name = None
        y_name = None
        for coord_name, coord in ds.coords.items():
            if hasattr(coord, "units"):
                if any(
                    [
                        x in coord.units.lower()
                        for x in ["degrees_north", "degree_north"]
                    ]
                ):
                    y_name = coord_name
                elif any(
                    [x in coord.units.lower() for x in ["degrees_east", "degree_east"]]
                ):
                    x_name = coord_name

        if not x_name or not y_name:
            raise ValueError("Spatial Coordinate Units not found in Dataset.")

        return x_name, y_name

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

        data_arrays: list["np.typing.NDArray"] = []
        for item in items:
            with xr.open_dataset(item, decode_cf=True) as ds:
                ds = self.harmonize_format(ds)
                # select time dimension
                if hasattr(ds, "time"):
                    try:
                        ds["time"] = ds.indexes["time"].to_datetimeindex()
                    except AttributeError:
                        ds["time"] = ds.indexes["time"]
                    ds = ds.sel(
                        time=slice(
                            datetime.fromtimestamp(query.mint),
                            datetime.fromtimestamp(query.maxt),
                        )
                    )

                for variable in self.data_variables:
                    if hasattr(ds, variable):
                        da = ds[variable]
                        if not da.rio.crs:
                            da.rio.write_crs(self._crs, inplace=True)
                        elif da.rio.crs != self._crs:
                            da = da.rio.reproject(self._crs)
                        # clip box ignores time dimension
                        clipped = da.rio.clip_box(
                            minx=query.minx,
                            miny=query.miny,
                            maxx=query.maxx,
                            maxy=query.maxy,
                        )
                        # rioxarray expects this order
                        clipped = clipped.transpose(
                            "time", self.spatial_y_name, self.spatial_x_name, ...
                        )

                        # set proper transform # TODO not working
                        clipped.rio.write_transform(self.transform)
                        data_arrays.append(clipped.squeeze())

        import pdb

        pdb.set_trace()
        merged_data = torch.from_numpy(
            merge_arrays(
                data_arrays, bounds=(query.minx, query.miny, query.maxx, query.maxy)
            ).data
        )
        sample = {"crs": self.crs, "bbox": query}

        merged_data = merged_data.to(self.dtype)
        if self.is_image:
            sample["image"] = merged_data
        else:
            sample["mask"] = merged_data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
