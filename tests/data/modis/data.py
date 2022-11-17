#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Create test data for modis dataset."""

import numpy as np
import rioxarray  # noqa: F401
import xarray as xa
from rasterio.crs import CRS

SIZE = 32

filenames = [
    "MOD09GA.A2022182.h12v04.006.2022184031828.hdf",
    "MOD09GA.A2022182.h12v05.006.2022184031828.hdf",
]


def create_file(path: str, dtype: str, num_channels: int) -> None:
    """Create .hdf file."""
    xa_dataset = xa.Dataset(attrs={"count": num_channels})

    # add spatial_ref
    for i in range(num_channels):

        band = xa.DataArray(
            np.random.randint(0, 100, (1, SIZE, SIZE), dtype=dtype),
            dims=("band", "y", "x"),
            coords={"band": [1], "y": np.arange(0, SIZE), "x": np.arange(0, SIZE)},
        )

        band.rio.write_crs(
            CRS.from_wkt(
                'PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom,'
                'spheroid", DATUM["Not specified (based on custom spheroid)",'
                'SPHEROID["Custom spheroid",6371007.181,0]]'
                ',PRIMEM["Greenwich",0],'
                'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]]'
                ',PROJECTION["Sinusoidal"]'
                ',PARAMETER["longitude_of_center",0]'
                ',PARAMETER["false_easting",0]'
                ',PARAMETER["false_northing",0]'
                ',UNIT["Meter",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
            ),
            inplace=True,
        )

        xa_dataset[f"sur_refl_b{i+1:02d}_1"] = band

    xa_dataset.rio.to_raster(
        path
    )  # rioxarray.exceptions.TooManyDimensions: Only 2D and 3D data arrays supported.


if __name__ == "__main__":
    for filename in filenames:
        create_file(filename, "int16", 7)
