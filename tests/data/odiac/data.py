# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

# --- Configuration for Fake Data ---
# Use a small size for test files to keep them lightweight
SIZE = 32
# Define the ODIAC versions, years, and months to generate fake data for
# Keep this minimal to speed up test setup.
VERSIONS = [2023, 2022]
YEARS = {
    2023: [2021, 2022], # Years available for version 2023
    2022: [2020, 2021]  # Years available for version 2022
}
MONTHS = [1, 7] # Generate only January and July for testing

# Metadata extracted from gdalinfo
CRS_ODIAC = CRS.from_epsg(4326)
# Origin = (-180.0, 90.0) Pixel Size = (0.008333..., -0.008333...)
TRANSFORM_ODIAC = Affine(0.008333333333333, 0.0, -180.0,
                         0.0, -0.008333333333333, 90.0)
DTYPE_ODIAC = rasterio.float32
# -----------------------------------

np.random.seed(0) # for reproducibility

def create_fake_odiac_tif(
    filepath: Path,
    dtype: np.dtype,
    crs: CRS,
    transform: Affine,
    size: int
    ) -> None:
    """Creates a fake ODIAC GeoTIFF file."""
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'count': 1, # Single band for CO2 emission
        'crs': crs,
        'transform': transform,
        'height': size,
        'width': size,
        'nodata': None # Or set if ODIAC uses a specific nodata value
    }

    # Generate random emission data (e.g., positive values)
    # Adjust range if needed based on typical ODIAC values
    emission_data = np.random.rand(size, size).astype(dtype) * 100

    with rasterio.open(filepath, 'w', **profile) as dst:
        dst.write(emission_data, 1)

if __name__ == '__main__':
    # Get the directory of the script being run
    script_dir = Path(__file__).parent

    print(f"Generating fake ODIAC data in: {script_dir}")

    # Remove old generated data directories first
    for version in VERSIONS:
        for year in YEARS.get(version, []):
            year_dir = script_dir / str(year)
            if year_dir.exists():
                print(f"Removing old directory: {year_dir}")
                shutil.rmtree(year_dir)

    # Create fake data
    for version in VERSIONS:
        for year in YEARS.get(version, []):
            year_dir = script_dir / str(year)
            year_dir.mkdir(exist_ok=True)
            print(f"Creating files for Version {version}, Year {year}...")

            for month in MONTHS:
                yymm_str = f"{str(year)[-2:]}{month:02d}"
                filename = f"odiac{version}_1km_excl_intl_{yymm_str}.tif"
                filepath = year_dir / filename

                create_fake_odiac_tif(
                    filepath=filepath,
                    dtype=DTYPE_ODIAC,
                    crs=CRS_ODIAC,
                    transform=TRANSFORM_ODIAC, # Using real transform is good practice
                    size=SIZE
                )
                print(f"  Created: {filepath.name}")

    print("Fake data generation complete.")