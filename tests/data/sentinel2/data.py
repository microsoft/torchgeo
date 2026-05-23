#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import pyproj
import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.geometry import Polygon
from shapely.ops import transform

SIZE = 128

np.random.seed(0)

FILENAME_HIERARCHY = dict[str, 'FILENAME_HIERARCHY'] | list[str]

filenames: FILENAME_HIERARCHY = {
    # USGS Earth Explorer
    'S2A_MSIL1C_20220412T162841_N0400_R083_T16TFM_20220412T202300.SAFE': {
        'GRANULE': {
            'L1C_T16TFM_A035544_20220412T163959': {
                'IMG_DATA': [
                    'T16TFM_20220412T162841_B01.jp2',
                    'T16TFM_20220412T162841_B02.jp2',
                    'T16TFM_20220412T162841_B03.jp2',
                    'T16TFM_20220412T162841_B04.jp2',
                    'T16TFM_20220412T162841_B05.jp2',
                    'T16TFM_20220412T162841_B06.jp2',
                    'T16TFM_20220412T162841_B07.jp2',
                    'T16TFM_20220412T162841_B08.jp2',
                    'T16TFM_20220412T162841_B09.jp2',
                    'T16TFM_20220412T162841_B10.jp2',
                    'T16TFM_20220412T162841_B11.jp2',
                    'T16TFM_20220412T162841_B12.jp2',
                    'T16TFM_20220412T162841_B8A.jp2',
                    'T16TFM_20220412T162841_TCI.jp2',
                    'T16TFM_20190412T162841_B01.jp2',
                    'T16TFM_20190412T162841_B02.jp2',
                    'T16TFM_20190412T162841_B03.jp2',
                    'T16TFM_20190412T162841_B04.jp2',
                    'T16TFM_20190412T162841_B05.jp2',
                    'T16TFM_20190412T162841_B06.jp2',
                    'T16TFM_20190412T162841_B07.jp2',
                    'T16TFM_20190412T162841_B08.jp2',
                    'T16TFM_20190412T162841_B09.jp2',
                    'T16TFM_20190412T162841_B10.jp2',
                    'T16TFM_20190412T162841_B11.jp2',
                    'T16TFM_20190412T162841_B12.jp2',
                    'T16TFM_20190412T162841_B8A.jp2',
                    'T16TFM_20190412T162841_TCI.jp2',
                ]
            }
        }
    },
    # Copernicus Open Access Hub
    'S2A_MSIL2A_20220414T110751_N0400_R108_T26EMU_20220414T165533.SAFE': {
        'GRANULE': {
            'L2A_T26EMU_A035569_20220414T110747': {
                'IMG_DATA': {
                    'R10m': [
                        'T26EMU_20220414T110751_AOT_10m.jp2',
                        'T26EMU_20220414T110751_B02_10m.jp2',
                        'T26EMU_20220414T110751_B03_10m.jp2',
                        'T26EMU_20220414T110751_B04_10m.jp2',
                        'T26EMU_20220414T110751_B08_10m.jp2',
                        'T26EMU_20220414T110751_TCI_10m.jp2',
                        'T26EMU_20220414T110751_WVP_10m.jp2',
                        'T26EMU_20190414T110751_AOT_10m.jp2',
                        'T26EMU_20190414T110751_B02_10m.jp2',
                        'T26EMU_20190414T110751_B03_10m.jp2',
                        'T26EMU_20190414T110751_B04_10m.jp2',
                        'T26EMU_20190414T110751_B08_10m.jp2',
                        'T26EMU_20190414T110751_TCI_10m.jp2',
                        'T26EMU_20190414T110751_WVP_10m.jp2',
                    ],
                    'R20m': [
                        'T26EMU_20220414T110751_AOT_20m.jp2',
                        'T26EMU_20220414T110751_B01_20m.jp2',
                        'T26EMU_20220414T110751_B02_20m.jp2',
                        'T26EMU_20220414T110751_B03_20m.jp2',
                        'T26EMU_20220414T110751_B04_20m.jp2',
                        'T26EMU_20220414T110751_B05_20m.jp2',
                        'T26EMU_20220414T110751_B06_20m.jp2',
                        'T26EMU_20220414T110751_B07_20m.jp2',
                        'T26EMU_20220414T110751_B11_20m.jp2',
                        'T26EMU_20220414T110751_B12_20m.jp2',
                        'T26EMU_20220414T110751_B8A_20m.jp2',
                        'T26EMU_20220414T110751_SCL_20m.jp2',
                        'T26EMU_20220414T110751_TCI_20m.jp2',
                        'T26EMU_20220414T110751_WVP_20m.jp2',
                        'T26EMU_20190414T110751_AOT_20m.jp2',
                        'T26EMU_20190414T110751_B01_20m.jp2',
                        'T26EMU_20190414T110751_B02_20m.jp2',
                        'T26EMU_20190414T110751_B03_20m.jp2',
                        'T26EMU_20190414T110751_B04_20m.jp2',
                        'T26EMU_20190414T110751_B05_20m.jp2',
                        'T26EMU_20190414T110751_B06_20m.jp2',
                        'T26EMU_20190414T110751_B07_20m.jp2',
                        'T26EMU_20190414T110751_B11_20m.jp2',
                        'T26EMU_20190414T110751_B12_20m.jp2',
                        'T26EMU_20190414T110751_B8A_20m.jp2',
                        'T26EMU_20190414T110751_SCL_20m.jp2',
                        'T26EMU_20190414T110751_TCI_20m.jp2',
                        'T26EMU_20190414T110751_WVP_20m.jp2',
                    ],
                    'R60m': [
                        'T26EMU_20220414T110751_AOT_60m.jp2',
                        'T26EMU_20220414T110751_B01_60m.jp2',
                        'T26EMU_20220414T110751_B02_60m.jp2',
                        'T26EMU_20220414T110751_B03_60m.jp2',
                        'T26EMU_20220414T110751_B04_60m.jp2',
                        'T26EMU_20220414T110751_B05_60m.jp2',
                        'T26EMU_20220414T110751_B06_60m.jp2',
                        'T26EMU_20220414T110751_B07_60m.jp2',
                        'T26EMU_20220414T110751_B09_60m.jp2',
                        'T26EMU_20220414T110751_B11_60m.jp2',
                        'T26EMU_20220414T110751_B12_60m.jp2',
                        'T26EMU_20220414T110751_B8A_60m.jp2',
                        'T26EMU_20220414T110751_SCL_60m.jp2',
                        'T26EMU_20220414T110751_TCI_60m.jp2',
                        'T26EMU_20220414T110751_WVP_60m.jp2',
                        'T26EMU_20190414T110751_AOT_60m.jp2',
                        'T26EMU_20190414T110751_B01_60m.jp2',
                        'T26EMU_20190414T110751_B02_60m.jp2',
                        'T26EMU_20190414T110751_B03_60m.jp2',
                        'T26EMU_20190414T110751_B04_60m.jp2',
                        'T26EMU_20190414T110751_B05_60m.jp2',
                        'T26EMU_20190414T110751_B06_60m.jp2',
                        'T26EMU_20190414T110751_B07_60m.jp2',
                        'T26EMU_20190414T110751_B09_60m.jp2',
                        'T26EMU_20190414T110751_B11_60m.jp2',
                        'T26EMU_20190414T110751_B12_60m.jp2',
                        'T26EMU_20190414T110751_B8A_60m.jp2',
                        'T26EMU_20190414T110751_SCL_60m.jp2',
                        'T26EMU_20190414T110751_TCI_60m.jp2',
                        'T26EMU_20190414T110751_WVP_60m.jp2',
                    ],
                }
            }
        }
    },
}


# Minimal xml content for the gdal Sentinel2 driver to be able to read
# footprint from tags: `datasource.tags()['FOOTPRINT']`
# Leave placeholder for coords to be filled with actual valid footprint
xml_template = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<n1:Level-1C_User_Product
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
    xsi:schemaLocation="https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-1C.xsd">
    <n1:General_Info>
        <Product_Info>
            <Query_Options completeSingleTile="true">
                <PRODUCT_FORMAT>SAFE_COMPACT</PRODUCT_FORMAT>
            </Query_Options>
            <Product_Organisation></Product_Organisation>
        </Product_Info>
    </n1:General_Info>
    <n1:Geometric_Info>
        <Product_Footprint>
            <Product_Footprint>
                <Global_Footprint>
                    <EXT_POS_LIST>{coords}</EXT_POS_LIST>
                </Global_Footprint>
            </Product_Footprint>
        </Product_Footprint>
    </n1:Geometric_Info>
</n1:Level-1C_User_Product>
"""


def get_product_root(raster_path: str) -> str:
    return raster_path.split('GRANULE')[0]


def _nodata_corner(t: Affine, height: int, width: int) -> Polygon:
    """Return the upper-left triangle containing nodata pixels in the raster's native CRS."""
    cutoff = min(height, width) // 2
    return Polygon([t * (0, 0), t * (cutoff, 0), t * (0, cutoff)])


def create_metadata_file(raster_path: str, nodata_value: float | None) -> None:
    product_root = get_product_root(raster_path)
    metadata_path = os.path.join(product_root, 'MTD_MSIL1C.xml')

    with rasterio.open(raster_path) as src:
        source_crs = src.crs
        t = src.transform
        w, h = src.width, src.height

    bounds = Polygon([t * (0, 0), t * (w, 0), t * (w, h), t * (0, h)])
    valid_footprint = (
        bounds.difference(_nodata_corner(t, h, w))
        if nodata_value is not None
        else bounds
    )

    # .SAFE format always stores the valid footprint in WGS84
    project = pyproj.Transformer.from_crs(
        source_crs, CRS.from_epsg(4326), always_xy=True
    ).transform
    footprint_wgs84 = transform(project, valid_footprint)

    coords = ' '.join(f'{lat} {lon}' for lon, lat in footprint_wgs84.exterior.coords)
    xml_content = xml_template.format(coords=coords)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)


def create_file(
    path: str, dtype: str, num_channels: int, nodata_value: float | None
) -> None:
    res = 10
    root, _ = os.path.splitext(path)
    if root.endswith('m'):
        res = int(root[-3:-1])

    profile = {}
    profile['driver'] = 'JP2OpenJPEG'
    profile['dtype'] = dtype
    profile['count'] = num_channels
    crs = CRS.from_epsg(32616)
    profile['crs'] = crs
    profile['transform'] = Affine(res, 0.0, 399960.0, 0.0, -res, 4500000.0)
    raster_height = round(SIZE * 10 / res)
    raster_width = round(SIZE * 10 / res)
    profile['height'] = raster_height
    profile['width'] = raster_width
    # NB! .SAFE format does not include nodata value in the raster profile...

    if 'float' in profile['dtype']:
        Z = np.random.randn(raster_height, raster_width).astype(profile['dtype'])
    else:
        Z = np.random.randint(
            np.iinfo(profile['dtype']).max,
            size=(raster_height, raster_width),
            dtype=profile['dtype'],
        )

    if nodata_value is not None:
        # Simulates Sentinel-2 acquisitions not fully covering the MGRS cell.
        t = profile['transform']
        corner = _nodata_corner(t, raster_height, raster_width)
        mask = rasterize([corner], out_shape=(raster_height, raster_width), transform=t)
        Z[mask == 1] = nodata_value

    with rasterio.open(path, 'w', **profile) as src:
        for i in range(1, profile['count'] + 1):
            src.write(Z, i)


def create_directory(directory: str, hierarchy: FILENAME_HIERARCHY) -> None:
    if isinstance(hierarchy, dict):
        # Recursive case
        for key, value in hierarchy.items():
            path = os.path.join(directory, key)
            os.makedirs(path, exist_ok=True)
            create_directory(path, value)
    else:
        # Base case
        prev_root = None
        for value in hierarchy:
            path = os.path.join(directory, value)
            nodata_value = 0
            create_file(path, dtype='uint16', num_channels=1, nodata_value=nodata_value)
            # Create the metadata file once for each product
            if (curr_root := get_product_root(path)) != prev_root:
                create_metadata_file(path, nodata_value)
                prev_root = curr_root


if __name__ == '__main__':
    create_directory('.', filenames)
