# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import zipfile
from pathlib import Path

from tests.data.utils import write_image

images = [
    (
        'task1/RemoteSensing/CHM/MLBS_4.tif',
        1,
        'float32',
        'EPSG:32617',
        (1.0, 0.0, 541960.0, 0.0, -1.0, 4136489.0),
    ),
    (
        'task1/RemoteSensing/CHM/OSBS_11.tif',
        1,
        'float32',
        'EPSG:32617',
        (1.0, 0.0, 401464.0, 0.0, -1.0, 3285871.0),
    ),
    (
        'task1/RemoteSensing/CHM/TALL_1.tif',
        1,
        'float32',
        'EPSG:32616',
        (1.0, 0.0, 458915.0, 0.0, -1.0, 3642021.0),
    ),
    (
        'task1/RemoteSensing/HSI/MLBS_4.tif',
        369,
        'int16',
        'EPSG:32617',
        (1.0, 0.0, 541960.0, 0.0, -1.0, 4136489.0),
    ),
    (
        'task1/RemoteSensing/HSI/OSBS_11.tif',
        369,
        'int16',
        'EPSG:32617',
        (1.0, 0.0, 401464.0, 0.0, -1.0, 3285871.0),
    ),
    (
        'task1/RemoteSensing/HSI/TALL_1.tif',
        369,
        'int16',
        'EPSG:32616',
        (1.0, 0.0, 458915.0, 0.0, -1.0, 3642021.0),
    ),
    (
        'task1/RemoteSensing/RGB/MLBS_4.tif',
        3,
        'float32',
        'EPSG:32617',
        (0.1, 0.0, 541960.0, 0.0, -0.1, 4136489.0),
    ),
    (
        'task1/RemoteSensing/RGB/OSBS_11.tif',
        3,
        'float32',
        'EPSG:32617',
        (0.1, 0.0, 401464.0, 0.0, -0.1, 3285871.0),
    ),
    (
        'task1/RemoteSensing/RGB/TALL_1.tif',
        3,
        'float32',
        'EPSG:32616',
        (0.1, 0.0, 458915.0, 0.0, -0.1, 3642021.0),
    ),
    (
        'task2/RemoteSensing/CHM/MLBS_1.tif',
        1,
        'float32',
        'EPSG:32617',
        (1.0, 0.0, 541876.0, 0.0, -1.0, 4136599.0),
    ),
    (
        'task2/RemoteSensing/CHM/OSBS_15.tif',
        1,
        'float32',
        'EPSG:32617',
        (1.0, 0.0, 401562.0, 0.0, -1.0, 3285936.0),
    ),
    (
        'task2/RemoteSensing/CHM/TALL_2.tif',
        1,
        'float32',
        'EPSG:32616',
        (1.0, 0.0, 458915.0, 0.0, -1.0, 3642000.0),
    ),
    (
        'task2/RemoteSensing/HSI/MLBS_1.tif',
        369,
        'int16',
        'EPSG:32617',
        (1.0, 0.0, 541876.0, 0.0, -1.0, 4136599.0),
    ),
    (
        'task2/RemoteSensing/HSI/OSBS_15.tif',
        369,
        'int16',
        'EPSG:32617',
        (1.0, 0.0, 401562.0, 0.0, -1.0, 3285936.0),
    ),
    (
        'task2/RemoteSensing/HSI/TALL_2.tif',
        369,
        'int16',
        'EPSG:32616',
        (1.0, 0.0, 458915.0, 0.0, -1.0, 3642000.0),
    ),
    (
        'task2/RemoteSensing/RGB/MLBS_1.tif',
        3,
        'float32',
        'EPSG:32617',
        (0.1, 0.0, 541876.0, 0.0, -0.1, 4136599.0),
    ),
    (
        'task2/RemoteSensing/RGB/OSBS_15.tif',
        3,
        'float32',
        'EPSG:32617',
        (0.1, 0.0, 401562.0, 0.0, -0.1, 3285936.0),
    ),
    (
        'task2/RemoteSensing/RGB/TALL_2.tif',
        3,
        'float32',
        'EPSG:32616',
        (0.1, 0.0, 458915.0, 0.0, -0.1, 3642000.0),
    ),
    (
        'train/RemoteSensing/CHM/MLBS_1.tif',
        1,
        'float32',
        'EPSG:32617',
        (1.0, 0.0, 542055.0, 0.0, -1.0, 4134999.0),
    ),
    (
        'train/RemoteSensing/CHM/OSBS_1.tif',
        1,
        'float32',
        'EPSG:32617',
        (1.0, 0.0, 404023.0, 0.0, -1.0, 3284961.0),
    ),
    (
        'train/RemoteSensing/CHM/OSBS_39.tif',
        1,
        'float32',
        'EPSG:32617',
        (1.0, 0.0, 404242.0, 0.0, -1.0, 3284853.0),
    ),
    (
        'train/RemoteSensing/HSI/MLBS_1.tif',
        369,
        'int16',
        'EPSG:32617',
        (1.0, 0.0, 542055.0, 0.0, -1.0, 4134999.0),
    ),
    (
        'train/RemoteSensing/HSI/OSBS_1.tif',
        369,
        'int16',
        'EPSG:32617',
        (1.0, 0.0, 404023.0, 0.0, -1.0, 3284961.0),
    ),
    (
        'train/RemoteSensing/HSI/OSBS_39.tif',
        369,
        'int16',
        'EPSG:32617',
        (1.0, 0.0, 404242.0, 0.0, -1.0, 3284853.0),
    ),
    (
        'train/RemoteSensing/RGB/MLBS_1.tif',
        3,
        'float32',
        'EPSG:32617',
        (0.1, 0.0, 542055.0, 0.0, -0.1, 4134999.0),
    ),
    (
        'train/RemoteSensing/RGB/OSBS_1.tif',
        3,
        'float32',
        'EPSG:32617',
        (0.1, 0.0, 404023.0, 0.0, -0.1, 3284961.0),
    ),
    (
        'train/RemoteSensing/RGB/OSBS_39.tif',
        3,
        'float32',
        'EPSG:32617',
        (0.1, 0.0, 404242.0, 0.0, -0.1, 3284853.0),
    ),
]
for path, count, dtype, crs, transform in images:
    write_image(
        path,
        {
            'driver': 'GTiff',
            'height': 2,
            'width': 2,
            'count': count,
            'dtype': dtype,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',
        },
    )

import laspy
import numpy as np

path = Path('task1/RemoteSensing/LAS/MLBS_4.las')
path.parent.mkdir(parents=True, exist_ok=True)
las = laspy.create(point_format=1, file_version='1.3')
las.x = np.arange(4)
las.y = las.x
las.z = las.x
las.write(path)

import laspy
import numpy as np

path = Path('task1/RemoteSensing/LAS/OSBS_11.las')
path.parent.mkdir(parents=True, exist_ok=True)
las = laspy.create(point_format=3, file_version='1.3')
las.x = np.arange(4)
las.y = las.x
las.z = las.x
las.write(path)

import laspy
import numpy as np

path = Path('task1/RemoteSensing/LAS/TALL_1.las')
path.parent.mkdir(parents=True, exist_ok=True)
las = laspy.create(point_format=3, file_version='1.3')
las.x = np.arange(4)
las.y = las.x
las.z = las.x
las.write(path)

import geopandas as gpd

gpd.read_file('task2/ITC/test_MLBS.geojson').to_file(
    'task2/ITC/test_MLBS.shp', driver='ESRI Shapefile'
)

import geopandas as gpd

gpd.read_file('task2/ITC/test_OSBS.geojson').to_file(
    'task2/ITC/test_OSBS.shp', driver='ESRI Shapefile'
)

import geopandas as gpd

gpd.read_file('task2/ITC/test_TALL.geojson').to_file(
    'task2/ITC/test_TALL.shp', driver='ESRI Shapefile'
)

import laspy
import numpy as np

path = Path('task2/RemoteSensing/LAS/MLBS_1.las')
path.parent.mkdir(parents=True, exist_ok=True)
las = laspy.create(point_format=1, file_version='1.3')
las.x = np.arange(4)
las.y = las.x
las.z = las.x
las.write(path)

import laspy
import numpy as np

path = Path('task2/RemoteSensing/LAS/OSBS_15.las')
path.parent.mkdir(parents=True, exist_ok=True)
las = laspy.create(point_format=1, file_version='1.3')
las.x = np.arange(4)
las.y = las.x
las.z = las.x
las.write(path)

import laspy
import numpy as np

path = Path('task2/RemoteSensing/LAS/TALL_2.las')
path.parent.mkdir(parents=True, exist_ok=True)
las = laspy.create(point_format=3, file_version='1.3')
las.x = np.arange(4)
las.y = las.x
las.z = las.x
las.write(path)

import geopandas as gpd

gpd.read_file('train/ITC/train_MLBS.geojson').to_file(
    'train/ITC/train_MLBS.shp', driver='ESRI Shapefile'
)

import geopandas as gpd

gpd.read_file('train/ITC/train_OSBS.geojson').to_file(
    'train/ITC/train_OSBS.shp', driver='ESRI Shapefile'
)

import laspy
import numpy as np

path = Path('train/RemoteSensing/LAS/MLBS_1.las')
path.parent.mkdir(parents=True, exist_ok=True)
las = laspy.create(point_format=1, file_version='1.3')
las.x = np.arange(4)
las.y = las.x
las.z = las.x
las.write(path)

import laspy
import numpy as np

path = Path('train/RemoteSensing/LAS/OSBS_1.las')
path.parent.mkdir(parents=True, exist_ok=True)
las = laspy.create(point_format=3, file_version='1.3')
las.x = np.arange(4)
las.y = las.x
las.z = las.x
las.write(path)

import laspy
import numpy as np

path = Path('train/RemoteSensing/LAS/OSBS_39.las')
path.parent.mkdir(parents=True, exist_ok=True)
las = laspy.create(point_format=3, file_version='1.3')
las.x = np.arange(4)
las.y = las.x
las.z = las.x
las.write(path)

with zipfile.ZipFile(
    'IDTREES_competition_test_v2.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'task1/RemoteSensing/HSI/MLBS_4.tif',
        'task1/RemoteSensing/HSI/TALL_1.tif',
        'task1/RemoteSensing/HSI/OSBS_11.tif',
        'task1/RemoteSensing/RGB/MLBS_4.tif',
        'task1/RemoteSensing/RGB/TALL_1.tif',
        'task1/RemoteSensing/RGB/OSBS_11.tif',
        'task1/RemoteSensing/LAS/TALL_1.las',
        'task1/RemoteSensing/LAS/MLBS_4.las',
        'task1/RemoteSensing/LAS/OSBS_11.las',
        'task1/RemoteSensing/CHM/MLBS_4.tif',
        'task1/RemoteSensing/CHM/TALL_1.tif',
        'task1/RemoteSensing/CHM/OSBS_11.tif',
        'task2/RemoteSensing/HSI/OSBS_15.tif',
        'task2/RemoteSensing/HSI/MLBS_1.tif',
        'task2/RemoteSensing/HSI/TALL_2.tif',
        'task2/RemoteSensing/RGB/OSBS_15.tif',
        'task2/RemoteSensing/RGB/MLBS_1.tif',
        'task2/RemoteSensing/RGB/TALL_2.tif',
        'task2/RemoteSensing/LAS/OSBS_15.las',
        'task2/RemoteSensing/LAS/MLBS_1.las',
        'task2/RemoteSensing/LAS/TALL_2.las',
        'task2/RemoteSensing/CHM/OSBS_15.tif',
        'task2/RemoteSensing/CHM/MLBS_1.tif',
        'task2/RemoteSensing/CHM/TALL_2.tif',
        'task2/ITC/test_MLBS.shx',
        'task2/ITC/test_TALL.shx',
        'task2/ITC/test_OSBS.dbf',
        'task2/ITC/test_MLBS.dbf',
        'task2/ITC/test_OSBS.shp',
        'task2/ITC/test_MLBS.shp',
        'task2/ITC/test_TALL.shp',
        'task2/ITC/test_TALL.dbf',
        'task2/ITC/test_OSBS.shx',
    ]:
        archive.write(member, member)

with zipfile.ZipFile(
    'IDTREES_competition_train_v2.zip', 'w', compression=zipfile.ZIP_DEFLATED
) as archive:
    for member in [
        'train/RemoteSensing/HSI/OSBS_39.tif',
        'train/RemoteSensing/HSI/OSBS_1.tif',
        'train/RemoteSensing/HSI/MLBS_1.tif',
        'train/RemoteSensing/RGB/OSBS_39.tif',
        'train/RemoteSensing/RGB/OSBS_1.tif',
        'train/RemoteSensing/RGB/MLBS_1.tif',
        'train/RemoteSensing/LAS/OSBS_39.las',
        'train/RemoteSensing/LAS/MLBS_1.las',
        'train/RemoteSensing/LAS/OSBS_1.las',
        'train/RemoteSensing/CHM/OSBS_39.tif',
        'train/RemoteSensing/CHM/OSBS_1.tif',
        'train/RemoteSensing/CHM/MLBS_1.tif',
        'train/ITC/train_OSBS.shx',
        'train/ITC/train_OSBS.dbf',
        'train/ITC/train_OSBS.shp',
        'train/ITC/train_MLBS.shp',
        'train/ITC/train_MLBS.shx',
        'train/ITC/train_MLBS.dbf',
        'train/Field/train_data.csv',
        'train/Field/itc_rsFile.csv',
    ]:
        archive.write(member, member)
