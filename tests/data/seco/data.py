# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import shutil

from tests.data.utils import write_image

directories = [
    'seasonal_contrast_100k/000000/20190803T154559_20190803T154611_T18QVG',
    'seasonal_contrast_100k/000000/20191027T154551_20191027T154553_T18QVG',
    'seasonal_contrast_100k/000000/20200120T154549_20200120T154543_T18QVH',
    'seasonal_contrast_100k/000000/20200414T154551_20200414T154549_T18QVG',
    'seasonal_contrast_100k/000000/20200623T154601_20200623T154555_T18QVH',
    'seasonal_contrast_100k/000001/20190214T071009_20190214T072118_T39RUJ',
    'seasonal_contrast_100k/000001/20190515T070629_20190515T071534_T39RUJ',
    'seasonal_contrast_100k/000001/20190803T070629_20190803T071729_T39RUJ',
    'seasonal_contrast_100k/000001/20191030T072031_20191030T072025_T39RUJ',
    'seasonal_contrast_100k/000001/20200108T072301_20200108T072255_T39RUJ',
    'seasonal_contrast_1m/000000/20190602T171901_20190602T172737_T13QGE',
    'seasonal_contrast_1m/000000/20190828T170851_20190828T172350_T13QGE',
    'seasonal_contrast_1m/000000/20191121T171619_20191121T172127_T13QGE',
    'seasonal_contrast_1m/000000/20200214T171411_20200214T172644_T13QGE',
    'seasonal_contrast_1m/000000/20200507T171901_20200507T173641_T13QGE',
    'seasonal_contrast_1m/000001/20190706T083611_20190706T084745_T36RTS',
    'seasonal_contrast_1m/000001/20190929T083729_20190929T085127_T36RTS',
    'seasonal_contrast_1m/000001/20191220T083341_20191220T083342_T36RTS',
    'seasonal_contrast_1m/000001/20200309T082751_20200309T084046_T36RTS',
    'seasonal_contrast_1m/000001/20200607T082611_20200607T083813_T36RTS',
]
bands = {
    'B1': 44,
    'B11': 132,
    'B12': 132,
    'B2': 264,
    'B3': 264,
    'B4': 264,
    'B5': 132,
    'B6': 132,
    'B7': 132,
    'B8': 264,
    'B8A': 132,
    'B9': 44,
}
for directory in directories:
    for band, size in bands.items():
        write_image(
            f'{directory}/{band}.tif',
            {
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': 1,
                'height': size,
                'width': size,
                'crs': 'EPSG:4326',
                'transform': (1 / size, 0, 0, 0, -1 / size, 1),
                'compress': 'lzw',
            },
        )
directories = ['seasonal_contrast_100k/000000']
bands = {
    '20190803T154559_20190803T154611_T18QVG': 44,
    '20191027T154551_20191027T154553_T18QVG': 44,
    '20200120T154549_20200120T154543_T18QVH': 44,
    '20200414T154551_20200414T154549_T18QVG': 44,
    '20200623T154601_20200623T154555_T18QVH': 44,
}
for directory in directories:
    for band, size in bands.items():
        write_image(
            f'{directory}/{band}.tif',
            {
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': 1,
                'height': size,
                'width': size,
                'crs': 'EPSG:4326',
                'transform': (1 / size, 0, 0, 0, -1 / size, 1),
                'compress': 'lzw',
            },
        )
directories = ['seasonal_contrast_100k/000001']
bands = {
    '20190214T071009_20190214T072118_T39RUJ': 44,
    '20190515T070629_20190515T071534_T39RUJ': 44,
    '20190803T070629_20190803T071729_T39RUJ': 44,
    '20191030T072031_20191030T072025_T39RUJ': 44,
    '20200108T072301_20200108T072255_T39RUJ': 44,
}
for directory in directories:
    for band, size in bands.items():
        write_image(
            f'{directory}/{band}.tif',
            {
                'driver': 'GTiff',
                'dtype': 'uint8',
                'count': 1,
                'height': size,
                'width': size,
                'crs': 'EPSG:4326',
                'transform': (1 / size, 0, 0, 0, -1 / size, 1),
                'compress': 'lzw',
            },
        )
shutil.make_archive('seco_100k', 'zip', '.', 'seasonal_contrast_100k')
shutil.make_archive('seco_1m', 'zip', '.', 'seasonal_contrast_1m')
