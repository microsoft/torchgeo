# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet 3 dataset."""

from typing import ClassVar

from .base import SpaceNet


class SpaceNet3(SpaceNet):
    r"""SpaceNet 3: Road Network Detection.

    `SpaceNet 3 <https://spacenet.ai/spacenet-roads-dataset/>`_
    is a dataset of road networks over the cities of Las Vegas, Paris, Shanghai,
    and Khartoum.

    Collection features:

    +------------+---------------------+------------+---------------------------+
    |    AOI     | Area (km\ :sup:`2`\)| # Images   | # Road Network Labels (km)|
    +============+=====================+============+===========================+
    | Vegas      |    216              |   854      |         3685              |
    +------------+---------------------+------------+---------------------------+
    | Paris      |    1030             |   257      |         425               |
    +------------+---------------------+------------+---------------------------+
    | Shanghai   |    1000             |   1028     |         3537              |
    +------------+---------------------+------------+---------------------------+
    | Khartoum   |    765              |   283      |         1030              |
    +------------+---------------------+------------+---------------------------+

    Imagery features:

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1
        :stub-columns: 1

        *   -
            - PAN
            - MS
            - PS-MS
            - PS-RGB
        *   - GSD (m)
            - 0.31
            - 1.24
            - 0.30
            - 0.30
        *   - Chip size (px)
            - 1300 x 1300
            - 325 x 325
            - 1300 x 1300
            - 1300 x 1300

    Dataset format:

    * Imagery - Worldview-3 GeoTIFFs

        * PAN.tif (Panchromatic)
        * MS.tif (Multispectral)
        * PS-MS (Pansharpened Multispectral)
        * PS-RGB (Pansharpened RGB)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232

    .. versionadded:: 0.3
    """

    dataset_id = 'SN3_roads'
    tarballs: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {
            2: [
                'SN3_roads_train_AOI_2_Vegas.tar.gz',
                'SN3_roads_train_AOI_2_Vegas_geojson_roads_speed.tar.gz',
            ],
            3: [
                'SN3_roads_train_AOI_3_Paris.tar.gz',
                'SN3_roads_train_AOI_3_Paris_geojson_roads_speed.tar.gz',
            ],
            4: [
                'SN3_roads_train_AOI_4_Shanghai.tar.gz',
                'SN3_roads_train_AOI_4_Shanghai_geojson_roads_speed.tar.gz',
            ],
            5: [
                'SN3_roads_train_AOI_5_Khartoum.tar.gz',
                'SN3_roads_train_AOI_5_Khartoum_geojson_roads_speed.tar.gz',
            ],
        },
        'test': {
            2: ['SN3_roads_test_public_AOI_2_Vegas.tar.gz'],
            3: ['SN3_roads_test_public_AOI_3_Paris.tar.gz'],
            4: ['SN3_roads_test_public_AOI_4_Shanghai.tar.gz'],
            5: ['SN3_roads_test_public_AOI_5_Khartoum.tar.gz'],
        },
    }
    md5s: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {
            2: ['06317255b5e0c6df2643efd8a50f22ae', '4acf7846ed8121db1319345cfe9fdca9'],
            3: ['c13baf88ee10fe47870c303223cabf82', 'abc8199d4c522d3a14328f4f514702ad'],
            4: ['ef3de027c3da734411d4333bee9c273b', 'f1db36bd17b2be2281f5f7d369e9e25d'],
            5: ['46f327b550076f87babb5f7b43f27c68', 'd969693760d59401a84bd9215375a636'],
        },
        'test': {
            2: ['e9eb2220888ba38cab175fc6db6799a2'],
            3: ['21098cfe471dba6208c92b37b8203ae9'],
            4: ['2e7438b870ffd33d4453366db1c5b317'],
            5: ['f367c79fa0fc1d38e63a0fdd065ed957'],
        },
    }
    valid_aois: ClassVar[dict[str, list[int]]] = {
        'train': [2, 3, 4, 5],
        'test': [2, 3, 4, 5],
    }
    valid_images: ClassVar[dict[str, list[str]]] = {
        'train': ['MS', 'PS-MS', 'PAN', 'PS-RGB'],
        'test': ['MUL', 'MUL-PanSharpen', 'PAN', 'RGB-PanSharpen'],
    }
    valid_masks: tuple[str, ...] = ('geojson_roads', 'geojson_roads_speed')
