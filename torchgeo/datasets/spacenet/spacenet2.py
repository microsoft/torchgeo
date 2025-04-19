# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet 2 dataset."""

import os
from typing import ClassVar

from .base import SpaceNet


class SpaceNet2(SpaceNet):
    r"""SpaceNet 2: Building Detection v2 Dataset.

    `SpaceNet 2 <https://spacenet.ai/spacenet-buildings-dataset-v2/>`_
    is a dataset of building footprints over the cities of Las Vegas,
    Paris, Shanghai and Khartoum.

    Collection features:

    +------------+---------------------+------------+------------+
    |    AOI     | Area (km\ :sup:`2`\)| # Images   | # Buildings|
    +============+=====================+============+============+
    | Las Vegas  |    216              |   3850     |  151,367   |
    +------------+---------------------+------------+------------+
    | Paris      |    1030             |   1148     |  23,816    |
    +------------+---------------------+------------+------------+
    | Shanghai   |    1000             |   4582     |  92,015    |
    +------------+---------------------+------------+------------+
    | Khartoum   |    765              |   1012     |  35,503    |
    +------------+---------------------+------------+------------+

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
            - 650 x 650
            - 163 x 163
            - 650 x 650
            - 650 x 650

    Dataset format:

    * Imagery - Worldview-3 GeoTIFFs

        * PAN.tif (Panchromatic)
        * MS.tif (Multispectral)
        * PS-MS (Pansharpened Multispectral)
        * PS-RGB (Pansharpened RGB)

    * Labels - GeoJSON

        * label.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232
    """

    dataset_id = 'SN2_buildings'
    tarballs: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {
            2: ['SN2_buildings_train_AOI_2_Vegas.tar.gz'],
            3: ['SN2_buildings_train_AOI_3_Paris.tar.gz'],
            4: ['SN2_buildings_train_AOI_4_Shanghai.tar.gz'],
            5: ['SN2_buildings_train_AOI_5_Khartoum.tar.gz'],
        },
        'test': {
            2: ['AOI_2_Vegas_Test_public.tar.gz'],
            3: ['AOI_3_Paris_Test_public.tar.gz'],
            4: ['AOI_4_Shanghai_Test_public.tar.gz'],
            5: ['AOI_5_Khartoum_Test_public.tar.gz'],
        },
    }
    md5s: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {
            2: ['307da318bc43aaf9481828f92eda9126'],
            3: ['4db469e3e4e7bf025368ad730aec0888'],
            4: ['986129eecd3e842ebc2063d43b407adb'],
            5: ['462b4bf0466c945d708befabd4d9115b'],
        },
        'test': {
            2: ['d45405afd6629e693e2f9168b1291ea3'],
            3: ['2eaee95303e88479246e4ee2f2279b7f'],
            4: ['f51dc51fa484dc7fb89b3697bd15a950'],
            5: ['037d7be10530f0dd1c43d4ef79f3236e'],
        },
    }
    valid_aois: ClassVar[dict[str, list[int]]] = {
        'train': [2, 3, 4, 5],
        'test': [2, 3, 4, 5],
    }
    valid_images: ClassVar[dict[str, list[str]]] = {
        'train': ['MUL', 'MUL-PanSharpen', 'PAN', 'RGB-PanSharpen'],
        'test': ['MUL', 'MUL-PanSharpen', 'PAN', 'RGB-PanSharpen'],
    }
    valid_masks = (os.path.join('geojson', 'buildings'),)
    chip_size: ClassVar[dict[str, tuple[int, int]]] = {'MUL': (163, 163)}
