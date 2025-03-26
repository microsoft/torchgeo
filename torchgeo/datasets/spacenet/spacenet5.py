# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet 5 dataset."""

from typing import ClassVar

from .spacenet3 import SpaceNet3


class SpaceNet5(SpaceNet3):
    r"""SpaceNet 5: Automated Road Network Extraction and Route Travel Time Estimation.

    `SpaceNet 5 <https://spacenet.ai/sn5-challenge/>`_
    is a dataset of road networks over the cities of Moscow, Mumbai and San
    Juan (unavailable).

    Collection features:

    +------------+---------------------+------------+---------------------------+
    |    AOI     | Area (km\ :sup:`2`\)| # Images   | # Road Network Labels (km)|
    +============+=====================+============+===========================+
    | Moscow     |    1353             |   1353     |         3066              |
    +------------+---------------------+------------+---------------------------+
    | Mumbai     |    1021             |   1016     |         1951              |
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

    If you use this dataset in your research, please use the following citation:

    * The SpaceNet Partners, “SpaceNet5: Automated Road Network Extraction and
      Route Travel Time Estimation from Satellite Imagery”,
      https://spacenet.ai/sn5-challenge/

    .. versionadded:: 0.2
    """

    file_regex = r'_chip(\d+)\.'
    dataset_id = 'SN5_roads'
    tarballs: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {
            7: ['SN5_roads_train_AOI_7_Moscow.tar.gz'],
            8: ['SN5_roads_train_AOI_8_Mumbai.tar.gz'],
        },
        'test': {9: ['SN5_roads_test_public_AOI_9_San_Juan.tar.gz']},
    }
    md5s: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {
            7: ['03082d01081a6d8df2bc5a9645148d2a'],
            8: ['1ee20ba781da6cb7696eef9a95a5bdcc'],
        },
        'test': {9: ['fc45afef219dfd3a20f2d4fc597f6882']},
    }
    valid_aois: ClassVar[dict[str, list[int]]] = {'train': [7, 8], 'test': [9]}
    valid_images: ClassVar[dict[str, list[str]]] = {
        'train': ['MS', 'PAN', 'PS-MS', 'PS-RGB'],
        'test': ['MS', 'PAN', 'PS-MS', 'PS-RGB'],
    }
    valid_masks = ('geojson_roads_speed',)
