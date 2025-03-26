# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet 6 dataset."""

from typing import ClassVar

from .base import SpaceNet


class SpaceNet6(SpaceNet):
    r"""SpaceNet 6: Multi-Sensor All-Weather Mapping.

    `SpaceNet 6 <https://spacenet.ai/sn6-challenge/>`_ is a dataset
    of optical and SAR imagery over the city of Rotterdam.

    Collection features:

    +------------+---------------------+------------+-----------------------------+
    |    AOI     | Area (km\ :sup:`2`\)| # Images   | # Building Footprint Labels |
    +============+=====================+============+=============================+
    | Rotterdam  |    120              |   3401     |         48000               |
    +------------+---------------------+------------+-----------------------------+


    Imagery features:

    .. list-table::
        :widths: 10 10 10 10 10 10
        :header-rows: 1
        :stub-columns: 1

        *   -
            - PAN
            - RGBNIR
            - PS-RGB
            - PS-RGBNIR
            - SAR-Intensity
        *   - GSD (m)
            - 0.5
            - 2.0
            - 0.5
            - 0.5
            - 0.5
        *   - Chip size (px)
            - 900 x 900
            - 450 x 450
            - 900 x 900
            - 900 x 900
            - 900 x 900


    Dataset format:

    * Imagery - GeoTIFFs from Worldview-2 (optical) and Capella Space (SAR)

        * PAN.tif (Panchromatic)
        * RGBNIR.tif (Multispectral)
        * PS-RGB (Pansharpened RGB)
        * PS-RGBNIR (Pansharpened RGBNIR)
        * SAR-Intensity (SAR Intensity)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2004.06500

    .. versionadded:: 0.4
    """

    file_regex = r'_tile_(\d+)\.'
    dataset_id = 'SN6_buildings'
    tarballs: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {11: ['SN6_buildings_AOI_11_Rotterdam_train.tar.gz']},
        'test': {11: ['SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz']},
    }
    md5s: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {11: ['10ca26d2287716e3b6ef0cf0ad9f946e']},
        'test': {11: ['a07823a5e536feeb8bb6b6f0cb43cf05']},
    }
    valid_aois: ClassVar[dict[str, list[int]]] = {'train': [11], 'test': [11]}
    valid_images: ClassVar[dict[str, list[str]]] = {
        'train': ['PAN', 'PS-RGB', 'PS-RGBNIR', 'RGBNIR', 'SAR-Intensity'],
        'test': ['SAR-Intensity'],
    }
    valid_masks = ('geojson_buildings',)
