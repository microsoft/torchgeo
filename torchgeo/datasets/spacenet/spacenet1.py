# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet 1 dataset."""

from typing import ClassVar

from .base import SpaceNet


class SpaceNet1(SpaceNet):
    """SpaceNet 1: Building Detection v1 Dataset.

    `SpaceNet 1 <https://spacenet.ai/spacenet-buildings-dataset-v1/>`_
    is a dataset of building footprints over the city of Rio de Janeiro.

    Dataset features:

    * No. of images: 6940 (8 Band) + 6940 (RGB)
    * No. of polygons: 382,534 building labels
    * Area Coverage: 2544 sq km
    * GSD: 1 m (8 band),  50 cm (rgb)
    * Chip size: 101 x 110 (8 band), 406 x 438 (rgb)

    Dataset format:

    * Imagery - Worldview-2 GeoTIFFs

        * 8Band.tif (Multispectral)
        * RGB.tif (Pansharpened RGB)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232
    """

    directory_glob = '{product}'
    dataset_id = 'SN1_buildings'
    tarballs: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {
            1: [
                'SN1_buildings_train_AOI_1_Rio_3band.tar.gz',
                'SN1_buildings_train_AOI_1_Rio_8band.tar.gz',
                'SN1_buildings_train_AOI_1_Rio_geojson_buildings.tar.gz',
            ]
        },
        'test': {
            1: [
                'SN1_buildings_test_AOI_1_Rio_3band.tar.gz',
                'SN1_buildings_test_AOI_1_Rio_8band.tar.gz',
            ]
        },
    }
    md5s: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {
            1: [
                '279e334a2120ecac70439ea246174516',
                '6440a9eedbd7c4fe9741875135362c8c',
                'b6e02fbd727f252ea038abe4f77a77b3',
            ]
        },
        'test': {
            1: ['18283d78b21c239bc1831f3bf1d2c996', '732b3a40603b76e80aac84e002e2b3e8']
        },
    }
    valid_aois: ClassVar[dict[str, list[int]]] = {'train': [1], 'test': [1]}
    valid_images: ClassVar[dict[str, list[str]]] = {
        'train': ['3band', '8band'],
        'test': ['3band', '8band'],
    }
    valid_masks = ('geojson',)
    chip_size: ClassVar[dict[str, tuple[int, int]]] = {
        '3band': (406, 439),
        '8band': (102, 110),
    }
