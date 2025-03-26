# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet 7 dataset."""

import os
from typing import ClassVar

from .base import SpaceNet


class SpaceNet7(SpaceNet):
    """SpaceNet 7: Multi-Temporal Urban Development Challenge.

    `SpaceNet 7 <https://spacenet.ai/sn7-challenge/>`_ is a dataset which
    consist of medium resolution (4.0m) satellite imagery mosaics acquired from
    Planet Labs' Dove constellation between 2017 and 2020. It includes ≈ 24
    images (one per month) covering > 100 unique geographies, and comprises >
    40,000 km2 of imagery and exhaustive polygon labels of building footprints
    therein, totaling over 11M individual annotations.

    Dataset features:

    * No. of train samples: 1423
    * No. of test samples: 466
    * No. of building footprints: 11,080,000
    * Area Coverage: 41,000 sq km
    * Chip size: 1024 x 1024
    * GSD: ~4m

    Dataset format:

    * Imagery - Planet Dove GeoTIFF

        * mosaic.tif

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2102.04420

    .. versionadded:: 0.2
    """

    directory_glob = os.path.join('**', '{product}')
    mask_glob = '*_Buildings.geojson'
    file_regex = r'global_monthly_(\d+.*\d+)'
    dataset_id = 'SN7_buildings'
    tarballs: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {0: ['SN7_buildings_train.tar.gz']},
        'test': {0: ['SN7_buildings_test_public.tar.gz']},
    }
    md5s: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {0: ['6eda13b9c28f6f5cdf00a7e8e218c1b1']},
        'test': {0: ['b3bde95a0f8f32f3bfeba49464b9bc97']},
    }
    valid_aois: ClassVar[dict[str, list[int]]] = {'train': [0], 'test': [0]}
    valid_images: ClassVar[dict[str, list[str]]] = {
        'train': ['images', 'images_masked'],
        'test': ['images_masked'],
    }
    valid_masks = ('labels', 'labels_match', 'labels_match_pix')
    chip_size: ClassVar[dict[str, tuple[int, int]]] = {
        'images': (1024, 1024),
        'images_masked': (1024, 1024),
    }
