# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet 4 dataset."""

import os
from typing import ClassVar

from .base import SpaceNet


class SpaceNet4(SpaceNet):
    """SpaceNet 4: Off-Nadir Buildings Dataset.

    `SpaceNet 4 <https://spacenet.ai/off-nadir-building-detection/>`_ is a
    dataset of 27 WV-2 imagery captured at varying off-nadir angles and
    associated building footprints over the city of Atlanta. The off-nadir angle
    ranges from 7 degrees to 54 degrees.

    Dataset features:

    * No. of chipped images: 28,728 (PAN/MS/PS-RGBNIR)
    * No. of label files: 1064
    * No. of building footprints: >120,000
    * Area Coverage: 665 sq km
    * Chip size: 225 x 225 (MS), 900 x 900 (PAN/PS-RGBNIR)

    Dataset format:

    * Imagery - Worldview-2 GeoTIFFs

        * PAN.tif (Panchromatic)
        * MS.tif (Multispectral)
        * PS-RGBNIR (Pansharpened RGBNIR)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1903.12239
    """

    directory_glob = os.path.join('**', '{product}')
    file_regex = r'_(\d+_\d+)\.'
    dataset_id = 'SN4_buildings'
    tarballs: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {
            6: [
                'Atlanta_nadir7_catid_1030010003D22F00.tar.gz',
                'Atlanta_nadir8_catid_10300100023BC100.tar.gz',
                'Atlanta_nadir10_catid_1030010003993E00.tar.gz',
                'Atlanta_nadir10_catid_1030010003CAF100.tar.gz',
                'Atlanta_nadir13_catid_1030010002B7D800.tar.gz',
                'Atlanta_nadir14_catid_10300100039AB000.tar.gz',
                'Atlanta_nadir16_catid_1030010002649200.tar.gz',
                'Atlanta_nadir19_catid_1030010003C92000.tar.gz',
                'Atlanta_nadir21_catid_1030010003127500.tar.gz',
                'Atlanta_nadir23_catid_103001000352C200.tar.gz',
                'Atlanta_nadir25_catid_103001000307D800.tar.gz',
                'Atlanta_nadir27_catid_1030010003472200.tar.gz',
                'Atlanta_nadir29_catid_1030010003315300.tar.gz',
                'Atlanta_nadir30_catid_10300100036D5200.tar.gz',
                'Atlanta_nadir32_catid_103001000392F600.tar.gz',
                'Atlanta_nadir34_catid_1030010003697400.tar.gz',
                'Atlanta_nadir36_catid_1030010003895500.tar.gz',
                'Atlanta_nadir39_catid_1030010003832800.tar.gz',
                'Atlanta_nadir42_catid_10300100035D1B00.tar.gz',
                'Atlanta_nadir44_catid_1030010003CCD700.tar.gz',
                'Atlanta_nadir46_catid_1030010003713C00.tar.gz',
                'Atlanta_nadir47_catid_10300100033C5200.tar.gz',
                'Atlanta_nadir49_catid_1030010003492700.tar.gz',
                'Atlanta_nadir50_catid_10300100039E6200.tar.gz',
                'Atlanta_nadir52_catid_1030010003BDDC00.tar.gz',
                'Atlanta_nadir53_catid_1030010003193D00.tar.gz',
                'Atlanta_nadir53_catid_1030010003CD4300.tar.gz',
                'geojson.tar.gz',
            ]
        },
        'test': {6: ['SN4_buildings_AOI_6_Atlanta_test_public.tar.gz']},
    }
    md5s: ClassVar[dict[str, dict[int, list[str]]]] = {
        'train': {
            6: [
                'd41ab6ec087b07e1e046c55d1fa5754b',
                '72f04a7c0c34dd4595c181ee1ae6cb4c',
                '89559f42ac11a8de570cef9802a577ad',
                '5489ac756249c336ea506ef0acb3c09d',
                'bd9ed231cedd8631683ea51ea0602de1',
                'c497a8a448ed7ccdf63e7706507c0603',
                '45d54eeecefdc60aa38320be6f29a17c',
                '611528c0188bbc7e9cdf98609c6b0c49',
                '532fbf1ca73d3d2e8b03c585f61b7316',
                '538f48429b0968b6cfad97eb61fa8de1',
                '3c48e94bc6d9e66e27c3a9bc8d35d65d',
                'b78cdf951e7bf4fedbe9259abd1e047a',
                'f307ce3c623d12d5a2fd5acb1e0607e0',
                '9a17574332cd5513d68a0bcc9c607bdd',
                'fe905ca809f7bd2ceef75bde23c326f3',
                'd9f2e4a5c8462f6f9f7d5c573d9a1dc6',
                'f9425ff38dc82bf0e8f25a6287ff1ad1',
                '7a6005d6fd972d5ce04caf9b42b36897',
                '7c5aa16bb64cacf766cf88f89b3093bd',
                '8f7e959eb0156ad2dfb0b966a1de06a9',
                '62c4babcbe70034b7deb7c14d5ff61c2',
                '8001d75f67534edf6932242324b8c1a7',
                'bc299cb5de432b5f5a1ce65a3bdb0abc',
                'd7640eda7c4efaf825665e853037bec9',
                'd4e1931551e9d3c6fd9bf1d8adfd07a0',
                'b313e23ead8fe6e2c8671a49f2c9de37',
                '3bd8f07ad57bff841d0cf91c91c6f5ed',
                '2556339e26a09e57559452eb240ef29c',
            ]
        },
        'test': {6: ['0ec3874bfc19aed63b33ac47b039aace']},
    }
    valid_aois: ClassVar[dict[str, list[int]]] = {'train': [6], 'test': [6]}
    valid_images: ClassVar[dict[str, list[str]]] = {
        'train': ['MS', 'PAN', 'Pan-Sharpen'],
        'test': ['MS', 'PAN', 'Pan-Sharpen'],
    }
    valid_masks = (os.path.join('geojson', 'spacenet-buildings'),)
