# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench LC100Cls-S3 dataset."""

import glob
import os
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..utils import Path, stack_samples
from .base import CopernicusBenchBase


class CopernicusBenchLC100ClsS3(CopernicusBenchBase):
    """Copernicus-Bench LC100Cls-S3 dataset.

    LC100Cls-S3 is a multilabel land use/land cover classification dataset based on
    Sentinel-3 OLCI images and CGLS-LC100 land cover maps. CGLS-LC100 is a product in
    the Copernicus Global Land Service (CGLS) portfolio and delivers a global 23-class
    land cover map at 100 m spatial resolution.

    This benchmark supports both static (1 image/location) and time series
    (1-4 images/location) modes, the former is used in the original benchmark.

    .. list-table:: Classes
       :header-rows: 1

       * - Value
         - Description
       * - 0
         - Unknown. No or not enough satellite data available.
       * - 20
         - Shrubs. Woody perennial plants with persistent and woody stems and without
           any defined main stem being less than 5 m tall. The shrub foliage can be
           either evergreen or deciduous.
       * - 30
         - Herbaceous vegetation. Plants without persistent stem or shoots above ground
           and lacking definite firm structure. Tree and shrub cover is less than 10 %.
       * - 40
         - Cultivated and managed vegetation / agriculture. Lands covered with temporary
           crops followed by harvest and a bare soil period (e.g., single and multiple
           cropping systems). Note that perennial woody crops will be classified as the
           appropriate forest or shrub land cover type.
       * - 50
         - Urban / built up. Land covered by buildings and other man-made structures.
       * - 60
         - Bare / sparse vegetation. Lands with exposed soil, sand, or rocks and never
           has more than 10 % vegetated cover during any time of the year.
       * - 70
         - Snow and ice. Lands under snow or ice cover throughout the year.
       * - 80
         - Permanent water bodies. Lakes, reservoirs, and rivers. Can be either fresh
           or salt-water bodies.
       * - 90
         - Herbaceous wetland. Lands with a permanent mixture of water and herbaceous
           or woody vegetation. The vegetation can be present in either salt, brackish,
           or fresh water.
       * - 100
         - Moss and lichen.
       * - 111
         - Closed forest, evergreen needle leaf. Tree canopy >70 %, almost all needle
           leaf trees remain green all year. Canopy is never without green foliage.
       * - 112
         - Closed forest, evergreen broad leaf. Tree canopy >70 %, almost all broadleaf
           trees remain green year round. Canopy is never without green foliage.
       * - 113
         - Closed forest, deciduous needle leaf. Tree canopy >70 %, consists of seasonal
           needle leaf tree communities with an annual cycle of leaf-on and leaf-off
           periods.
       * - 114
         - Closed forest, deciduous broad leaf. Tree canopy >70 %, consists of seasonal
           broadleaf tree communities with an annual cycle of leaf-on and leaf-off
           periods.
       * - 115
         - Closed forest, mixed.
       * - 116
         - Closed forest, not matching any of the other definitions.
       * - 121
         - Open forest, evergreen needle leaf. Top layer- trees 15-70 % and second
           layer- mixed of shrubs and grassland, almost all needle leaf trees remain
           green all year. Canopy is never without green foliage.
       * - 122
         - Open forest, evergreen broad leaf. Top layer- trees 15-70 % and second layer-
           mixed of shrubs and grassland, almost all broadleaf trees remain green year
           round. Canopy is never without green foliage.
       * - 123
         - Open forest, deciduous needle leaf. Top layer- trees 15-70 % and second
           layer- mixed of shrubs and grassland, consists of seasonal needle leaf tree
           communities with an annual cycle of leaf-on and leaf-off periods.
       * - 124
         - Open forest, deciduous broad leaf. Top layer- trees 15-70 % and second layer-
           mixed of shrubs and grassland, consists of seasonal broadleaf tree
           communities with an annual cycle of leaf-on and leaf-off periods.
       * - 125
         - Open forest, mixed.
       * - 126
         - Open forest, not matching any of the other definitions.
       * - 200
         - Oceans, seas. Can be either fresh or salt-water bodies.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://doi.org/10.5281/zenodo.3939049

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l2_lc100_s3/lc100_s3.zip'
    md5 = '967d1da6286e0d0e346e425a8f3800e9'
    zipfile = 'lc100_s3.zip'
    filename = 'multilabel-{}.csv'
    directory = 'lc100_s3'
    filename_regex = r'S3[AB]_(?P<date>\d{8}T\d{6})'
    all_bands = (
        'Oa01_radiance',
        'Oa02_radiance',
        'Oa03_radiance',
        'Oa04_radiance',
        'Oa05_radiance',
        'Oa06_radiance',
        'Oa07_radiance',
        'Oa08_radiance',
        'Oa09_radiance',
        'Oa10_radiance',
        'Oa11_radiance',
        'Oa12_radiance',
        'Oa13_radiance',
        'Oa14_radiance',
        'Oa15_radiance',
        'Oa16_radiance',
        'Oa17_radiance',
        'Oa18_radiance',
        'Oa19_radiance',
        'Oa20_radiance',
        'Oa21_radiance',
    )
    rgb_bands = ('Oa08_radiance', 'Oa06_radiance', 'Oa04_radiance')
    classes = (
        'Unknown',
        'Shrubs',
        'Herbaceous vegetation',
        'Cultivated and managed vegetation / agriculture',
        'Urban / built up',
        'Bare / sparse vegetation',
        'Snow and ice',
        'Permanent water bodies',
        'Herbaceous wetland',
        'Moss and lichen',
        'Closed forest, evergreen needle leaf',
        'Closed forest, evergreen broad leaf',
        'Closed forest, deciduous needle leaf',
        'Closed forest, deciduous broad leaf',
        'Closed forest, mixed',
        'Closed forest, not matching any of the other definitions',
        'Open forest, evergreen needle leaf',
        'Open forest, evergreen broad leaf',
        'Open forest, deciduous needle leaf',
        'Open forest, deciduous broad leaf',
        'Open forest, mixed',
        'Open forest, not matching any of the other definitions',
        'Oceans, seas',
    )

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        mode: Literal['static', 'time-series'] = 'static',
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CopernicusBenchLC100ClsS3 instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            mode: One of 'static' or 'time-series'.
            bands: Sequence of band names to load (defaults to all bands).
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.mode = mode
        super().__init__(root, split, bands, transforms, download, checksum)
        filepath = os.path.join(root, self.directory, self.filename.format(split))
        self.files = pd.read_csv(filepath)
        if mode == 'static':
            filepath = os.path.join(root, self.directory, f'static_fnames-{split}.csv')
            self.static_files = pd.read_csv(filepath, header=None)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        row = self.files.iloc[index].values
        match self.mode:
            case 'static':
                pid, file = self.static_files.iloc[index]
                path = os.path.join(self.root, self.directory, 's3_olci', pid, file)
                sample = self._load_image(path)
                sample['label'] = torch.tensor(row[1:].astype(np.int64))
            case 'time-series':
                pid = row[0]
                paths = os.path.join(self.root, self.directory, 's3_olci', pid, '*.tif')
                samples = [self._load_image(path) for path in sorted(glob.glob(paths))]
                sample = stack_samples(samples)
                sample['label'] = torch.tensor(row[1:].astype(np.int64))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
