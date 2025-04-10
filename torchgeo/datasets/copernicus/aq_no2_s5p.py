# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench AQ-NO2-S5P dataset."""

import os
from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import Tensor

from ..utils import Path, stack_samples
from .base import CopernicusBenchBase


class CopernicusBenchAQNO2S5P(CopernicusBenchBase):
    """Copernicus-Bench AQ-NO2-S5P dataset.

    AQ-NO2-S5P is a regression dataset based on Sentinel-5P NO2 images and
    EEA air quality data products. Specifically, this dataset combines 2021
    measurements of NO2 (annual average concentration) from EEA with S5P NO2
    ("tropospheric NO2 column number density") from GEE.

    This benchmark supports both annual (1 image/location) and seasonal
    (4 images/location) modes, the former is used in the original benchmark.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://www.researchgate.net/profile/Jan-Horalek/publication/389165501_Air_quality_maps_of_EEA_member_and_cooperating_countries_for_2022/links/67b72628207c0c20fa8ec116/Air-quality-maps-of-EEA-member-and-cooperating-countries-for-2022.pdf

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l3_airquality_s5p/airquality_s5p.zip'
    md5 = '92081c7437c5c1daf783868ad7669877'
    zipfile = 'airquality_s5p.zip'
    directory = os.path.join('airquality_s5p', 'no2')
    filename = '{}.csv'
    dtype = torch.float
    filename_regex = r'(?P<start>\d{4}-\d{2}-\d{2})_(?P<stop>\d{4}-\d{2}-\d{2})'
    date_format = '%Y-%m-%d'
    all_bands = ('NO2',)
    rgb_bands = ('NO2',)
    cmap = 'Wistia'

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        mode: Literal['annual', 'seasonal'] = 'annual',
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CopernicusBenchAQNO2S5P instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            mode: One of 'annual' or 'seasonal'.
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

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        pid = self.files[index]
        match self.mode:
            case 'annual':
                file = '2021-01-01_2021-12-31.tif'
                path = os.path.join(self.root, self.directory, 's5p_annual', pid, file)
                sample = self._load_image(path)
            case 'seasonal':
                files = [
                    '2021-01-01_2021-04-01.tif',
                    '2021-04-01_2021-07-01.tif',
                    '2021-07-01_2021-10-01.tif',
                    '2021-10-01_2021-12-31.tif',
                ]
                root = os.path.join(self.root, self.directory, 's5p_seasonal', pid)
                samples = [self._load_image(os.path.join(root, file)) for file in files]
                sample = stack_samples(samples)

        path = os.path.join(self.root, self.directory, 'label_annual', f'{pid}.tif')
        sample |= self._load_mask(path)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
