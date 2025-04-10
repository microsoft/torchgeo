# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench DFC2020-S2 dataset."""

import os

from matplotlib.colors import ListedColormap
from torch import Tensor

from .base import CopernicusBenchBase


class CopernicusBenchDFC2020S2(CopernicusBenchBase):
    """Copernicus-Bench DFC2020-S2 dataset.

    DFC2020-S2 is a land use/land cover segmentation datasets derived from the
    IEEE GRSS Data Fusion Contest 2020 (DFC2020).

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l2_dfc2020_s1s2/dfc2020.zip'
    md5 = 'f10ba017dab6f38b7a6857b169ea924b'
    zipfile = 'dfc2020.zip'
    directory = 'dfc2020_s1s2'
    filename = 'dfc-{}-new.csv'
    all_bands = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B8A',
        'B09',
        'B10',
        'B11',
        'B12',
    )
    rgb_bands = ('B04', 'B03', 'B02')
    classes = (
        'Background',
        'Forest',
        'Shrubland',
        'Savanna',
        'Grassland',
        'Wetlands',
        'Croplands',
        'Urban/Built-up',
        'Snow/Ice',
        'Barren',
        'Water',
    )
    cmap = ListedColormap(
        [
            '#000000',
            '#009900',
            '#c6b044',
            '#fbff13',
            '#b6ff05',
            '#27ff87',
            '#c24f44',
            '#a5a5a5',
            '#69fff8',
            '#f9ffa4',
            '#1c0dff',
        ]
    )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        file = self.files[index]
        image_path = os.path.join(
            self.root, self.directory, 's2', file.replace('dfc', 's2')
        )
        mask_path = os.path.join(self.root, self.directory, 'dfc', file)
        sample = self._load_image(image_path) | self._load_mask(mask_path)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
