# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench EuroSAT-S2 dataset."""

import os

import torch
from torch import Tensor

from .base import CopernicusBenchBase


class CopernicusBenchEuroSATS2(CopernicusBenchBase):
    """Copernicus-Bench EuroSAT-S2 dataset.

    EuroSAT-S2 is a multi-class land use/land cover classification dataset,
    and is functionally identical to EuroSAT-MS.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://ieeexplore.ieee.org/document/8736785
    * https://ieeexplore.ieee.org/document/8519248

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l2_eurosat_s1s2/eurosat_s2.zip'
    md5 = 'b2be02ca9767554c717f2e9bd15bbd23'
    zipfile = 'eurosat_s2.zip'
    directory = 'eurosat_s2'
    filename = 'eurosat-{}.txt'
    all_bands = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B09',
        'B10',
        'B11',
        'B12',
        'B8A',
    )
    rgb_bands = ('B04', 'B03', 'B02')
    classes = (
        'AnnualCrop',
        'HerbaceousVegetation',
        'Industrial',
        'PermanentCrop',
        'River',
        'Forest',
        'Highway',
        'Pasture',
        'Residential',
        'SeaLake',
    )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        file = self.files[index].replace('.jpg', '.tif')
        classname = file.split('_')[0]
        path = os.path.join(self.root, self.directory, 'all_imgs', classname, file)
        sample = self._load_image(path)
        sample['label'] = torch.tensor(self.classes.index(classname))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
