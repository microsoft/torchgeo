# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench EuroSAT-S1 dataset."""

import os

import torch
from torch import Tensor

from .base import CopernicusBenchBase


class CopernicusBenchEuroSATS1(CopernicusBenchBase):
    """Copernicus-Bench EuroSAT-S1 dataset.

    EuroSAT-S1 is a multi-class land use/land cover classification dataset,
    and is functionally identical to EuroSAT-SAR.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://doi.org/10.1109/JSTARS.2024.3493237

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l2_eurosat_s1s2/eurosat_s1.zip'
    md5 = 'e7e7f8fc68fc55a7a689cb654912ff3f'
    zipfile = 'eurosat_s1.zip'
    directory = 'eurosat_s1'
    filename = 'eurosat-{}.txt'
    all_bands = ('VV', 'VH')
    rgb_bands = ('VV', 'VH')
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
