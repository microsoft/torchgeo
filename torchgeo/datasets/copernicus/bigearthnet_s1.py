# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench BigEarthNet-S1 dataset."""

import os
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..utils import Path
from .base import CopernicusBenchBase


class CopernicusBenchBigEarthNetS1(CopernicusBenchBase):
    """Copernicus-Bench BigEarthNet-S1 dataset.

    BigEarthNet-S1 is a multilabel land use/land cover classification dataset
    composed of 5% of the Sentinel-1 data of BigEarthNet-v2.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://arxiv.org/abs/2407.03653

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l2_bigearthnet_s1s2/bigearthnetv2.zip'
    md5 = '269355db0449e0da7213c95f30c346d4'
    zipfile = 'bigearthnetv2.zip'
    directory = 'bigearthnet_s1s2'
    filename = 'multilabel-{}.csv'
    filename_regex = r'.{16}_(?P<date>\d{8}T\d{6})'
    all_bands = ('VV', 'VH')
    rgb_bands = ('VV', 'VH')
    classes = (
        'Urban fabric',
        'Industrial or commercial units',
        'Arable land',
        'Permanent crops',
        'Pastures',
        'Complex cultivation patterns',
        'Land principally occupied by agriculture, with significant areas of natural vegetation',
        'Agro-forestry areas',
        'Broad-leaved forest',
        'Coniferous forest',
        'Mixed forest',
        'Natural grassland and sparsely vegetated areas',
        'Moors, heathland and sclerophyllous vegetation',
        'Transitional woodland, shrub',
        'Beaches, dunes, sands',
        'Inland wetlands',
        'Coastal wetlands',
        'Inland waters',
        'Marine waters',
    )

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CopernicusBenchBigEarthNetS1 instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            bands: Sequence of band names to load (defaults to all bands).
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        super().__init__(root, split, bands, transforms, download, checksum)
        filepath = os.path.join(root, self.directory, self.filename.format(split))
        self.files = pd.read_csv(filepath)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        row = self.files.iloc[index].values
        file = row[0]
        path = os.path.join(self.root, self.directory, 'BigEarthNet-S1-5%', file)
        sample = self._load_image(path)
        sample['label'] = torch.tensor(row[2:].astype(np.int64))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
