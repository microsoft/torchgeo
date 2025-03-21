# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench BigEarthNet-S1 dataset."""

import os
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor

from ..errors import RGBBandsMissingError
from ..utils import Path, percentile_normalization
from .base import CopernicusBenchBase


class CopernicusBenchBigEarthNetS1(CopernicusBenchBase):
    """Copernicus-Bench BigEarthNet-S1 dataset.

    BigEarthNet-S1 is a multilabel land use/land cover classification dataset
    composed of 5% of the data of BigEarthNet-v2.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://arxiv.org/abs/2407.03653

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/dcd0b3a45165251e8f5944b040a3411d5e6466a5/l2_bigearthnet_s1s2/bigearthnetv2.zip'
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
        file = self.files.iloc[index, 0]
        path = os.path.join(self.root, self.directory, 'BigEarthNet-S1-5%', file)
        sample = self._load_image(path)
        sample['label'] = torch.tensor(self.files.iloc[index, 2:])

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        fig, ax = plt.subplots()

        image = sample['image'][rgb_indices].numpy()
        image = np.stack([image[0], image[1], (image[0] + image[1]) / 2])
        image = rearrange(image, 'c h w -> h w c')
        image = percentile_normalization(image)
        ax.imshow(image)
        ax.axis('off')

        if show_titles:
            title = 'Label: ' + str(sample['label'].numpy())
            if 'prediction' in sample:
                title += '\nPrediction: ' + str(sample['prediction'].numpy())
            ax.set_title(title)

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
