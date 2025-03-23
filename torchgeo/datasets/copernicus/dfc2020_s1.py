# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench DFC2020-S1 dataset."""

import os

import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from torch import Tensor

from ..errors import RGBBandsMissingError
from ..utils import percentile_normalization
from .base import CopernicusBenchBase


class CopernicusBenchDFC2020S1(CopernicusBenchBase):
    """Copernicus-Bench DFC2020-S1 dataset.

    DFC2020-S1 is a land use/land cover segmentation datasets derived from the
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
    all_bands = ('VV', 'VH')
    rgb_bands = ('VV', 'VH')
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
            self.root, self.directory, 's1', file.replace('dfc', 's1')
        )
        mask_path = os.path.join(self.root, self.directory, 'dfc', file)
        sample = self._load_image(image_path) | self._load_mask(mask_path)

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

        ncols = 3 if 'prediction' in sample else 2
        fig, ax = plt.subplots(ncols=ncols)

        image = sample['image'][rgb_indices].numpy()
        image = np.stack([image[0], image[1], (image[0] + image[1]) / 2])
        image = rearrange(image, 'c h w -> h w c')
        image = percentile_normalization(image)
        ax[0].imshow(image)
        ax[0].axis('off')

        kwargs = {'cmap': self.cmap, 'vmin': 0, 'vmax': 10, 'interpolation': 'none'}
        mask = sample['mask']
        ax[1].imshow(mask, **kwargs)
        ax[1].axis('off')

        if 'prediction' in sample:
            prediction = sample['prediction']
            ax[2].imshow(prediction, **kwargs)
            ax[2].axis('off')

        if show_titles:
            ax[0].set_title('Image')
            ax[1].set_title('Mask')
            if 'prediction' in sample:
                ax[2].set_title('Prediction')

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
