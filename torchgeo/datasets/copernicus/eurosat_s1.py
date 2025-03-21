# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench EuroSAT-S1 dataset."""

import os

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor

from ..errors import RGBBandsMissingError
from ..utils import percentile_normalization
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

    url = 'https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/8c294253fa04f8a0cac0f4249850fdd652e43ec2/l2_eurosat_s1s2/eurosat_s1.zip'
    md5 = 'e7e7f8fc68fc55a7a689cb654912ff3f'
    zipfile = 'eurosat_s1.zip'
    directory = 'eurosat_s1'
    filename = 'eurosat-{}.csv'
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
            title = 'Label: ' + self.classes[sample['label']]
            if 'prediction' in sample:
                title += '\nPrediction: ' + self.classes[sample['prediction']]
            ax.set_title(title)

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
