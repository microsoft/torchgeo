# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench Cloud-S2 dataset."""

import os

import numpy as np
import rasterio as rio
import torch
from torch import Tensor

from .base import CopernicusBenchBase


class CopernicusBenchCloudS2(CopernicusBenchBase):
    """Copernicus-Bench Cloud-S2 dataset.

    Cloud-S2 is a multi-class cloud segmentation dataset derived from
    `CloudSEN12+ <https://www.sciencedirect.com/science/article/pii/S2352340924008163>`_,
    one of the largest Sentinel-2 cloud and cloud shadow detection datasets with
    expert-labeled pixels. We take 25% samples with high-quality labels, and split them
    into 1699/567/551 train/val/test subsets.

    .. list-table:: Classes
       :header-rows: 1

       * - Code
         - Class
         - Description
       * - 0
         - Clear
         - Pixels without cloud and cloud shadow contamination.
       * - 1
         - Thick Cloud
         - Opaque clouds that block all the reflected light from the Earth's surface.
       * - 2
         - Thin Cloud
         - Semitransparent clouds that alter the surface spectral signal but still allow
           to recognize the background. This is the hardest class to identify.
       * - 3
         - Cloud Shadow
         - Dark pixels where light is occluded by thick or thin clouds.

    If you use this dataset in your research, please cite the following papers:

    * TODO

    .. versionadded:: 0.7
    """

    url = 'https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/86342afa2409e49d80688fe00c05201c0f46569b/l1_cloud_s2/cloud_s2.zip'
    md5 = '39a1f966e76455549a3e6c209ba751c1'
    zipfile = 'cloud_s2.zip'
    directory = 'cloud_s2'
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

    def _load_image(self, index: int) -> dict[str, Tensor]:
        """Load an image.

        Args:
            index: Index to return.

        Returns:
            An image sample.
        """
        sample: dict[str, Tensor] = {}
        file = self.files[index] + '.tif'
        with rio.open(os.path.join(self.root, self.directory, 's2_toa', file)) as f:
            sample['image'] = torch.tensor(f.read(self.band_indices).astype(np.float32))

        return sample

    def _load_target(self, index: int) -> dict[str, Tensor]:
        """Load a target mask.

        Args:
            index: Index to return.

        Returns:
            A target sample.
        """
        sample: dict[str, Tensor] = {}
        file = self.files[index] + '.tif'
        with rio.open(os.path.join(self.root, self.directory, 'cloud', file)) as f:
            sample['mask'] = torch.tensor(f.read(1).astype(np.int64))

        return sample
