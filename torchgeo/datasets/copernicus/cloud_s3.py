# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench Cloud-S3 dataset."""

import os
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import rasterio as rio
import torch
from torch import Tensor

from ..utils import Path
from .base import CopernicusBenchBase


class CopernicusBenchCloudS3(CopernicusBenchBase):
    """Copernicus-Bench Cloud-S3 dataset.

    Cloud-S3 is a cloud segmentation dataset with raw images from Sentinel-3 OLCI
    and labels from the
    `IdePix <https://step.esa.int/main/snap-supported-plugins/idepix-tool/>`__
    classification algorithm.

    This dataset has two modes:

    .. list-table:: Multiclass Classification
       :header-rows: 1

       * - Code
         - Class
         - Description
       * - 255
         - Invalid
         - Invalid pixels, should be ignored during training.
       * - 0
         - Clear
         - Land, coastline, or water pixels.
       * - 1
         - Cloud-Sure
         - Fully-opaque clouds with full confidence of their detection.
       * - 2
         - Cloud-Ambiguous
         - Semi-transparent clouds, or clouds where the detection level is uncertain.
       * - 3
         - Cloud Shadow
         - Pixels are affected by a cloud shadow.
       * - 4
         - Snow/Ice
         - Clear snow/ice pixels.

    .. list-table:: Binary Classification
       :header-rows: 1

       * - Code
         - Class
         - Description
       * - 255
         - Invalid
         - Invalid pixels, should be ignored during training.
       * - 0
         - Clear
         - Land, coastline, water, snow, or ice pixels.
       * - 1
         - Cloud
         - Pixels which are either cloud-sure or cloud-ambiguous.

    If you use this dataset in your research, please cite the following paper:

    * TODO

    .. versionadded:: 0.7
    """

    url = 'https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/86342afa2409e49d80688fe00c05201c0f46569b/l1_cloud_s3/cloud_s3.zip'
    md5 = '1f82a8ccf16a0c44f0b1729e523e343a'
    zipfile = 'cloud_s3.zip'
    directory = 'cloud_s3'
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
    classes: tuple[str, ...] = (
        'Clear',
        'Cloud-Sure',
        'Cloud-Ambiguous',
        'Cloud Shadow',
        'Snow/Ice',
    )

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        mode: Literal['binary', 'multi'] = 'multi',
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CopernicusBenchBase instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            mode: One of 'binary' or 'multi'.
            bands: Sequence of band names to load (defaults to all bands).
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.mode = mode
        self.classes = ('Clear', 'Cloud') if mode == 'binary' else self.classes
        super().__init__(root, split, bands, transforms, download, checksum)

    def _load_image(self, index: int) -> dict[str, Tensor]:
        """Load an image.

        Args:
            index: Index to return.

        Returns:
            An image sample.
        """
        sample: dict[str, Tensor] = {}
        file = self.files[index]
        with rio.open(os.path.join(self.root, self.directory, 's3_olci', file)) as f:
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
        file = self.files[index]
        mode = f'cloud_{self.mode}'
        with rio.open(os.path.join(self.root, self.directory, mode, file)) as f:
            sample['mask'] = torch.tensor(f.read(1).astype(np.int64))

        return sample
