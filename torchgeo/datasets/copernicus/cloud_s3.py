# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench Cloud-S3 dataset."""

import os
from collections.abc import Callable, Sequence
from typing import Literal

from matplotlib.colors import ListedColormap
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
       * - 0
         - Invalid
         - Invalid pixels, should be ignored during training.
       * - 1
         - Clear
         - Land, coastline, or water pixels.
       * - 2
         - Cloud-Ambiguous
         - Semi-transparent clouds, or clouds where the detection level is uncertain.
       * - 3
         - Cloud-Sure
         - Fully-opaque clouds with full confidence of their detection.
       * - 4
         - Cloud Shadow
         - Pixels are affected by a cloud shadow.
       * - 5
         - Snow/Ice
         - Clear snow/ice pixels.

    .. list-table:: Binary Classification
       :header-rows: 1

       * - Code
         - Class
         - Description
       * - 0
         - Invalid
         - Invalid pixels, should be ignored during training.
       * - 1
         - Clear
         - Land, coastline, water, snow, or ice pixels.
       * - 2
         - Cloud
         - Pixels which are either cloud-sure or cloud-ambiguous.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.11849

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l1_cloud_s3/cloud_s3.zip'
    md5 = '1f82a8ccf16a0c44f0b1729e523e343a'
    zipfile = 'cloud_s3.zip'
    directory = 'cloud_s3'
    filename_regex = r'S3[AB]_OL_1_EFR____(?P<date>\d{8}T\d{6})'
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
    cmap = ListedColormap(['red', 'gray', 'lightgray', 'white', 'black', 'snow'])
    classes: tuple[str, ...] = (
        'Invalid',
        'Clear',
        'Cloud-Ambiguous',
        'Cloud-Sure',
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
        if mode == 'binary':
            self.classes = ('Invalid', 'Clear', 'Cloud')
            self.cmap = ListedColormap(['red', 'gray', 'white'])
        super().__init__(root, split, bands, transforms, download, checksum)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        file = self.files[index]
        image_path = os.path.join(self.root, self.directory, 's3_olci', file)
        mask_path = os.path.join(self.root, self.directory, f'cloud_{self.mode}', file)
        sample = self._load_image(image_path) | self._load_mask(mask_path)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
