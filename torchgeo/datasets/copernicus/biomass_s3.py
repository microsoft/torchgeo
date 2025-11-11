# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench Biomass-S3 dataset."""

import glob
import os
from collections.abc import Callable, Sequence
from typing import Literal

import pandas as pd
import torch
import torch.nn.functional as F

from ..utils import Path, Sample, stack_samples
from .base import CopernicusBenchBase


class CopernicusBenchBiomassS3(CopernicusBenchBase):
    """Copernicus-Bench Biomass-S3 dataset.

    Biomass-S3 is a regression dataset based on Sentinel-3 OLCI images and CCI biomass.
    The biomass product is part of the European Space Agency's Climate Change Initiative
    (CCI) program and delivers global forest above-ground biomass at 100 m spatial
    resolution.

    This benchmark supports both static (1 image/location) and time series
    (1-4 images/location) modes, the former is used in the original benchmark.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://catalogue.ceda.ac.uk/uuid/02e1b18071ad45a19b4d3e8adafa2817/

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l3_biomass_s3/biomass_s3.zip'
    sha256 = '1d005b200d50f2e8b5f4482959bdfa6e2d7d05a8cd828d7f438c99a4e1cfbaef'
    zipfile = 'biomass_s3.zip'
    directory = 'biomass_s3'
    filename = 'static_fnames-{}.csv'
    dtype = torch.float
    filename_regex = r'S3[AB]_(?P<date>\d{8}T\d{6})'
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
    cmap = 'YlGn'

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        mode: Literal['static', 'time-series'] = 'static',
        bands: Sequence[str] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = True,
    ) -> None:
        """Initialize a new CopernicusBenchBiomassS3 instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            mode: One of 'static' or 'time-series'.
            bands: Sequence of band names to load (defaults to all bands).
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, verify the checksum of the downloaded files (may be slow).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.mode = mode
        super().__init__(root, split, bands, transforms, download, checksum)
        filepath = os.path.join(root, self.directory, self.filename.format(split))
        self.files = pd.read_csv(filepath, header=None)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        pid, file = self.files.iloc[index]
        match self.mode:
            case 'static':
                path = os.path.join(self.root, self.directory, 's3_olci', pid, file)
                sample = self._load_image(path)
            case 'time-series':
                paths = os.path.join(self.root, self.directory, 's3_olci', pid, '*.tif')
                samples = [self._load_image(path) for path in sorted(glob.glob(paths))]
                if not samples:
                    msg = f'No Sentinel-3 samples found for {pid}.'
                    raise FileNotFoundError(msg)

                max_h = max(sample['image'].shape[-2] for sample in samples)
                max_w = max(sample['image'].shape[-1] for sample in samples)
                if any(
                    sample['image'].shape[-2] != max_h
                    or sample['image'].shape[-1] != max_w
                    for sample in samples
                ):
                    padded_samples: list[Sample] = []
                    for sample_dict in samples:
                        image = sample_dict['image']
                        h, w = image.shape[-2:]
                        pad_h = max_h - h
                        pad_w = max_w - w
                        if pad_h or pad_w:
                            padded_image = F.pad(image, (0, pad_w, 0, pad_h))
                            sample_dict = sample_dict.copy()
                            sample_dict['image'] = padded_image
                        padded_samples.append(sample_dict)
                    samples = padded_samples
                sample = stack_samples(samples)

        path = os.path.join(self.root, self.directory, 'biomass', f'{pid}.tif')
        sample |= self._load_mask(path)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
