# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""HySpecNet dataset."""

import os
import re
from collections.abc import Callable, Sequence
from typing import ClassVar

import rasterio as rio
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor

from .enmap import EnMAP
from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import (
    Path,
    disambiguate_timestamp,
    download_url,
    extract_archive,
    percentile_normalization,
)


class HySpecNet11k(NonGeoDataset):
    """HySpecNet-11k dataset.

    `HySpecNet-11k <https://doi.org/10.5061/dryad.fttdz08zh>`__ is a large-scale
    benchmark dataset for hyperspectral image compression and self-supervised learning.
    It is made up of 11,483 nonoverlapping image patches acquired by the
    `EnMAP satellite <https://www.enmap.org/>`_. Each patch is a portion of 128 x 128
    pixels with 224 spectral bands and with a ground sample distance of 30 m.

    To construct HySpecNet-11k, a total of 250 EnMAP tiles acquired during the routine
    operation phase between 2 November 2022 and 9 November 2022 were considered. The
    considered tiles are associated with less than 10% cloud and snow cover. The tiles
    were radiometrically, geometrically and atmospherically corrected (L2A water & land
    product). Then, the tiles were divided into nonoverlapping image patches. The
    cropped patches at the borders of the tiles were eliminated. As a result, more than
    45 patches per tile are obtained, resulting in 11,483 patches for the full dataset.

    We provide predefined splits obtained by randomly dividing HySpecNet into:

    #. a training set that includes 70% of the patches,
    #. a validation set that includes 20% of the patches, and
    #. a test set that includes 10% of the patches.

    Depending on the way that we used for splitting the dataset, we define two
    different splits:

    #. an easy split, where patches from the same tile can be present in different sets
       (patchwise splitting); and
    #. a hard split, where all patches from one tile belong to the same set
       (tilewise splitting).

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2306.00385

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/torchgeo/hyspecnet/resolve/13e110422a6925cbac0f11edff610219b9399227/'
    md5s: ClassVar[dict[str, str]] = {
        'hyspecnet-11k-01.tar.gz': '974aae9197006727b42ec81796049efe',
        'hyspecnet-11k-02.tar.gz': 'f80574485f835b8a263b6c64076c0c62',
        'hyspecnet-11k-03.tar.gz': '6bc1de573f97fa4a75b79719b9270cb3',
        'hyspecnet-11k-04.tar.gz': '2463dc10653cb8be10d44951307c5e7d',
        'hyspecnet-11k-05.tar.gz': '16c1bd9e684673e741c0849bd015c988',
        'hyspecnet-11k-06.tar.gz': '8eef16b67d71af6eb4bc836d294fe3c4',
        'hyspecnet-11k-07.tar.gz': 'f61f0e7d6b05c861e69026b09130a5d6',
        'hyspecnet-11k-08.tar.gz': '19d390bc9e61b85e7d765f3077984976',
        'hyspecnet-11k-09.tar.gz': '197ff47befe5b9de88be5e1321c5ce5d',
        'hyspecnet-11k-10.tar.gz': '9e674cca126a9d139d6584be148d4bac',
        'hyspecnet-11k-splits.tar.gz': '94fad9e3c979c612c29a045406247d6c',
    }

    all_bands = EnMAP.all_bands
    default_bands = EnMAP.default_bands
    rgb_bands = EnMAP.rgb_bands

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        strategy: str = 'easy',
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new HySpecNet11k instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            strategy: Either 'easy' for patchwise splitting or 'hard' for tilewise
                splitting.
            bands: Bands to return.
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.split = split
        self.strategy = strategy
        self.bands = bands or self.default_bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self.wavelengths = torch.tensor([EnMAP.wavelengths[b] for b in self.bands])
        self.band_indices = [self.all_bands.index(b) + 1 for b in self.bands]

        self._verify()

        path = os.path.join(root, 'hyspecnet-11k', 'splits', strategy, f'{split}.csv')
        with open(path) as f:
            self.files = f.read().strip().split('\n')

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and label at that index.
        """
        path = self.files[index].replace('DATA.npy', 'SPECTRAL_IMAGE.TIF')
        file = os.path.basename(path)
        match = re.match(EnMAP.filename_regex, file, re.VERBOSE)
        assert match
        mint, maxt = disambiguate_timestamp(match.group('date'), EnMAP.date_format)

        with rio.open(os.path.join(self.root, 'hyspecnet-11k', 'patches', file)) as src:
            minx, maxx = src.bounds.left, src.bounds.right
            miny, maxy = src.bounds.bottom, src.bounds.top
            sample = {
                'image': torch.tensor(src.read(self.band_indices).astype('float32')),
                'x': torch.tensor((minx + maxx) / 2),
                'y': torch.tensor((miny + maxy) / 2),
                't': torch.tensor((mint.timestamp() + maxt.timestamp()) / 2),
                'wavelength': self.wavelengths,
                'res': torch.tensor(30),
            }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        exists = []
        for directory in ['patches', 'splits']:
            path = os.path.join(self.root, 'hyspecnet-11k', directory)
            exists.append(os.path.isdir(path))

        if all(exists):
            return

        for file, md5 in self.md5s.items():
            # Check if the file has already been downloaded
            path = os.path.join(self.root, file)
            if os.path.isfile(path):
                extract_archive(path)
                continue

            # Check if the user requested to download the dataset
            if self.download:
                url = self.url + file
                download_url(url, self.root, md5=md5 if self.checksum else None)
                extract_archive(path)
                continue

            raise DatasetNotFoundError(self)

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            suptitle: optional string to use as a suptitle

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

        image = sample['image'][rgb_indices].cpu().numpy()
        image = rearrange(image, 'c h w -> h w c')
        image = percentile_normalization(image)

        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')

        if suptitle:
            fig.suptitle(suptitle)

        return fig
