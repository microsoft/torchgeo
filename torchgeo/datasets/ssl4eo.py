# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Self-Supervised Learning for Earth Observation."""

import glob
import os
import random
from collections.abc import Callable
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, check_integrity, download_url, extract_archive


class SSL4EO(NonGeoDataset):
    """Base class for all SSL4EO datasets.

    Self-Supervised Learning for Earth Observation (SSL4EO) is a collection of
    large-scale multimodal multitemporal datasets for unsupervised/self-supervised
    pre-training in Earth observation.

    .. versionadded:: 0.5
    """


class SSL4EOL(NonGeoDataset):
    """SSL4EO-L dataset.

    Landsat version of SSL4EO.

    The dataset consists of a parallel corpus (same locations and dates for SR/TOA)
    for the following sensors:

    .. list-table::
       :widths: 10 10 10 10 10
       :header-rows: 1

       * - Satellites
         - Sensors
         - Level
         - # Bands
         - Link
       * - Landsat 4--5
         - TM
         - TOA
         - 7
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_TOA>`__
       * - Landsat 7
         - ETM+
         - SR
         - 6
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2>`__
       * - Landsat 7
         - ETM+
         - TOA
         - 9
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_TOA>`__
       * - Landsat 8--9
         - OLI+TIRS
         - TOA
         - 11
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA>`__
       * - Landsat 8--9
         - OLI
         - SR
         - 7
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2>`__

    Each patch has the following properties:

    * 264 x 264 pixels
    * Resampled to 30 m resolution (7920 x 7920 m)
    * Single multispectral GeoTIFF file

    .. note::

       Each split is 300--400 GB and requires 3x that to concatenate and extract
       tarballs. Tarballs can be safely deleted after extraction to save space.
       The dataset takes about 1.5 hrs to download and checksum and another 3 hrs
       to extract.

    If you use this dataset in your research, please cite the following paper:

    * https://proceedings.neurips.cc/paper_files/paper/2023/hash/bbf7ee04e2aefec136ecf60e346c2e61-Abstract-Datasets_and_Benchmarks.html

    .. versionadded:: 0.5
    """

    class _Metadata(TypedDict):
        num_bands: int
        rgb_bands: list[int]

    metadata: dict[str, _Metadata] = {
        'tm_toa': {'num_bands': 7, 'rgb_bands': [2, 1, 0]},
        'etm_toa': {'num_bands': 9, 'rgb_bands': [2, 1, 0]},
        'etm_sr': {'num_bands': 6, 'rgb_bands': [2, 1, 0]},
        'oli_tirs_toa': {'num_bands': 11, 'rgb_bands': [3, 2, 1]},
        'oli_sr': {'num_bands': 7, 'rgb_bands': [3, 2, 1]},
    }

    url = 'https://hf.co/datasets/torchgeo/ssl4eo_l/resolve/e2467887e6a6bcd7547d9d5999f8d9bc3323dc31/{0}/ssl4eo_l_{0}.tar.gz{1}'
    checksums = {
        'tm_toa': {
            'aa': '553795b8d73aa253445b1e67c5b81f11',
            'ab': 'e9e0739b5171b37d16086cb89ab370e8',
            'ac': '6cb27189f6abe500c67343bfcab2432c',
            'ad': '15a885d4f544d0c1849523f689e27402',
            'ae': '35523336bf9f8132f38ff86413dcd6dc',
            'af': 'fa1108436034e6222d153586861f663b',
            'ag': 'd5c91301c115c00acaf01ceb3b78c0fe',
        },
        'etm_toa': {
            'aa': '587c3efc7d0a0c493dfb36139d91ccdf',
            'ab': 'ec34f33face893d2d8fd152496e1df05',
            'ac': '947acc2c6bc3c1d1415ac92bab695380',
            'ad': 'e31273dec921e187f5c0dc73af5b6102',
            'ae': '43390a47d138593095e9a6775ae7dc75',
            'af': '082881464ca6dcbaa585f72de1ac14fd',
            'ag': 'de2511aaebd640bd5e5404c40d7494cb',
            'ah': '124c5fbcda6871f27524ae59480dabc5',
            'ai': '12b5f94824b7f102df30a63b1139fc57',
        },
        'etm_sr': {
            'aa': 'baa36a9b8e42e234bb44ab4046f8f2ac',
            'ab': '9fb0f948c76154caabe086d2d0008fdf',
            'ac': '99a55367178373805d357a096d68e418',
            'ad': '59d53a643b9e28911246d4609744ef25',
            'ae': '7abfcfc57528cb9c619c66ee307a2cc9',
            'af': 'bb23cf26cc9fe156e7a68589ec69f43e',
            'ag': '97347e5a81d24c93cf33d99bb46a5b91',
        },
        'oli_tirs_toa': {
            'aa': '4711369b861c856ebfadbc861e928d3a',
            'ab': '660a96cda1caf54df837c4b3c6c703f6',
            'ac': 'c9b6a1117916ba318ac3e310447c60dc',
            'ad': 'b8502e9e92d4a7765a287d21d7c9146c',
            'ae': '5c11c14cfe45f78de4f6d6faf03f3146',
            'af': '5b0ed3901be1000137ddd3a6d58d5109',
            'ag': 'a3b6734f8fe6763dcf311c9464a05d5b',
            'ah': '5e55f92e3238a8ab3e471be041f8111b',
            'ai': 'e20617f73d0232a0c0472ce336d4c92f',
        },
        'oli_sr': {
            'aa': 'ca338511c9da4dcbfddda28b38ca9e0a',
            'ab': '7f4100aa9791156958dccf1bb2a88ae0',
            'ac': '6b0f18be2b63ba9da194cc7886dbbc01',
            'ad': '57efbcc894d8da8c4975c29437d8b775',
            'ae': '2594a0a856897f3f5a902c830186872d',
            'af': 'a03839311a2b3dc17dfb9fb9bc4f9751',
            'ag': '6a329d8fd9fdd591e400ab20f9d11dea',
        },
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'oli_sr',
        seasons: int = 1,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SSL4EOL instance.

        Args:
            root: root directory where dataset can be found
            split: one of ['tm_toa', 'etm_toa', 'etm_sr', 'oli_tirs_toa', 'oli_sr']
            seasons: number of seasonal patches to sample per location, 1--4
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            AssertionError: if any arguments are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.metadata
        assert seasons in range(1, 5)

        self.root = root
        self.subdir = os.path.join(root, f'ssl4eo_l_{split}')
        self.split = split
        self.seasons = seasons
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.scenes = sorted(os.listdir(self.subdir))

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image sample
        """
        root = os.path.join(self.subdir, self.scenes[index])
        subdirs = os.listdir(root)
        subdirs = random.sample(subdirs, self.seasons)

        images = []
        for subdir in subdirs:
            directory = os.path.join(root, subdir)
            filename = os.path.join(directory, 'all_bands.tif')
            with rasterio.open(filename) as f:
                image = f.read()
                images.append(torch.from_numpy(image.astype(np.float32)))

        sample = {'image': torch.cat(images)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.scenes)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        path = os.path.join(self.subdir, '00000*', '*', 'all_bands.tif')
        if glob.glob(path):
            return

        # Check if the tar.gz files have already been downloaded
        exists = []
        for suffix in self.checksums[self.split]:
            path = self.subdir + f'.tar.gz{suffix}'
            exists.append(os.path.exists(path))

        if all(exists):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for suffix, md5 in self.checksums[self.split].items():
            download_url(
                self.url.format(self.split, suffix),
                self.root,
                md5=md5 if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        # Concatenate all tarballs together
        chunk_size = 2**15  # same as torchvision
        path = self.subdir + '.tar.gz'
        with open(path, 'wb') as f:
            for suffix in self.checksums[self.split]:
                with open(path + suffix, 'rb') as g:
                    while chunk := g.read(chunk_size):
                        f.write(chunk)

        # Extract the concatenated tarball
        extract_archive(path)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        fig, axes = plt.subplots(
            ncols=self.seasons, squeeze=False, figsize=(4 * self.seasons, 4)
        )
        num_bands = self.metadata[self.split]['num_bands']
        rgb_bands = self.metadata[self.split]['rgb_bands']

        for i in range(self.seasons):
            image = sample['image'][i * num_bands : (i + 1) * num_bands].byte()

            image = image[rgb_bands].permute(1, 2, 0)
            axes[0, i].imshow(image)
            axes[0, i].axis('off')

            if show_titles:
                axes[0, i].set_title(f'Split {self.split}, Season {i + 1}')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class SSL4EOS12(NonGeoDataset):
    """SSL4EO-S12 dataset.

    `Sentinel-1/2 <https://github.com/zhu-xlab/SSL4EO-S12>`_ version of SSL4EO.

    The dataset consists of unlabeled patch triplets (Sentinel-1 dual-pol SAR,
    Sentinel-2 top-of-atmosphere multispectral, Sentinel-2 surface reflectance
    multispectral) from 251079 locations across the globe, each patch covering
    2640mx2640m and including four seasonal time stamps.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2211.07044

    .. note::

       This dataset can be downloaded using:

       .. code-block:: console

          $ export RSYNC_PASSWORD=m1660427.001
          $ rsync -av rsync://m1660427.001@dataserv.ub.tum.de/m1660427.001/ .

       The dataset is about 1.5 TB when compressed and 3.7 TB when uncompressed, and
       takes roughly 36 hrs to download, 1 hr to checksum, and 12 hrs to extract.

    .. versionadded:: 0.5
    """

    size = 264

    class _Metadata(TypedDict):
        filename: str
        md5: str
        bands: list[str]

    metadata: dict[str, _Metadata] = {
        's1': {
            'filename': 's1.tar.gz',
            'md5': '51ee23b33eb0a2f920bda25225072f3a',
            'bands': ['VV', 'VH'],
        },
        's2c': {
            'filename': 's2_l1c.tar.gz',
            'md5': 'b4f8b03c365e4a85780ded600b7497ab',
            'bands': [
                'B1',
                'B2',
                'B3',
                'B4',
                'B5',
                'B6',
                'B7',
                'B8',
                'B8A',
                'B9',
                'B10',
                'B11',
                'B12',
            ],
        },
        's2a': {
            'filename': 's2_l2a.tar.gz',
            'md5': '85496cd9d6742aee03b6a1c99cee0ac1',
            'bands': [
                'B1',
                'B2',
                'B3',
                'B4',
                'B5',
                'B6',
                'B7',
                'B8',
                'B8A',
                'B9',
                'B11',
                'B12',
            ],
        },
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 's2c',
        seasons: int = 1,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SSL4EOS12 instance.

        Args:
            root: root directory where dataset can be found
            split: one of "s1" (Sentinel-1 dual-pol SAR), "s2c" (Sentinel-2 Level-1C
                top-of-atmosphere reflectance), and "s2a" (Sentinel-2 Level-2a surface
                reflectance)
            seasons: number of seasonal patches to sample per location, 1--4
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found.
        """
        assert split in self.metadata
        assert seasons in range(1, 5)

        self.root = root
        self.split = split
        self.seasons = seasons
        self.transforms = transforms
        self.checksum = checksum

        self.bands = self.metadata[self.split]['bands']

        self._verify()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image sample
        """
        root = os.path.join(self.root, self.split, f'{index:07}')
        subdirs = os.listdir(root)
        subdirs = random.sample(subdirs, self.seasons)

        images = []
        for subdir in subdirs:
            directory = os.path.join(root, subdir)
            for band in self.bands:
                filename = os.path.join(directory, f'{band}.tif')
                with rasterio.open(filename) as f:
                    image = f.read(out_shape=(1, self.size, self.size))
                    images.append(torch.from_numpy(image.astype(np.float32)))

        sample = {'image': torch.cat(images)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return 251079

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        directory_path = os.path.join(self.root, self.split)
        if os.path.exists(directory_path):
            return

        # Check if the zip files have already been downloaded
        filename = self.metadata[self.split]['filename']
        zip_path = os.path.join(self.root, filename)
        md5 = self.metadata[self.split]['md5'] if self.checksum else None
        integrity = check_integrity(zip_path, md5)
        if integrity:
            self._extract()
        else:
            raise DatasetNotFoundError(self)

    def _extract(self) -> None:
        """Extract the dataset."""
        filename = self.metadata[self.split]['filename']
        extract_archive(os.path.join(self.root, filename))

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        nrows = 2 if self.split == 's1' else 1
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=self.seasons,
            squeeze=False,
            figsize=(4 * self.seasons, 4 * nrows),
        )

        for i in range(self.seasons):
            image = sample['image'][i * len(self.bands) : (i + 1) * len(self.bands)]

            if self.split == 's1':
                axes[0, i].imshow(image[0])
                axes[1, i].imshow(image[1])
            else:
                image = image[[3, 2, 1]].permute(1, 2, 0)
                image = torch.clamp(image / 3000, min=0, max=1)
                axes[0, i].imshow(image)

            axes[0, i].axis('off')

            if show_titles:
                axes[0, i].set_title(f'Split {self.split}, Season {i + 1}')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
