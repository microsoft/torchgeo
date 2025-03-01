# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Self-Supervised Learning for Earth Observation."""

import glob
import os
import random
import re
from collections.abc import Callable
from typing import ClassVar, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .landsat import Landsat, Landsat5TM, Landsat7, Landsat8
from .sentinel import Sentinel1, Sentinel2
from .utils import Path, disambiguate_timestamp, download_url, extract_archive


class SSL4EO(NonGeoDataset):
    """Base class for all SSL4EO datasets.

    Self-Supervised Learning for Earth Observation (SSL4EO) is a collection of
    large-scale multimodal multitemporal datasets for unsupervised/self-supervised
    pre-training in Earth observation.

    .. versionadded:: 0.5
    """


class SSL4EOL(SSL4EO):
    """SSL4EO-L dataset.

    Landsat version of SSL4EO.

    The dataset consists of a parallel corpus (same locations and dates for SR/TOA)
    for the following sensors:

    .. list-table::
       :widths: 10 10 10 10 10 10
       :header-rows: 1

       * - Split
         - Satellites
         - Sensors
         - Level
         - # Bands
         - Link
       * - tm_toa
         - Landsat 4--5
         - TM
         - TOA
         - 7
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_TOA>`__
       * - etm_sr
         - Landsat 7
         - ETM+
         - SR
         - 6
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2>`__
       * - etm_toa
         - Landsat 7
         - ETM+
         - TOA
         - 9
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_TOA>`__
       * - oli_tirs_toa
         - Landsat 8--9
         - OLI+TIRS
         - TOA
         - 11
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA>`__
       * - oli_sr
         - Landsat 8--9
         - OLI
         - SR
         - 7
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2>`__

    Each patch has the following properties:

    * 264 x 264 pixels
    * Resampled to 30 m resolution (7920 x 7920 m)
    * 4 seasonal timestamps
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
        all_bands: list[str]
        rgb_bands: list[int]

    metadata: ClassVar[dict[str, _Metadata]] = {
        'tm_toa': {
            'all_bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
            'rgb_bands': [2, 1, 0],
        },
        'etm_toa': {
            'all_bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B6', 'B7', 'B8'],
            'rgb_bands': [2, 1, 0],
        },
        'etm_sr': {
            'all_bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
            'rgb_bands': [2, 1, 0],
        },
        'oli_tirs_toa': {
            'all_bands': [
                'B1',
                'B2',
                'B3',
                'B4',
                'B5',
                'B6',
                'B7',
                'B8',
                'B9',
                'B10',
                'B11',
            ],
            'rgb_bands': [3, 2, 1],
        },
        'oli_sr': {
            'all_bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
            'rgb_bands': [3, 2, 1],
        },
    }

    url = 'https://hf.co/datasets/torchgeo/ssl4eo_l/resolve/e2467887e6a6bcd7547d9d5999f8d9bc3323dc31/{0}/ssl4eo_l_{0}.tar.gz{1}'
    checksums: ClassVar[dict[str, dict[str, str]]] = {
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

        if split.startswith('tm'):
            base: type[Landsat] = Landsat5TM
        elif split.startswith('etm'):
            base = Landsat7
        else:
            base = Landsat8

        self.wavelengths = []
        for band in self.metadata[split]['all_bands']:
            self.wavelengths.append(base.wavelengths[band])

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
        xs = []
        ys = []
        ts = []
        wavelengths = []
        for subdir in subdirs:
            mint, maxt = disambiguate_timestamp(subdir[-8:], Landsat.date_format)
            directory = os.path.join(root, subdir)
            filename = os.path.join(directory, 'all_bands.tif')
            with rasterio.open(filename) as f:
                minx, maxx = f.bounds.left, f.bounds.right
                miny, maxy = f.bounds.bottom, f.bounds.top
                image = f.read()
                images.append(torch.from_numpy(image.astype(np.float32)))
                xs.append((minx + maxx) / 2)
                ys.append((miny + maxy) / 2)
                ts.append((mint + maxt) / 2)
                wavelengths.extend(self.wavelengths)

        sample = {
            'image': torch.cat(images),
            'x': torch.tensor(xs),
            'y': torch.tensor(ys),
            't': torch.tensor(ts),
            'wavelength': torch.tensor(wavelengths),
            'res': torch.tensor(30),
        }

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
        num_bands = len(self.metadata[self.split]['all_bands'])
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


class SSL4EOS12(SSL4EO):
    """SSL4EO-S12 dataset.

    `Sentinel-1/2 <https://github.com/zhu-xlab/SSL4EO-S12>`_ version of SSL4EO.

    The dataset consists of a parallel corpus (same locations and dates)
    for the following satellites:

    .. list-table::
       :widths: 10 10 10 10 10
       :header-rows: 1

       * - Split
         - Satellite
         - Level
         - # Bands
         - Link
       * - s1
         - Sentinel-1
         - GRD
         - 2
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD>`__
       * - s2c
         - Sentinel-2
         - TOA
         - 12
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED>`__
       * - s2a
         - Sentinel-2
         - SR
         - 13
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED>`__

    Each patch has the following properties:

    * 264 x 264 pixels
    * Resampled to 10 m resolution (2640 x 2640 m)
    * 4 seasonal timestamps

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2211.07044

    .. note::

       The dataset is about 1.5 TB when compressed and 3.7 TB when uncompressed.

    .. versionadded:: 0.5
    """

    size = 264

    class _Metadata(TypedDict):
        bands: list[str]
        filename_regex: str

    metadata: ClassVar[dict[str, _Metadata]] = {
        's1': {
            'bands': ['VV', 'VH'],
            'filename_regex': r'^.{16}_(?P<date>\d{8}T\d{6})',
        },
        's2c': {
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
            'filename_regex': r'^(?P<date>\d{8}T\d{6})',
        },
        's2a': {
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
            'filename_regex': r'^(?P<date>\d{8}T\d{6})',
        },
    }

    url = 'https://hf.co/datasets/wangyi111/SSL4EO-S12/resolve/3f5ddad68ba2ea29d019b0cef6cf292ff8af0d62/{0}/{0}.tar.gz.part{1}'
    filenames: ClassVar[dict[str, str]] = {
        's1': 's1_grd',
        's2c': 's2_l1c',
        's2a': 's2_l2a',
    }
    checksums: ClassVar[dict[str, dict[str, str]]] = {
        's1': {
            'aa': '6df278053fc3e4c3fd7de2f77856e606',
            'ab': '837755b4ba8d82faf254df9e5fec13a7',
            'ac': '6400423305d6084e2006eede75cf288e',
            'ad': '22a50d7362d9cbc9714e0740fe2122c7',
            'ae': 'd6ac97ead00b4296a95376949c946b12',
            'af': 'd8047814061431dc627b9ae345c80891',
            'ag': '089ce0548cb7902ce873181cc61f5d70',
            'ah': '745b48c2896ca764ef54f91e4e7c555e',
            'ai': 'c36595cf9617b3b7ea722f63dcccbedc',
            'aj': 'cf16f1d81e8bff2d663e4eba79ec6fa3',
        },
        's2c': {
            'aa': 'a3ef419cc65c4d8ac19b9acc55726166',
            'ab': '580451e8fdf93067ad79202b95dd1a5c',
            'ac': 'a6f7318868f5ba1d94fb9363b50307e4',
            'ad': '86f324215b04cdf4242d07aaf3cdfe57',
            'ae': '5895a545460f34b1712c17732e0f5533',
            'af': '078078bc58d8ecc214ddfd838f796700',
            'ag': '3557dd4c24a5942020391a5baaf51abb',
            'ah': 'd59f89271e414648663d3acb66121761',
            'ai': '1a213539c989d16da4e5b4e09feaa98a',
            'aj': '0b229af5633c7f63486b6d7771b737db',
            'ak': 'babe8bed884d31b891151f5717a83b5c',
            'al': '8d1f5ad28ee868ab0595c889446b8e5f',
        },
        's2a': {
            'aa': 'ef847d906ab44cc9a94d086a89473833',
            'ab': '4a6a8ed9e2a08887707d83bcb6eb57af',
            'ac': '00b706a771df4c4df4cc70a20d790339',
            'ad': '579024e84bd9ab0b86e1182792c8dcf9',
            'ae': 'e259f3536355b665aea490c22c897e59',
            'af': '2a15be319ad15f749bfd4ed85d14c172',
            'ag': 'd8224cff1e727543473b0111e307110c',
            'ah': '0015d8aa5ea9201e13b401fd61c36c6f',
            'ai': 'dfce87c0a9550177fd4b82887902b6e3',
            'aj': '688392701760b737ad74cb0e8c7fb731',
            'ak': 'd8f3e4b110f22f0477973ed2c35586b6',
            'al': '1cc3641cd52afedaa1c50d14d84a6664',
        },
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 's2c',
        seasons: int = 1,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SSL4EOS12 instance.

        Args:
            root: root directory where dataset can be found
            split: one of "s1" (Sentinel-1 GRD dual-pol SAR),
                "s2c" (Sentinel-2 Level-1C top-of-atmosphere reflectance), or
                "s2a" (Sentinel-2 Level-2A surface reflectance)
            seasons: number of seasonal patches to sample per location, 1--4
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
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
        self.download = download
        self.checksum = checksum

        self._verify()

        self.bands = self.metadata[self.split]['bands']

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
        filename_regex = self.metadata[self.split]['filename_regex']

        images = []
        xs = []
        ys = []
        ts = []
        wavelengths: list[float] = []
        for subdir in subdirs:
            directory = os.path.join(root, subdir)
            if match := re.match(filename_regex, subdir):
                date_str = match.group('date')
                match self.split:
                    case 's1':
                        date_format = Sentinel1.date_format
                    case 's2c' | 's2a':
                        date_format = Sentinel2.date_format
                mint, maxt = disambiguate_timestamp(date_str, date_format)
                for band in self.bands:
                    match self.split:
                        case 's1':
                            wavelengths.append(Sentinel1.wavelength)
                        case 's2c' | 's2a':
                            wavelengths.append(Sentinel2.wavelengths[band])

                    filename = os.path.join(directory, f'{band}.tif')
                    with rasterio.open(filename) as f:
                        minx, maxx = f.bounds.left, f.bounds.right
                        miny, maxy = f.bounds.bottom, f.bounds.top
                        image = f.read(out_shape=(1, self.size, self.size))
                        images.append(torch.from_numpy(image.astype(np.float32)))
                xs.append((minx + maxx) / 2)
                ys.append((miny + maxy) / 2)
                ts.append((mint + maxt) / 2)

        sample = {
            'image': torch.cat(images),
            'x': torch.tensor(xs),
            'y': torch.tensor(ys),
            't': torch.tensor(ts),
            'wavelength': torch.tensor(wavelengths),
            'res': torch.tensor(10),
        }

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
        path = os.path.join(self.root, self.split, '00000*', '*', 'all_bands.tif')
        if glob.glob(path):
            return

        # Check if the tar.gz files have already been downloaded
        exists = []
        for suffix in self.checksums[self.split]:
            path = os.path.join(self.root, self.filenames[self.split] + f'.tar.gz.part{suffix}')
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
                self.url.format(self.filenames[self.split], suffix),
                self.root,
                md5=md5 if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        # Concatenate all tarballs together
        chunk_size = 2**15  # same as torchvision
        path = os.path.join(self.root, self.filenames[self.split] + '.tar.gz')
        with open(path, 'wb') as f:
            for suffix in self.checksums[self.split]:
                with open(f'{path}.part{suffix}', 'rb') as g:
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
