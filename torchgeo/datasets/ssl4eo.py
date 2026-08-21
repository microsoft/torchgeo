# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Self-Supervised Learning for Earth Observation."""

import glob
import os
import random
import re
from collections.abc import Callable
from typing import ClassVar, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .landsat import Landsat, Landsat5TM, Landsat7, Landsat8
from .sentinel import Sentinel1, Sentinel2
from .utils import Path, Sample, disambiguate_timestamp, download_url, extract_archive


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
        all_bands: tuple[str, ...]
        rgb_bands: list[int]

    metadata: ClassVar[dict[str, _Metadata]] = {
        'tm_toa': {
            'all_bands': ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'),
            'rgb_bands': [2, 1, 0],
        },
        'etm_toa': {
            'all_bands': ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B6', 'B7', 'B8'),
            'rgb_bands': [2, 1, 0],
        },
        'etm_sr': {
            'all_bands': ('B1', 'B2', 'B3', 'B4', 'B5', 'B7'),
            'rgb_bands': [2, 1, 0],
        },
        'oli_tirs_toa': {
            'all_bands': (
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
            ),
            'rgb_bands': [3, 2, 1],
        },
        'oli_sr': {
            'all_bands': ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'),
            'rgb_bands': [3, 2, 1],
        },
    }

    url = 'https://hf.co/datasets/torchgeo/ssl4eo_l/resolve/e2467887e6a6bcd7547d9d5999f8d9bc3323dc31/{0}/ssl4eo_l_{0}.tar.gz{1}'
    checksums: ClassVar[dict[str, dict[str, str]]] = {
        'tm_toa': {
            'aa': '6497dfadb892471a24b6c3d93e5b0efd5830421281375667defb52e9fc2e99e5',
            'ab': '0c4d616ca1fbb19c19a06f634f4295c76135bdbae18394dfb11487310899d599',
            'ac': '09071ca2ace91c5916f488ae934d50969f8beddbe27ae5cc88792a31adbbfa0f',
            'ad': '7345009b3f54a322429e1f9187bc645265997e2eb33593920a73d7f5fdf50d71',
            'ae': 'f912f20513f89b99c630e5b696095040bf03ee4f7bc59c1c91db3628d3e718fa',
            'af': '0a8ce38c95bdefb30862f98c08256ec5898b114b792d0dd425bf58a15d602db9',
            'ag': '779ff524858198316c4b812fdc8e0c5326fc3d9648fd62adb3414214b65d3608',
        },
        'etm_toa': {
            'aa': '36b18f3fb3d0163c5f2e2ec72fddb1034018fd3824b7f89fcfcfd23d0bc18dd4',
            'ab': '1f50ab7ef2ea409e7f5033548922434d9dc9881b86693faae91dd2e6409a7519',
            'ac': '6c4570553b4350765a6541d8bda12edccdc4a20c10358616af085b555fbdbc8b',
            'ad': '52c8dabcea173009e43c0cfcaf6e113b6f2e3089a406370db0ec364e0a091201',
            'ae': 'f5621debc63a81a84454132833da735514d26f8845f43127ebd7e5b71e9dee1c',
            'af': 'cf776034fa42df1b61f252a2be7dd3868e2b89721ad3bc2cb2b070119175eea0',
            'ag': '59757f09049ef21eec80a61920a539d2fda8cfcfa12c79b3bfe70edf8ddad24a',
            'ah': 'bb0fb086b9240623fd4f4d1b22220bb93c8a6ab316883aa7f4c58dc57c0cd6ca',
            'ai': '2e2e96f3bfe1dfe3ac535d98966854bae93d823830605071dcdaa6754f134d50',
        },
        'etm_sr': {
            'aa': '49b41ea679c295a083d273d7c29672c1a0532d94377b4509ec8f80309ef176c4',
            'ab': '2681880da886ef3949cab8d10916896454635d18cc63e236228969685d1b32da',
            'ac': '9625cace7961658ffb33732d6fb0e9a8cff3ca8e1cc3b80f1aed63a0c7b37f11',
            'ad': '4dd58e9d182b804d0f6459aaa298fc06984cec24530b3cc69db9356561582bd1',
            'ae': 'bd49ee678400d752e5537f112257f172f097034da057fbf726d652c04f6b36ce',
            'af': '07441efd63e4745cfda78925c390d0bc9e4c691c964bade34017a50685312e46',
            'ag': '52080a68199f2f8edcd09a10c0060c40effd12a9cf665e9c5ed3420d80fd9eb4',
        },
        'oli_tirs_toa': {
            'aa': '9bdfb1350c29129207983db7eb6677389653f4e61e2002b4b772be76aff564a5',
            'ab': '102acc3f87a0f06425d1c11b161ea4ed7e37cc4ba60c948e946c18f65644f4d5',
            'ac': '7133a55c174c8b3dfe1563db0e0e9c85827f3f4ac94752bce22e9586cfe66cd9',
            'ad': 'b07181606605e6e9116a82f4f2af81dd52738cba846f02404e5b9118c7a3ea01',
            'ae': 'b33accff16554c6fe9273ea2019512421c6a6b6bb900d1035442c693dc71069e',
            'af': '230ec04a60cd242c68867c283a1f5689c3b8f6d573e042a21ba484484f1082da',
            'ag': 'a521b70ade95b8f255f237e5ddb53cf659ddfcb0de4736b8657d12f2b473560b',
            'ah': 'cb1529c74475b7a9f1365945b2285c5164737f6b24e458e8493f05051a0e9dde',
            'ai': '80303b44b0a0f4e2d9e8141cc0e7fb00317e6298eeb88f2d675c018362de526c',
        },
        'oli_sr': {
            'aa': 'c5c0b18ceb00e7dd645b26ec8362b0cdb0064be33cd21402fb73de6f7cdfe6ed',
            'ab': '85ede1f18b26957e5a0644d1a414fc917e7451cd8e0d2e12e165c18a982c5ee6',
            'ac': 'be85283037b10de7d10ed6dab34bb5949166d9f13c84c934ce49f1b3b4b2d887',
            'ad': 'c10e08b7c0c44ed153e7f7d3a0fbb359cc2c137f2bee25ed24a846edbe8ef329',
            'ae': 'dbcd8b0dda4917bfa841fd505c68e8890444a02704cf7969454d8f2cf946d09f',
            'af': '871f0e73c259cf35cfb7e543c66f33f74209e0c1c7032fe48981c6398ddff0b7',
            'ag': 'b2a1cbb10096228deebaed9422a23cdc5898c7d32afe33613899ddeb170d5a4d',
        },
    }

    def __init__(
        self,
        root: Path = 'data',
        split: Literal[
            'tm_toa', 'etm_toa', 'etm_sr', 'oli_tirs_toa', 'oli_sr'
        ] = 'oli_sr',
        seasons: Literal[1, 2, 3, 4] = 1,
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = True,
    ) -> None:
        """Initialize a new SSL4EOL instance.

        Args:
            root: root directory where dataset can be found
            split: one of ['tm_toa', 'etm_toa', 'etm_sr', 'oli_tirs_toa', 'oli_sr']
            seasons: number of seasonal patches to sample per location, 1--4
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, verify the checksum after downloading files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
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

    def __getitem__(self, index: int) -> Sample:
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
                ts.append((mint.timestamp() + maxt.timestamp()) / 2)
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
        for suffix, sha256 in self.checksums[self.split].items():
            download_url(
                self.url.format(self.split, suffix),
                self.root,
                sha256=sha256 if self.checksum else None,
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
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
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
         - 13
         - `GEE <https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED>`__
       * - s2a
         - Sentinel-2
         - SR
         - 12
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
        bands: tuple[str, ...]
        filename_regex: str

    metadata: ClassVar[dict[str, _Metadata]] = {
        's1': {
            'bands': ('VV', 'VH'),
            'filename_regex': r'^.{16}_(?P<date>\d{8}T\d{6})',
        },
        's2c': {
            'bands': (
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
            ),
            'filename_regex': r'^(?P<date>\d{8}T\d{6})',
        },
        's2a': {
            'bands': (
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
            ),
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
            'aa': 'c875a430ed19c48fa90ea2574785069eb939629beec372b42d8ba283aa26d5e6',
            'ab': '517fe7eaa8a93b74ab8ea2cd5a8f65fb2130986266c7a702bcdd312afce2b396',
            'ac': '0146dd447d6f89d6bfe88fec05155df586f0b10643a50a7d9cf22c37edd1c3a2',
            'ad': '02871cc0d33356ad3d06f8a496d7f06d10121caa0e9078f095491bb8649f9d0d',
            'ae': 'c4694fbba061ad89d4bfd1de2069b2a5cd19de2922d81f5a6364178d23948b61',
            'af': 'c7ba827f7b35f8bd974ee626eca55358da55830d85a29ecbfaa6e4bf70811bae',
            'ag': 'e37fac95aa0e78fe418e392652cecb5dfff72b98b0d6996fdf9c824da254bccb',
            'ah': '1824079dfe4f844b0f6672582e296b74f25ffd5ae8c46ed92b70b08b7d5eb0e6',
            'ai': 'b246750cdffe2c77dcea624c01aa5e6b53db73e33ce58082af4ce1fef5fb5b98',
            'aj': '6d3253bb0b294c4cf14fe8e4f09505d370e32be20b0bb8de3a829ce004154535',
        },
        's2c': {
            'aa': '1003dd0b4e11fdcca00855c3de23fa3209926c24074a36f40a07efeba377a9da',
            'ab': '96dd31b2d53e131daa73604013678e21dd2b29ff4865c73a24ada77001f77a7a',
            'ac': '23cb241e947de1da08bb9a53c6c9b088232bbf622c9aee2117e4b9bf8e93d83d',
            'ad': 'e9502f21de324e6fc400c73af1a6486e1b75176c6511642763fb899537dada1e',
            'ae': 'f47a7aaaeda7df30af9b892e54eb6c00941c706f82584c8c5b4481ee411e3e56',
            'af': '207d2dc70378fc9c289b666fe1730821ec6dca0d2d6fb60d5325ba31b17de896',
            'ag': '060e01cc896e1ed3e426ff965ed6aba2538d065b398575968b0f1ae92f6ef8bc',
            'ah': 'c32211630526dad919891d1a6aae82b7d82829f343f2176f04b37378b49d1f61',
            'ai': 'a62a34fe0708885b4bd0dcc3d8e2a14c0c29737d94f744b41381d2a6618ff240',
            'aj': 'a6206d4e40569b9e3b92eb4d5174193543980b6710b02f847d3629c50fd89d59',
            'ak': '5ae573daf729a0d32ad8d7998b78d91e4083289df1d4444feb4907b9091d9f03',
            'al': '75ef65fdd931176ef82bc0d87dbd2c35202714499dfe7a3918bf48d01788538f',
        },
        's2a': {
            'aa': 'b124fe99809fa2eee663d683435399e8f8ffdbc7fc40c3e5cbe5b3e3d5f29752',
            'ab': '31b624e24e41192340f449b90b6fefac2a6ba91e640467df4e4bd5b7ae15c36f',
            'ac': '4a73229310e4f4cf61584353f1fc39ef715b08ce65e4bbd4033065391c26e262',
            'ad': '3a0dfd18476ee1dd0f1dcc77e91a8b82312a1b9f64e977bcf4577407f63375d4',
            'ae': 'd5a7d1f8c07de315710aa528d1aed877093a379f4d035d22d823212cc22fb0ae',
            'af': '15f2e4a365f517ea272a14905caef8da5214c93ebf166e94820a1a9ae8bfa0d7',
            'ag': '9f18d7544fcfa63ae1c3481cb452c8f00e298112bac7d08fe199433569963c1c',
            'ah': 'c682a97d2fd420d6be214f0f3cf3d588940077d9fb7004d5b95d56e8c5532aa8',
            'ai': 'a5dec0d93ac846c5c7f44a79b8d0ab5e2507a4543cb7acff78604fb7b6fe54d0',
            'aj': 'ff8378e1f12a566f9ce440569f1aff5c373908f8fc32ca233ebd98e455bfd172',
            'ak': 'e78649ee5de002b961c62fa02674ec6a4accbb67c69962754012af29a2d51dec',
            'al': 'b5b43650d5a5fa0dbd4ab5cc791920757b98ffb4d684c8a7bb000e70a44b071a',
        },
    }

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['s1', 's2c', 's2a'] = 's2c',
        seasons: Literal[1, 2, 3, 4] = 1,
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = True,
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
            checksum: if True, verify the checksum of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionadded:: 0.7
           The *download* parameter.
        """
        self.root = root
        self.split = split
        self.seasons = seasons
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.bands = self.metadata[self.split]['bands']

    def __getitem__(self, index: int) -> Sample:
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
                ts.append((mint.timestamp() + maxt.timestamp()) / 2)

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
        path = os.path.join(self.root, self.split, '00000*', '*', '*.tif')
        if glob.glob(path):
            return

        # Check if the tar.gz files have already been downloaded
        exists = []
        for suffix in self.checksums[self.split]:
            path = os.path.join(
                self.root, self.filenames[self.split] + f'.tar.gz.part{suffix}'
            )
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
        for suffix, sha256 in self.checksums[self.split].items():
            download_url(
                self.url.format(self.filenames[self.split], suffix),
                self.root,
                sha256=sha256 if self.checksum else None,
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
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
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
