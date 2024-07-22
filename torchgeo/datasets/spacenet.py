# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet datasets."""

import abc
import copy
import glob
import math
import os
import re
from collections.abc import Callable
from typing import Any

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import torch
from fiona.errors import FionaValueError
from fiona.transform import transform_geom
from matplotlib.figure import Figure
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import Affine
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    check_integrity,
    extract_archive,
    percentile_normalization,
    which,
)


class SpaceNet(NonGeoDataset, abc.ABC):
    """Abstract base class for the SpaceNet datasets.

    The `SpaceNet <https://spacenet.ai/datasets/>`__ datasets are a set of
    datasets that all together contain >11M building footprints and ~20,000 km
    of road labels mapped over high-resolution satellite imagery obtained from
    a variety of sensors such as Worldview-2, Worldview-3 and Dove.

    .. note::

       The SpaceNet datasets require the following additional library to be installed:

       * `AWS CLI <https://aws.amazon.com/cli/>`_: to download the dataset from AWS.
    """

    @property
    @abc.abstractmethod
    def dataset_id(self) -> str:
        """Dataset ID."""

    @property
    @abc.abstractmethod
    def all_aois(self) -> list[int]:
        """List of all valid areas of interest (AOIs)."""

    @property
    @abc.abstractmethod
    def images(self) -> list[str]:
        """List of all valid image products."""

    @property
    @abc.abstractmethod
    def tarballs(self) -> dict[str, dict[int, list[str]]]:
        """Mapping of tarballs[split][aoi] = [tarballs]."""

    @property
    @abc.abstractmethod
    def md5s(self) -> dict[str, dict[int, list[str]]]:
        """Mapping of md5s[split][aoi] = [md5s]."""

    @property
    @abc.abstractmethod
    def imagery(self) -> dict[str, str]:
        """Mapping of image identifier and filename."""

    @property
    @abc.abstractmethod
    def label_glob(self) -> str:
        """Label filename."""

    @property
    @abc.abstractmethod
    def collection_md5_dict(self) -> dict[str, str]:
        """Mapping of collection id and md5 checksum."""

    @property
    @abc.abstractmethod
    def chip_size(self) -> dict[str, tuple[int, int]]:
        """Mapping of images and their chip size."""

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        aois: list[str] = [],
        image: str | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: 'train' or 'test' split
            image: image selection
            aois: areas of interest
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: if True, download dataset and store it in the root directory.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: If any invalid arguments are passed.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in {'train', 'test'}
        assert set(aois) <= set(self.all_aois)
        assert image in self.images

        self.root = root
        self.split = split
        self.aois = aois
        self.image = image
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.files = self._load_files(root)

    def _load_files(self, root: Path) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image and label
        """
        files = []
        for collection in self.collections:
            images = glob.glob(os.path.join(root, collection, '*', self.filename))
            images = sorted(images)
            for imgpath in images:
                lbl_path = os.path.join(
                    f'{os.path.dirname(imgpath)}-labels', self.label_glob
                )
                files.append({'image_path': imgpath, 'label_path': lbl_path})
        return files

    def _load_image(self, path: Path) -> tuple[Tensor, Affine, CRS]:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        filename = os.path.join(path)
        with rio.open(filename) as img:
            array = img.read().astype(np.int32)
            tensor = torch.from_numpy(array).float()
            return tensor, img.transform, img.crs

    def _load_mask(
        self, path: Path, tfm: Affine, raster_crs: CRS, shape: tuple[int, int]
    ) -> Tensor:
        """Rasterizes the dataset's labels (in geojson format).

        Args:
            path: path to the label
            tfm: transform of corresponding image
            raster_crs: CRS of raster file
            shape: shape of corresponding image

        Returns:
            Tensor: label tensor
        """
        try:
            with fiona.open(path) as src:
                vector_crs = CRS(src.crs)
                if raster_crs == vector_crs:
                    labels = [feature['geometry'] for feature in src]
                else:
                    labels = [
                        transform_geom(
                            vector_crs.to_string(),
                            raster_crs.to_string(),
                            feature['geometry'],
                        )
                        for feature in src
                    ]
        except FionaValueError:
            labels = []

        if not labels:
            mask_data = np.zeros(shape=shape)
        else:
            mask_data = rasterize(
                labels,
                out_shape=shape,
                fill=0,  # nodata value
                transform=tfm,
                all_touched=False,
                dtype=np.uint8,
            )

        mask = torch.from_numpy(mask_data).long()

        return mask

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        img, tfm, raster_crs = self._load_image(files['image_path'])
        h, w = img.shape[1:]
        mask = self._load_mask(files['label_path'], tfm, raster_crs, (h, w))

        ch, cw = self.chip_size[self.image]
        sample = {'image': img[:, :ch, :cw], 'mask': mask[:ch, :cw]}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        root = os.path.join(self.root, self.dataset_id, self.split)
        for aoi in self.aois:
            # Check if the extracted files already exist
            if glob.glob(os.path.join(root, '**.tif'), recursive=True):
                continue

            # Check if the tarball has already been downloaded
            for tarball, md5 in zip(
                self.tarballs[self.split][aoi], self.md5s[self.split][aoi]
            ):
                if os.path.exists(os.path.join(root, tarball)):
                    extract_archive(os.path.join(root, tarball), root)

                # Check if the user requested to download the dataset
                if not self.download:
                    raise DatasetNotFoundError(self)

                # Download the dataset
                url = f's3://spacenet-dataset/spacenet/{self.dataset_id}/tarballs/{tarball}'
                aws = which('aws')
                aws('s3', 'cp', url, root)
                check_integrity(
                    os.path.join(root, tarball), md5 if self.checksum else None
                )
                extract_archive(os.path.join(root, tarball), root)

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

        .. versionadded:: 0.2
        """
        # image can be 1 channel or >3 channels
        if sample['image'].shape[0] == 1:
            image = np.rollaxis(sample['image'].numpy(), 0, 3)
        else:
            image = np.rollaxis(sample['image'][:3].numpy(), 0, 3)
        image = percentile_normalization(image, axis=(0, 1))

        ncols = 1
        show_mask = 'mask' in sample
        show_predictions = 'prediction' in sample

        if show_mask:
            mask = sample['mask'].numpy()
            ncols += 1

        if show_predictions:
            prediction = sample['prediction'].numpy()
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 8, 8))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        axs[0].imshow(image)
        axs[0].axis('off')
        if show_titles:
            axs[0].set_title('Image')

        if show_mask:
            axs[1].imshow(mask, interpolation='none')
            axs[1].axis('off')
            if show_titles:
                axs[1].set_title('Label')

        if show_predictions:
            axs[2].imshow(prediction, interpolation='none')
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class SpaceNet1(SpaceNet):
    """SpaceNet 1: Building Detection v1 Dataset.

    `SpaceNet 1 <https://spacenet.ai/spacenet-buildings-dataset-v1/>`_
    is a dataset of building footprints over the city of Rio de Janeiro.

    Dataset features:

    * No. of images: 6940 (8 Band) + 6940 (RGB)
    * No. of polygons: 382,534 building labels
    * Area Coverage: 2544 sq km
    * GSD: 1 m (8 band),  50 cm (rgb)
    * Chip size: 101 x 110 (8 band), 406 x 438 (rgb)

    Dataset format:

    * Imagery - Worldview-2 GeoTIFFs

        * 8Band.tif (Multispectral)
        * RGB.tif (Pansharpened RGB)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232
    """

    dataset_id = 'SN1_buildings'
    tarballs = {
        'train': {
            1: [
                'SN1_buildings_train_AOI_1_Rio_3band.tar.gz',
                'SN1_buildings_train_AOI_1_Rio_8band.tar.gz',
                'SN1_buildings_train_AOI_1_Rio_geojson_buildings.tar.gz',
            ]
        },
        'test': {
            1: [
                'SN1_buildings_test_AOI_1_Rio_3band.tar.gz',
                'SN1_buildings_test_AOI_1_Rio_8band.tar.gz',
            ]
        },
    }
    md5s = {
        'train': {
            1: [
                '279e334a2120ecac70439ea246174516',
                '6440a9eedbd7c4fe9741875135362c8c',
                'b6e02fbd727f252ea038abe4f77a77b3',
            ]
        },
        'test': {
            1: ['18283d78b21c239bc1831f3bf1d2c996', '732b3a40603b76e80aac84e002e2b3e8']
        },
    }

    imagery = {'rgb': 'RGB.tif', '8band': '8Band.tif'}
    chip_size = {'rgb': (406, 438), '8band': (101, 110)}


class SpaceNet2(SpaceNet):
    r"""SpaceNet 2: Building Detection v2 Dataset.

    `SpaceNet 2 <https://spacenet.ai/spacenet-buildings-dataset-v2/>`_
    is a dataset of building footprints over the cities of Las Vegas,
    Paris, Shanghai and Khartoum.

    Collection features:

    +------------+---------------------+------------+------------+
    |    AOI     | Area (km\ :sup:`2`\)| # Images   | # Buildings|
    +============+=====================+============+============+
    | Las Vegas  |    216              |   3850     |  151,367   |
    +------------+---------------------+------------+------------+
    | Paris      |    1030             |   1148     |  23,816    |
    +------------+---------------------+------------+------------+
    | Shanghai   |    1000             |   4582     |  92,015    |
    +------------+---------------------+------------+------------+
    | Khartoum   |    765              |   1012     |  35,503    |
    +------------+---------------------+------------+------------+

    Imagery features:

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1
        :stub-columns: 1

        *   -
            - PAN
            - MS
            - PS-MS
            - PS-RGB
        *   - GSD (m)
            - 0.31
            - 1.24
            - 0.30
            - 0.30
        *   - Chip size (px)
            - 650 x 650
            - 162 x 162
            - 650 x 650
            - 650 x 650

    Dataset format:

    * Imagery - Worldview-3 GeoTIFFs

        * PAN.tif (Panchromatic)
        * MS.tif (Multispectral)
        * PS-MS (Pansharpened Multispectral)
        * PS-RGB (Pansharpened RGB)

    * Labels - GeoJSON

        * label.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232
    """

    dataset_id = 'SN2_buildings'
    tarballs = {
        'train': {
            2: ['SN2_buildings_train_AOI_2_Vegas.tar.gz'],
            3: ['SN2_buildings_train_AOI_3_Paris.tar.gz'],
            4: ['SN2_buildings_train_AOI_4_Shanghai.tar.gz'],
            5: ['SN2_buildings_train_AOI_5_Khartoum.tar.gz'],
        },
        'test': {
            2: ['AOI_2_Vegas_Test_public.tar.gz'],
            3: ['AOI_3_Paris_Test_public.tar.gz'],
            4: ['AOI_4_Shanghai_Test_public.tar.gz'],
            5: ['AOI_5_Khartoum_Test_public.tar.gz'],
        },
    }
    md5s = {
        'train': {
            2: ['307da318bc43aaf9481828f92eda9126'],
            3: ['4db469e3e4e7bf025368ad730aec0888'],
            4: ['986129eecd3e842ebc2063d43b407adb'],
            5: ['462b4bf0466c945d708befabd4d9115b'],
        },
        'test': {
            2: ['d45405afd6629e693e2f9168b1291ea3'],
            3: ['2eaee95303e88479246e4ee2f2279b7f'],
            4: ['f51dc51fa484dc7fb89b3697bd15a950'],
            5: ['037d7be10530f0dd1c43d4ef79f3236e'],
        },
    }

    imagery = {
        'MS': 'MS.tif',
        'PAN': 'PAN.tif',
        'PS-MS': 'PS-MS.tif',
        'PS-RGB': 'PS-RGB.tif',
    }
    chip_size = {
        'MS': (162, 162),
        'PAN': (650, 650),
        'PS-MS': (650, 650),
        'PS-RGB': (650, 650),
    }
    label_glob = 'label.geojson'


class SpaceNet3(SpaceNet):
    r"""SpaceNet 3: Road Network Detection.

    `SpaceNet 3 <https://spacenet.ai/spacenet-roads-dataset/>`_
    is a dataset of road networks over the cities of Las Vegas, Paris, Shanghai,
    and Khartoum.

    Collection features:

    +------------+---------------------+------------+---------------------------+
    |    AOI     | Area (km\ :sup:`2`\)| # Images   | # Road Network Labels (km)|
    +============+=====================+============+===========================+
    | Vegas      |    216              |   854      |         3685              |
    +------------+---------------------+------------+---------------------------+
    | Paris      |    1030             |   257      |         425               |
    +------------+---------------------+------------+---------------------------+
    | Shanghai   |    1000             |   1028     |         3537              |
    +------------+---------------------+------------+---------------------------+
    | Khartoum   |    765              |   283      |         1030              |
    +------------+---------------------+------------+---------------------------+

    Imagery features:

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1
        :stub-columns: 1

        *   -
            - PAN
            - MS
            - PS-MS
            - PS-RGB
        *   - GSD (m)
            - 0.31
            - 1.24
            - 0.30
            - 0.30
        *   - Chip size (px)
            - 1300 x 1300
            - 325 x 325
            - 1300 x 1300
            - 1300 x 1300

    Dataset format:

    * Imagery - Worldview-3 GeoTIFFs

        * PAN.tif (Panchromatic)
        * MS.tif (Multispectral)
        * PS-MS (Pansharpened Multispectral)
        * PS-RGB (Pansharpened RGB)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1807.01232

    .. versionadded:: 0.3
    """

    dataset_id = 'SN3_roads'
    tarballs = {
        'train': {
            2: ['SN3_roads_train_AOI_2_Vegas.tar.gz', 'SN3_roads_train_AOI_2_Vegas_geojson_roads_speed.tar.gz'],
            3: ['SN3_roads_train_AOI_3_Paris.tar.gz', 'SN3_roads_train_AOI_3_Paris_geojson_roads_speed.tar.gz'],
            4: ['SN3_roads_train_AOI_4_Shanghai.tar.gz', 'SN3_roads_train_AOI_4_Shanghai_geojson_roads_speed.tar.gz'],
            5: ['SN3_roads_train_AOI_5_Khartoum.tar.gz', 'SN3_roads_train_AOI_5_Khartoum_geojson_roads_speed.tar.gz'],
        },
        'test': {
            2: ['SN3_roads_test_public_AOI_2_Vegas.tar.gz'],
            3: ['SN3_roads_test_public_AOI_3_Paris.tar.gz'],
            4: ['SN3_roads_test_public_AOI_4_Shanghai.tar.gz'],
            5: ['SN3_roads_test_public_AOI_5_Khartoum.tar.gz'],
        },
    }
    md5s = {
        'train': {
            2: ['06317255b5e0c6df2643efd8a50f22ae', '4acf7846ed8121db1319345cfe9fdca9'],
            3: ['c13baf88ee10fe47870c303223cabf82', 'abc8199d4c522d3a14328f4f514702ad'],
            4: ['ef3de027c3da734411d4333bee9c273b', 'f1db36bd17b2be2281f5f7d369e9e25d'],
            5: ['46f327b550076f87babb5f7b43f27c68', 'd969693760d59401a84bd9215375a636'],
        },
        'test': {
            2: ['e9eb2220888ba38cab175fc6db6799a2'],
            3: ['21098cfe471dba6208c92b37b8203ae9'],
            4: ['2e7438b870ffd33d4453366db1c5b317'],
            5: ['f367c79fa0fc1d38e63a0fdd065ed957'],
        },
    }

    imagery = {
        'MS': 'MS.tif',
        'PAN': 'PAN.tif',
        'PS-MS': 'PS-MS.tif',
        'PS-RGB': 'PS-RGB.tif',
    }
    chip_size = {
        'MS': (325, 325),
        'PAN': (1300, 1300),
        'PS-MS': (1300, 1300),
        'PS-RGB': (1300, 1300),
    }
    label_glob = 'labels.geojson'

    def __init__(
        self,
        root: Path = 'data',
        image: str = 'PS-RGB',
        speed_mask: bool | None = False,
        collections: list[str] = [],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet 3 Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection which must be in ["MS", "PAN", "PS-MS", "PS-RGB"]
            speed_mask: use multi-class speed mask (created by binning roads at
                10 mph increments) as label if true, else use binary mask
            collections: collection selection which must be a subset of:
                         [sn3_AOI_2_Vegas, sn3_AOI_3_Paris, sn3_AOI_4_Shanghai,
                         sn3_AOI_5_Khartoum]. If unspecified, all collections will be
                         used.
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert image in {'MS', 'PAN', 'PS-MS', 'PS-RGB'}
        self.speed_mask = speed_mask
        super().__init__(root, image, collections, transforms, download, checksum)

    def _load_mask(
        self, path: Path, tfm: Affine, raster_crs: CRS, shape: tuple[int, int]
    ) -> Tensor:
        """Rasterizes the dataset's labels (in geojson format).

        Args:
            path: path to the label
            tfm: transform of corresponding image
            raster_crs: CRS of raster file
            shape: shape of corresponding image

        Returns:
            Tensor: label tensor
        """
        min_speed_bin = 1
        max_speed_bin = 65
        speed_arr_bin = np.arange(min_speed_bin, max_speed_bin + 1)
        bin_size_mph = 10.0
        speed_cls_arr: np.typing.NDArray[np.int_] = np.array(
            [math.ceil(s / bin_size_mph) for s in speed_arr_bin]
        )

        try:
            with fiona.open(path) as src:
                vector_crs = CRS(src.crs)
                labels = []

                for feature in src:
                    if raster_crs != vector_crs:
                        geom = transform_geom(
                            vector_crs.to_string(),
                            raster_crs.to_string(),
                            feature['geometry'],
                        )
                    else:
                        geom = feature['geometry']

                    if self.speed_mask:
                        val = speed_cls_arr[
                            int(feature['properties']['inferred_speed_mph']) - 1
                        ]
                    else:
                        val = 1

                    labels.append((geom, val))

        except FionaValueError:
            labels = []

        if not labels:
            mask_data = np.zeros(shape=shape)
        else:
            mask_data = rasterize(
                labels,
                out_shape=shape,
                fill=0,  # nodata value
                transform=tfm,
                all_touched=False,
                dtype=np.uint8,
            )

        mask = torch.from_numpy(mask_data).long()
        return mask

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`SpaceNet.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        """
        # image can be 1 channel or >3 channels
        if sample['image'].shape[0] == 1:
            image = np.rollaxis(sample['image'].numpy(), 0, 3)
        else:
            image = np.rollaxis(sample['image'][:3].numpy(), 0, 3)
        image = percentile_normalization(image, axis=(0, 1))

        ncols = 1
        show_mask = 'mask' in sample
        show_predictions = 'prediction' in sample

        if show_mask:
            mask = sample['mask'].numpy()
            ncols += 1

        if show_predictions:
            prediction = sample['prediction'].numpy()
            ncols += 1

        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 8, 8))
        if not isinstance(axs, np.ndarray):
            axs = [axs]
        axs[0].imshow(image)
        axs[0].axis('off')
        if show_titles:
            axs[0].set_title('Image')

        if show_mask:
            if self.speed_mask:
                cmap = copy.copy(plt.get_cmap('autumn_r'))
                cmap.set_under(color='black')
                axs[1].imshow(mask, vmin=0.1, vmax=7, cmap=cmap, interpolation='none')
            else:
                axs[1].imshow(mask, cmap='Greys_r', interpolation='none')
            axs[1].axis('off')
            if show_titles:
                axs[1].set_title('Label')

        if show_predictions:
            if self.speed_mask:
                cmap = copy.copy(plt.get_cmap('autumn_r'))
                cmap.set_under(color='black')
                axs[2].imshow(
                    prediction, vmin=0.1, vmax=7, cmap=cmap, interpolation='none'
                )
            else:
                axs[2].imshow(prediction, cmap='Greys_r', interpolation='none')
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class SpaceNet4(SpaceNet):
    """SpaceNet 4: Off-Nadir Buildings Dataset.

    `SpaceNet 4 <https://spacenet.ai/off-nadir-building-detection/>`_ is a
    dataset of 27 WV-2 imagery captured at varying off-nadir angles and
    associated building footprints over the city of Atlanta. The off-nadir angle
    ranges from 7 degrees to 54 degrees.

    Dataset features:

    * No. of chipped images: 28,728 (PAN/MS/PS-RGBNIR)
    * No. of label files: 1064
    * No. of building footprints: >120,000
    * Area Coverage: 665 sq km
    * Chip size: 225 x 225 (MS), 900 x 900 (PAN/PS-RGBNIR)

    Dataset format:

    * Imagery - Worldview-2 GeoTIFFs

        * PAN.tif (Panchromatic)
        * MS.tif (Multispectral)
        * PS-RGBNIR (Pansharpened RGBNIR)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/1903.12239

    """

    collection_md5_dict = {'sn4_AOI_6_Atlanta': 'c597d639cba5257927a97e3eff07b753'}

    imagery = {'MS': 'MS.tif', 'PAN': 'PAN.tif', 'PS-RGBNIR': 'PS-RGBNIR.tif'}
    chip_size = {'MS': (225, 225), 'PAN': (900, 900), 'PS-RGBNIR': (900, 900)}
    label_glob = 'labels.geojson'

    angle_catalog_map = {
        'nadir': [
            '1030010003D22F00',
            '10300100023BC100',
            '1030010003993E00',
            '1030010003CAF100',
            '1030010002B7D800',
            '10300100039AB000',
            '1030010002649200',
            '1030010003C92000',
            '1030010003127500',
            '103001000352C200',
            '103001000307D800',
        ],
        'off-nadir': [
            '1030010003472200',
            '1030010003315300',
            '10300100036D5200',
            '103001000392F600',
            '1030010003697400',
            '1030010003895500',
            '1030010003832800',
        ],
        'very-off-nadir': [
            '10300100035D1B00',
            '1030010003CCD700',
            '1030010003713C00',
            '10300100033C5200',
            '1030010003492700',
            '10300100039E6200',
            '1030010003BDDC00',
            '1030010003CD4300',
            '1030010003193D00',
        ],
    }

    def __init__(
        self,
        root: Path = 'data',
        image: str = 'PS-RGBNIR',
        angles: list[str] = [],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet 4 Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection which must be in ["MS", "PAN", "PS-RGBNIR"]
            angles: angle selection which must be in ["nadir", "off-nadir",
                "very-off-nadir"]
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        collections = ['sn4_AOI_6_Atlanta']
        assert image in {'MS', 'PAN', 'PS-RGBNIR'}
        self.angles = angles
        if self.angles:
            for angle in self.angles:
                assert angle in self.angle_catalog_map.keys()
        super().__init__(root, image, collections, transforms, download, checksum)

    def _load_files(self, root: Path) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for each pair of image and label
        """
        files = []
        nadir = []
        offnadir = []
        veryoffnadir = []
        images = glob.glob(os.path.join(root, self.collections[0], '*', self.filename))
        images = sorted(images)

        catalog_id_pattern = re.compile(r'(_[A-Z0-9])\w+$')
        for imgpath in images:
            imgdir = os.path.basename(os.path.dirname(imgpath))
            match = catalog_id_pattern.search(imgdir)
            assert match is not None, 'Invalid image directory'
            catalog_id = match.group()[1:]

            lbl_dir = os.path.dirname(imgpath).split('-nadir')[0]

            lbl_path = os.path.join(f'{lbl_dir}-labels', self.label_glob)
            assert os.path.exists(lbl_path)

            _file = {'image_path': imgpath, 'label_path': lbl_path}
            if catalog_id in self.angle_catalog_map['very-off-nadir']:
                veryoffnadir.append(_file)
            elif catalog_id in self.angle_catalog_map['off-nadir']:
                offnadir.append(_file)
            elif catalog_id in self.angle_catalog_map['nadir']:
                nadir.append(_file)

        angle_file_map = {
            'nadir': nadir,
            'off-nadir': offnadir,
            'very-off-nadir': veryoffnadir,
        }

        if not self.angles:
            files.extend(nadir + offnadir + veryoffnadir)
        else:
            for angle in self.angles:
                files.extend(angle_file_map[angle])
        return files


class SpaceNet5(SpaceNet3):
    r"""SpaceNet 5: Automated Road Network Extraction and Route Travel Time Estimation.

    `SpaceNet 5 <https://spacenet.ai/sn5-challenge/>`_
    is a dataset of road networks over the cities of Moscow, Mumbai and San
    Juan (unavailable).

    Collection features:

    +------------+---------------------+------------+---------------------------+
    |    AOI     | Area (km\ :sup:`2`\)| # Images   | # Road Network Labels (km)|
    +============+=====================+============+===========================+
    | Moscow     |    1353             |   1353     |         3066              |
    +------------+---------------------+------------+---------------------------+
    | Mumbai     |    1021             |   1016     |         1951              |
    +------------+---------------------+------------+---------------------------+

    Imagery features:

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1
        :stub-columns: 1

        *   -
            - PAN
            - MS
            - PS-MS
            - PS-RGB
        *   - GSD (m)
            - 0.31
            - 1.24
            - 0.30
            - 0.30
        *   - Chip size (px)
            - 1300 x 1300
            - 325 x 325
            - 1300 x 1300
            - 1300 x 1300

    Dataset format:

    * Imagery - Worldview-3 GeoTIFFs

        * PAN.tif (Panchromatic)
        * MS.tif (Multispectral)
        * PS-MS (Pansharpened Multispectral)
        * PS-RGB (Pansharpened RGB)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please use the following citation:

    * The SpaceNet Partners, “SpaceNet5: Automated Road Network Extraction and
      Route Travel Time Estimation from Satellite Imagery”,
      https://spacenet.ai/sn5-challenge/

    .. versionadded:: 0.2
    """

    collection_md5_dict = {
        'sn5_AOI_7_Moscow': 'b18107f878152fe7e75444373c320cba',
        'sn5_AOI_8_Mumbai': '1f1e2b3c26fbd15bfbcdbb6b02ae051c',
    }

    imagery = {
        'MS': 'MS.tif',
        'PAN': 'PAN.tif',
        'PS-MS': 'PS-MS.tif',
        'PS-RGB': 'PS-RGB.tif',
    }
    chip_size = {
        'MS': (325, 325),
        'PAN': (1300, 1300),
        'PS-MS': (1300, 1300),
        'PS-RGB': (1300, 1300),
    }
    label_glob = 'labels.geojson'

    def __init__(
        self,
        root: Path = 'data',
        image: str = 'PS-RGB',
        speed_mask: bool | None = False,
        collections: list[str] = [],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet 5 Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection which must be in ["MS", "PAN", "PS-MS", "PS-RGB"]
            speed_mask: use multi-class speed mask (created by binning roads at
                10 mph increments) as label if true, else use binary mask
            collections: collection selection which must be a subset of:
                         [sn5_AOI_7_Moscow, sn5_AOI_8_Mumbai]. If unspecified, all
                         collections will be used.
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        super().__init__(
            root, image, speed_mask, collections, transforms, download, checksum
        )


class SpaceNet6(SpaceNet):
    r"""SpaceNet 6: Multi-Sensor All-Weather Mapping.

    `SpaceNet 6 <https://spacenet.ai/sn6-challenge/>`_ is a dataset
    of optical and SAR imagery over the city of Rotterdam.

    Collection features:

    +------------+---------------------+------------+-----------------------------+
    |    AOI     | Area (km\ :sup:`2`\)| # Images   | # Building Footprint Labels |
    +============+=====================+============+=============================+
    | Rotterdam  |    120              |   3401     |         48000               |
    +------------+---------------------+------------+-----------------------------+


    Imagery features:

    .. list-table::
        :widths: 10 10 10 10 10 10
        :header-rows: 1
        :stub-columns: 1

        *   -
            - PAN
            - RGBNIR
            - PS-RGB
            - PS-RGBNIR
            - SAR-Intensity
        *   - GSD (m)
            - 0.5
            - 2.0
            - 0.5
            - 0.5
            - 0.5
        *   - Chip size (px)
            - 900 x 900
            - 450 x 450
            - 900 x 900
            - 900 x 900
            - 900 x 900


    Dataset format:

    * Imagery - GeoTIFFs from Worldview-2 (optical) and Capella Space (SAR)

        * PAN.tif (Panchromatic)
        * RGBNIR.tif (Multispectral)
        * PS-RGB (Pansharpened RGB)
        * PS-RGBNIR (Pansharpened RGBNIR)
        * SAR-Intensity (SAR Intensity)

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2004.06500

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    .. versionadded:: 0.4
    """

    dataset_id = 'spacenet6'
    collections = ['sn6_AOI_11_Rotterdam']
    # This is actually the metadata hash
    collection_md5_dict = {'sn6_AOI_11_Rotterdam': '66f7312218fec67a1e0b3b02b22c95cc'}
    imagery = {
        'PAN': 'PAN.tif',
        'RGBNIR': 'RGBNIR.tif',
        'PS-RGB': 'PS-RGB.tif',
        'PS-RGBNIR': 'PS-RGBNIR.tif',
        'SAR-Intensity': 'SAR-Intensity.tif',
    }
    chip_size = {
        'PAN': (900, 900),
        'RGBNIR': (450, 450),
        'PS-RGB': (900, 900),
        'PS-RGBNIR': (900, 900),
        'SAR-Intensity': (900, 900),
    }
    label_glob = 'labels.geojson'

    def __init__(
        self,
        root: Path = 'data',
        image: str = 'PS-RGB',
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new SpaceNet 6 Dataset instance.

        Args:
            root: root directory where dataset can be found
            image: image selection which must be in ["PAN", "RGBNIR",
                "PS-RGB", "PS-RGBNIR", "SAR-Intensity"]
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.image = image  # For testing

        self.filename = self.imagery[image]
        self.transforms = transforms

        self.files = self._load_files(os.path.join(root, self.dataset_id))


class SpaceNet7(SpaceNet):
    """SpaceNet 7: Multi-Temporal Urban Development Challenge.

    `SpaceNet 7 <https://spacenet.ai/sn7-challenge/>`_ is a dataset which
    consist of medium resolution (4.0m) satellite imagery mosaics acquired from
    Planet Labs’ Dove constellation between 2017 and 2020. It includes ≈ 24
    images (one per month) covering > 100 unique geographies, and comprises >
    40,000 km2 of imagery and exhaustive polygon labels of building footprints
    therein, totaling over 11M individual annotations.

    Dataset features:

    * No. of train samples: 1423
    * No. of test samples: 466
    * No. of building footprints: 11,080,000
    * Area Coverage: 41,000 sq km
    * Chip size: 1023 x 1023
    * GSD: ~4m

    Dataset format:

    * Imagery - Planet Dove GeoTIFF

        * mosaic.tif

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2102.04420

    .. versionadded:: 0.2
    """

    collection_md5_dict = {
        'sn7_train_source': '9f8cc109d744537d087bd6ff33132340',
        'sn7_train_labels': '16f873e3f0f914d95a916fb39b5111b5',
        'sn7_test_source': 'e97914f58e962bba3e898f08a14f83b2',
    }

    imagery = {'img': 'mosaic.tif'}
    chip_size = {'img': (1023, 1023)}

    label_glob = 'labels.geojson'

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet 7 Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: split selection which must be in ["train", "test"]
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.split = split
        self.filename = self.imagery['img']
        self.transforms = transforms
        self.checksum = checksum

        assert split in {'train', 'test'}, 'Invalid split'

        if split == 'test':
            self.collections = ['sn7_test_source']
        else:
            self.collections = ['sn7_train_source', 'sn7_train_labels']

        self.files = self._load_files(root)

    def _load_files(self, root: Path) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for images and labels (if train split)
        """
        files = []
        if self.split == 'train':
            imgs = sorted(
                glob.glob(os.path.join(root, 'sn7_train_source', '*', self.filename))
            )
            lbls = sorted(
                glob.glob(os.path.join(root, 'sn7_train_labels', '*', self.label_glob))
            )
            for img, lbl in zip(imgs, lbls):
                files.append({'image_path': img, 'label_path': lbl})
        else:
            imgs = sorted(
                glob.glob(os.path.join(root, 'sn7_test_source', '*', self.filename))
            )
            for img in imgs:
                files.append({'image_path': img})
        return files

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data at that index
        """
        files = self.files[index]
        img, tfm, raster_crs = self._load_image(files['image_path'])
        h, w = img.shape[1:]

        ch, cw = self.chip_size['img']
        sample = {'image': img[:, :ch, :cw]}
        if self.split == 'train':
            mask = self._load_mask(files['label_path'], tfm, raster_crs, (h, w))
            sample['mask'] = mask[:ch, :cw]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
