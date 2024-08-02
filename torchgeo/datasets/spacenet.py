# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet datasets."""

import abc
import glob
import os
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

    url = 's3://spacenet-dataset/spacenet/{dataset_id}/tarballs/{tarball}'
    directory_glob = os.path.join('**', 'AOI_{aoi}_*', '{product}')

    cities = {
        1: 'Rio',
        2: 'Vegas',
        3: 'Paris',
        4: 'Shanghai',
        5: 'Khartoum',
        6: 'Atlanta',
        7: 'Moscow',
        8: 'Mumbai',
        9: 'San Juan',
        10: 'Dar Es Salaam',
        11: 'Rotterdam',
    }

    @property
    @abc.abstractmethod
    def dataset_id(self) -> str:
        """Dataset ID."""

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
    def valid_aois(self) -> dict[str, list[int]]:
        """Mapping of valid_aois[split] = [aois]."""

    @property
    @abc.abstractmethod
    def valid_images(self) -> dict[str, list[str]]:
        """Mapping of valid_images[split] = [images]."""

    @property
    @abc.abstractmethod
    def valid_masks(self) -> list[str]:
        """List of valid masks."""

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        aois: list[str] = [],
        image: str | None = None,
        mask: str | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: 'train' or 'test' split
            aois: areas of interest
            image: image selection
            mask: mask selection
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: if True, download dataset and store it in the root directory.
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: If any invalid arguments are passed.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.split = split
        self.aois = aois or self.valid_aois[split]
        self.image = image or self.valid_images[split][0]
        self.mask = mask or self.valid_masks[0]
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        assert self.split in {'train', 'test'}
        assert set(self.aois) <= set(self.valid_aois[split])
        assert self.image in self.valid_images[split]
        assert self.mask in self.valid_masks

        self._verify()

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

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

    def _list_files(self, aoi: int) -> list[str]:
        """List all files in a particular AOI.

        Args:
            aoi: Area of interest.

        Returns:
            A list of files.
        """
        kwargs = {}
        if '{aoi}' in self.directory_glob:
            kwargs['aoi'] = aoi

        product_glob = os.path.join(
            self.root, self.dataset_id, self.split, self.directory_glob, '*.{ext}'
        )
        image_glob = product_glob.format(product=self.image, ext='tif', **kwargs)
        mask_glob = product_glob.format(product=self.mask, ext='geojson', **kwargs)
        images = sorted(glob.glob(image_glob, recursive=True))
        masks = sorted(glob.glob(mask_glob, recursive=True))
        return list(zip(images, masks))

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        self.files = []
        root = os.path.join(self.root, self.dataset_id, self.split)
        for aoi in self.aois:
            # Check if the extracted files already exist
            files = self._list_files(aoi)
            if files:
                self.files.extend(files)
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
                url = self.url.format(dataset_id=self.dataset_id, tarball=tarball)
                aws = which('aws')
                aws('s3', 'cp', url, root)
                check_integrity(
                    os.path.join(root, tarball), md5 if self.checksum else None
                )
                extract_archive(os.path.join(root, tarball), root)
                self.files.extend(self._list_files(aoi))

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

    directory_glob = '{product}'
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
    valid_aois = {'train': [1], 'test': [1]}
    valid_images = {'train': ['3band', '8band'], 'test': ['3band', '8band']}
    valid_masks = ['geojson']


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
    valid_aois = {'train': [2, 3, 4, 5], 'test': [2, 3, 4, 5]}
    valid_images = {
        'train': ['MUL', 'MUL-PanSharpen', 'PAN', 'RGB-PanSharpen'],
        'test': ['MUL', 'MUL-PanSharpen', 'PAN', 'RGB-PanSharpen'],
    }
    valid_masks = [os.path.join('geojson', 'buildings')]


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
            2: [
                'SN3_roads_train_AOI_2_Vegas.tar.gz',
                'SN3_roads_train_AOI_2_Vegas_geojson_roads_speed.tar.gz',
            ],
            3: [
                'SN3_roads_train_AOI_3_Paris.tar.gz',
                'SN3_roads_train_AOI_3_Paris_geojson_roads_speed.tar.gz',
            ],
            4: [
                'SN3_roads_train_AOI_4_Shanghai.tar.gz',
                'SN3_roads_train_AOI_4_Shanghai_geojson_roads_speed.tar.gz',
            ],
            5: [
                'SN3_roads_train_AOI_5_Khartoum.tar.gz',
                'SN3_roads_train_AOI_5_Khartoum_geojson_roads_speed.tar.gz',
            ],
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
    valid_aois = {'train': [2, 3, 4, 5], 'test': [2, 3, 4, 5]}
    valid_images = {
        'train': ['MS', 'PS-MS', 'PAN', 'PS-RGB'],
        'test': ['MUL', 'MUL-PanSharpen', 'PAN', 'RGB-PanSharpen'],
    }
    valid_masks = ['geojson_roads', 'geojson_roads_speed']


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

    directory_glob = os.path.join('**', '{product}')
    dataset_id = 'SN4_buildings'
    tarballs = {
        'train': {
            6: [
                'Atlanta_nadir7_catid_1030010003D22F00.tar.gz',
                'Atlanta_nadir8_catid_10300100023BC100.tar.gz',
                'Atlanta_nadir10_catid_1030010003993E00.tar.gz',
                'Atlanta_nadir10_catid_1030010003CAF100.tar.gz',
                'Atlanta_nadir13_catid_1030010002B7D800.tar.gz',
                'Atlanta_nadir14_catid_10300100039AB000.tar.gz',
                'Atlanta_nadir16_catid_1030010002649200.tar.gz',
                'Atlanta_nadir19_catid_1030010003C92000.tar.gz',
                'Atlanta_nadir21_catid_1030010003127500.tar.gz',
                'Atlanta_nadir23_catid_103001000352C200.tar.gz',
                'Atlanta_nadir25_catid_103001000307D800.tar.gz',
                'Atlanta_nadir27_catid_1030010003472200.tar.gz',
                'Atlanta_nadir29_catid_1030010003315300.tar.gz',
                'Atlanta_nadir30_catid_10300100036D5200.tar.gz',
                'Atlanta_nadir32_catid_103001000392F600.tar.gz',
                'Atlanta_nadir34_catid_1030010003697400.tar.gz',
                'Atlanta_nadir36_catid_1030010003895500.tar.gz',
                'Atlanta_nadir39_catid_1030010003832800.tar.gz',
                'Atlanta_nadir42_catid_10300100035D1B00.tar.gz',
                'Atlanta_nadir44_catid_1030010003CCD700.tar.gz',
                'Atlanta_nadir46_catid_1030010003713C00.tar.gz',
                'Atlanta_nadir47_catid_10300100033C5200.tar.gz',
                'Atlanta_nadir49_catid_1030010003492700.tar.gz',
                'Atlanta_nadir50_catid_10300100039E6200.tar.gz',
                'Atlanta_nadir52_catid_1030010003BDDC00.tar.gz',
                'Atlanta_nadir53_catid_1030010003193D00.tar.gz',
                'Atlanta_nadir53_catid_1030010003CD4300.tar.gz',
                'geojson.tar.gz',
            ]
        },
        'test': {6: ['SN4_buildings_AOI_6_Atlanta_test_public.tar.gz']},
    }
    md5s = {
        'train': {
            6: [
                'd41ab6ec087b07e1e046c55d1fa5754b',
                '72f04a7c0c34dd4595c181ee1ae6cb4c',
                '89559f42ac11a8de570cef9802a577ad',
                '5489ac756249c336ea506ef0acb3c09d',
                'bd9ed231cedd8631683ea51ea0602de1',
                'c497a8a448ed7ccdf63e7706507c0603',
                '45d54eeecefdc60aa38320be6f29a17c',
                '611528c0188bbc7e9cdf98609c6b0c49',
                '532fbf1ca73d3d2e8b03c585f61b7316',
                '538f48429b0968b6cfad97eb61fa8de1',
                '3c48e94bc6d9e66e27c3a9bc8d35d65d',
                'b78cdf951e7bf4fedbe9259abd1e047a',
                'f307ce3c623d12d5a2fd5acb1e0607e0',
                '9a17574332cd5513d68a0bcc9c607bdd',
                'fe905ca809f7bd2ceef75bde23c326f3',
                'd9f2e4a5c8462f6f9f7d5c573d9a1dc6',
                'f9425ff38dc82bf0e8f25a6287ff1ad1',
                '7a6005d6fd972d5ce04caf9b42b36897',
                '7c5aa16bb64cacf766cf88f89b3093bd',
                '8f7e959eb0156ad2dfb0b966a1de06a9',
                '62c4babcbe70034b7deb7c14d5ff61c2',
                '8001d75f67534edf6932242324b8c1a7',
                'bc299cb5de432b5f5a1ce65a3bdb0abc',
                'd7640eda7c4efaf825665e853037bec9',
                'd4e1931551e9d3c6fd9bf1d8adfd07a0',
                'b313e23ead8fe6e2c8671a49f2c9de37',
                '3bd8f07ad57bff841d0cf91c91c6f5ed',
                '2556339e26a09e57559452eb240ef29c',
            ]
        },
        'test': {6: ['0ec3874bfc19aed63b33ac47b039aace']},
    }
    valid_aois = {'train': [6], 'test': [6]}
    valid_images = {
        'train': ['MS', 'PAN', 'Pan-Sharpen'],
        'test': ['MS', 'PAN', 'Pan-Sharpen'],
    }
    valid_masks = [os.path.join('geojson', 'spacenet-buildings')]


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

    dataset_id = 'SN5_roads'
    tarballs = {
        'train': {
            7: ['SN5_roads_train_AOI_7_Moscow.tar.gz'],
            8: ['SN5_roads_train_AOI_8_Mumbai.tar.gz'],
        },
        'test': {9: ['SN5_roads_test_public_AOI_9_San_Juan.tar.gz']},
    }
    md5s = {
        'train': {
            7: ['03082d01081a6d8df2bc5a9645148d2a'],
            8: ['1ee20ba781da6cb7696eef9a95a5bdcc'],
        },
        'test': {9: ['fc45afef219dfd3a20f2d4fc597f6882']},
    }
    valid_aois = {'train': [7, 8], 'test': [9]}
    valid_images = {
        'train': ['MS', 'PAN', 'PS-MS', 'PS-RGB'],
        'test': ['MS', 'PAN', 'PS-MS', 'PS-RGB'],
    }
    valid_masks = ['geojson_roads_speed']


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

    .. versionadded:: 0.4
    """

    dataset_id = 'SN6_buildings'
    tarballs = {
        'train': {11: ['SN6_buildings_AOI_11_Rotterdam_train.tar.gz']},
        'test': {11: ['SN6_buildings_AOI_11_Rotterdam_test_public.tar.gz']},
    }
    md5s = {
        'train': {11: ['10ca26d2287716e3b6ef0cf0ad9f946e']},
        'test': {11: ['a07823a5e536feeb8bb6b6f0cb43cf05']},
    }
    valid_aois = {'train': [11], 'test': [11]}
    valid_images = {
        'train': ['PAN', 'PS-RGB', 'PS-RGBNIR', 'RGBNIR', 'SAR-Intensity'],
        'test': ['SAR-Intensity'],
    }
    valid_masks = ['geojson_buildings']


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

    directory_glob = os.path.join('**', '{product}')
    dataset_id = 'SN7_buildings'
    tarballs = {
        'train': {0: ['SN7_buildings_train.tar.gz', 'SN7_buildings_train_csvs.tar.gz']},
        'test': {0: ['SN7_buildings_test_public.tar.gz']},
    }
    md5s = {
        'train': {
            0: ['6eda13b9c28f6f5cdf00a7e8e218c1b1', '0266ffea18950b1472cedafa8bead7bb']
        },
        'test': {0: ['b3bde95a0f8f32f3bfeba49464b9bc97']},
    }
    valid_aois = {'train': [0], 'test': [0]}
    valid_images = {'train': ['images', 'images_masked'], 'test': ['images_masked']}
    valid_masks = ['labels', 'labels_match', 'labels_match_pix']


class SpaceNet8(SpaceNet):
    """SpaceNet8: Flood Detection Challenge Using Multiclass Segmentation.

    `SpaceNet 8 <https://spacenet.ai/sn8-challenge/>`_ is a dataset focusing on
    infrastructure and flood mapping related to hurricanes and heavy rains that cause
    route obstructions and significant damage.

    If you use this dataset in your research, please cite the following paper:

    * `SpaceNet 8 - The Detection of Flooded Roads and Buildings
       <https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Hansch_SpaceNet_8_-_The_Detection_of_Flooded_Roads_and_Buildings_CVPRW_2022_paper.html>`_

    .. versionadded:: 0.6
    """

    directory_glob = '{product}'
    dataset_id = 'SN8_floods'
    tarballs = {
        'train': {
            0: [
                'Germany_Training_Public.tar.gz',
                'Louisiana-East_Training_Public.tar.gz',
            ]
        },
        'test': {0: ['Louisiana-West_Test_Public.tar.gz']},
    }
    md5s = {
        'train': {
            0: ['81383a9050b93e8f70c8557d4568e8a2', 'fa40ae3cf6ac212c90073bf93d70bd95']
        },
        'test': {0: ['d41d8cd98f00b204e9800998ecf8427e']},
    }
    valid_aois = {'train': [0], 'test': [0]}
    valid_images = {
        'train': ['PRE-event', 'POST-event'],
        'test': ['PRE-event', 'POST-event'],
    }
    valid_masks = ['annotations']
