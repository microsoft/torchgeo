# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SpaceNet abstract base class."""

import glob
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import torch
from fiona.errors import FionaError, FionaValueError
from fiona.transform import transform_geom
from matplotlib.figure import Figure
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from torch import Tensor

from ..errors import DatasetNotFoundError
from ..geo import NonGeoDataset
from ..utils import (
    Path,
    check_integrity,
    extract_archive,
    percentile_normalization,
    which,
)


class SpaceNet(NonGeoDataset, ABC):
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
    image_glob = '*.tif'
    mask_glob = '*.geojson'
    file_regex = r'_img(\d+)\.'
    chip_size: ClassVar[dict[str, tuple[int, int]]] = {}

    cities: ClassVar[dict[int, str]] = {
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
    @abstractmethod
    def dataset_id(self) -> str:
        """Dataset ID."""

    @property
    @abstractmethod
    def tarballs(self) -> dict[str, dict[int, list[str]]]:
        """Mapping of tarballs[split][aoi] = [tarballs]."""

    @property
    @abstractmethod
    def md5s(self) -> dict[str, dict[int, list[str]]]:
        """Mapping of md5s[split][aoi] = [md5s]."""

    @property
    @abstractmethod
    def valid_aois(self) -> dict[str, list[int]]:
        """Mapping of valid_aois[split] = [aois]."""

    @property
    @abstractmethod
    def valid_images(self) -> dict[str, list[str]]:
        """Mapping of valid_images[split] = [images]."""

    @property
    @abstractmethod
    def valid_masks(self) -> tuple[str, ...]:
        """List of valid masks."""

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        aois: list[int] = [],
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

        if self.split == 'train':
            assert len(self.images) == len(self.masks)

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.images)

    def _load_image(self, path: Path) -> tuple[Tensor, Affine, CRS]:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rio.open(path) as img:
            out_shape = (img.count, img.height, img.width)
            if self.image in self.chip_size:
                out_shape = (img.count, *self.chip_size[self.image])
            array = img.read(out_shape=out_shape, resampling=Resampling.bilinear)
            tensor = torch.from_numpy(array.astype(np.float32))
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
                labels = [
                    transform_geom(
                        vector_crs.to_string(),
                        raster_crs.to_string(),
                        feature['geometry'],
                    )
                    for feature in src
                    if feature['geometry']
                ]
        except (FionaError, FionaValueError):
            # Empty geojson files, geometries that cannot be transformed (SN7)
            labels = []

        if labels:
            mask = rasterize(
                labels,
                out_shape=shape,
                fill=0,  # nodata value
                transform=tfm,
                all_touched=False,
                dtype=np.int64,
            )
        else:
            mask = np.zeros(shape=shape, dtype=np.int64)

        return torch.from_numpy(mask)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image_path = self.images[index]
        img, tfm, raster_crs = self._load_image(image_path)
        h, w = img.shape[1:]
        sample = {'image': img}

        if self.split == 'train':
            mask_path = self.masks[index]
            mask = self._load_mask(mask_path, tfm, raster_crs, (h, w))
            sample['mask'] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _image_id(self, path: str) -> list[Any]:
        """Return the image ID.

        Args:
            path: An image or mask filepath.

        Returns:
            A list of integers.
        """
        keys: list[Any] = []
        if match := re.search(self.file_regex, path):
            for key in match.group(1).split('_'):
                try:
                    keys.append(int(key))
                except ValueError:
                    keys.append(key)

        return keys

    def _list_files(self, aoi: int) -> tuple[list[str], list[str]]:
        """List all files in a particular AOI.

        Args:
            aoi: Area of interest.

        Returns:
            Lists of image and mask files.
        """
        # Produce a list of files
        kwargs = {}
        if '{aoi}' in self.directory_glob:
            kwargs['aoi'] = aoi

        product_glob = os.path.join(
            self.root, self.dataset_id, self.split, self.directory_glob
        )
        image_glob = product_glob.format(product=self.image, **kwargs)
        mask_glob = product_glob.format(product=self.mask, **kwargs)
        images = glob.glob(os.path.join(image_glob, self.image_glob), recursive=True)
        masks = glob.glob(os.path.join(mask_glob, self.mask_glob), recursive=True)

        # Sort files based on image ID
        images.sort(key=self._image_id)
        masks.sort(key=self._image_id)

        # Remove images missing masks (SN3) or duplicate images (SN8)
        if self.split == 'train':
            images_iter = iter(images)
            images = []
            for mask in masks:
                mask_id = self._image_id(mask)
                for image in images_iter:
                    image_id = self._image_id(image)
                    if image_id == mask_id:
                        images.append(image)
                        break

        return images, masks

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        self.images = []
        self.masks = []
        root = os.path.join(self.root, self.dataset_id, self.split)
        os.makedirs(root, exist_ok=True)
        for aoi in self.aois:
            # Check if the extracted files already exist
            images, masks = self._list_files(aoi)
            if images:
                self.images.extend(images)
                self.masks.extend(masks)
                continue

            # Check if the tarball has already been downloaded
            for tarball, md5 in zip(
                self.tarballs[self.split][aoi], self.md5s[self.split][aoi]
            ):
                if os.path.exists(os.path.join(root, tarball)):
                    extract_archive(os.path.join(root, tarball), root)
                    continue

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
                images, masks = self._list_files(aoi)
                self.images.extend(images)
                self.masks.extend(masks)

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

        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(ncols * 8, 8))
        axs[0, 0].imshow(image)
        axs[0, 0].axis('off')
        if show_titles:
            axs[0, 0].set_title('Image')

        if show_mask:
            axs[0, 1].imshow(mask, interpolation='none')
            axs[0, 1].axis('off')
            if show_titles:
                axs[0, 1].set_title('Label')

        if show_predictions:
            axs[0, 2].imshow(prediction, interpolation='none')
            axs[0, 2].axis('off')
            if show_titles:
                axs[0, 2].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
