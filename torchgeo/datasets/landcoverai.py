# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""LandCover.ai dataset."""

import abc
import glob
import hashlib
import os
from collections.abc import Callable
from functools import lru_cache
from typing import Any, ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from PIL import Image
from rasterio.crs import CRS
from torch import Tensor
from torch.utils.data import Dataset

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset, RasterDataset
from .utils import BoundingBox, Path, download_url, extract_archive, working_dir


class LandCoverAIBase(Dataset[dict[str, Any]], abc.ABC):
    r"""Abstract base class for LandCover.ai Geo and NonGeo datasets.

    The `LandCover.ai <https://landcover.ai.linuxpolska.com/>`__ (Land Cover from
    Aerial Imagery) dataset is a dataset for automatic mapping of buildings, woodlands,
    water and roads from aerial images. This implementation is specifically for
    Version 1 of LandCover.ai.

    Dataset features:

    * land cover from Poland, Central Europe
    * three spectral bands - RGB
    * 33 orthophotos with 25 cm per pixel resolution (~9000x9500 px)
    * 8 orthophotos with 50 cm per pixel resolution (~4200x4700 px)
    * total area of 216.27 km\ :sup:`2`

    Dataset format:

    * rasters are three-channel GeoTiffs with EPSG:2180 spatial reference system
    * masks are single-channel GeoTiffs with EPSG:2180 spatial reference system

    Dataset classes:

    1. building (1.85 km\ :sup:`2`\ )
    2. woodland (72.02 km\ :sup:`2`\ )
    3. water (13.15 km\ :sup:`2`\ )
    4. road (3.5 km\ :sup:`2`\ )

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2005.02264v4

    .. versionadded:: 0.5
    """

    url = 'https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip'
    filename = 'landcover.ai.v1.zip'
    md5 = '3268c89070e8734b4e91d531c0617e03'
    classes = ('Background', 'Building', 'Woodland', 'Water', 'Road')
    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {
        0: (0, 0, 0, 0),
        1: (97, 74, 74, 255),
        2: (38, 115, 0, 255),
        3: (0, 197, 255, 255),
        4: (207, 207, 207, 255),
    }

    def __init__(
        self, root: Path = 'data', download: bool = False, checksum: bool = False
    ) -> None:
        """Initialize a new LandCover.ai dataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.download = download
        self.checksum = checksum

        lc_colors = np.zeros((max(self.cmap.keys()) + 1, 4))
        lc_colors[list(self.cmap.keys())] = list(self.cmap.values())
        lc_colors = lc_colors[:, :3] / 255
        self._lc_cmap = ListedColormap(lc_colors)

        self._verify()

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        if self._verify_data():
            return

        # Check if the zip file has already been downloaded
        pathname = os.path.join(self.root, self.filename)
        if os.path.exists(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    @abc.abstractmethod
    def __getitem__(self, query: Any) -> dict[str, Any]:
        """Retrieve image, mask and metadata indexed by index.

        Args:
            query: coordinates or an index

        Returns:
            sample of image, mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """

    @abc.abstractmethod
    def _verify_data(self) -> bool:
        """Verify if the images and masks are present."""

    def _download(self) -> None:
        """Download the dataset."""
        download_url(self.url, self.root, md5=self.md5 if self.checksum else None)

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.filename))

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
        image = np.rollaxis(sample['image'].numpy().astype('uint8').squeeze(), 0, 3)
        mask = sample['mask'].numpy().astype('uint8').squeeze()

        num_panels = 2
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            predictions = sample['prediction'].numpy()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(mask, vmin=0, vmax=4, cmap=self._lc_cmap, interpolation='none')
        axs[1].axis('off')
        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if showing_predictions:
            axs[2].imshow(
                predictions, vmin=0, vmax=4, cmap=self._lc_cmap, interpolation='none'
            )
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class LandCoverAIGeo(LandCoverAIBase, RasterDataset):
    """LandCover.ai Geo dataset.

    See the abstract LandCoverAIBase class to find out more.

    .. versionadded:: 0.5
    """

    filename_glob = os.path.join('images', '*.tif')
    filename_regex = '.*tif'

    def __init__(
        self,
        root: Path = 'data',
        crs: CRS | None = None,
        res: float | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LandCover.ai NonGeo dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        LandCoverAIBase.__init__(self, root, download, checksum)
        RasterDataset.__init__(self, root, crs, res, transforms=transforms, cache=cache)

    def _verify_data(self) -> bool:
        """Verify if the images and masks are present."""
        img_query = os.path.join(self.root, 'images', '*.tif')
        mask_query = os.path.join(self.root, 'masks', '*.tif')
        images = glob.glob(img_query)
        masks = glob.glob(mask_query)
        return len(images) > 0 and len(images) == len(masks)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image, mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        img_filepaths = cast(list[str], [hit.object for hit in hits])
        mask_filepaths = [
            str(path).replace('images', 'masks') for path in img_filepaths
        ]

        if not img_filepaths:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        img = self._merge_files(img_filepaths, query, self.band_indexes)
        mask = self._merge_files(mask_filepaths, query, self.band_indexes)
        sample = {
            'crs': self.crs,
            'bounds': query,
            'image': img.float(),
            'mask': mask.long(),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class LandCoverAI(LandCoverAIBase, NonGeoDataset):
    """LandCover.ai dataset.

    See the abstract LandCoverAIBase class to find out more.

    .. note::

       This dataset requires the following additional library to be installed:

       * `opencv-python <https://pypi.org/project/opencv-python/>`_ to generate
         the train/val/test split
    """

    sha256 = '15ee4ca9e3fd187957addfa8f0d74ac31bc928a966f76926e11b3c33ea76daa1'

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LandCover.ai dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in ['train', 'val', 'test']

        super().__init__(root, download, checksum)

        self.transforms = transforms
        self.split = split
        with open(os.path.join(self.root, split + '.txt')) as f:
            self.ids = f.readlines()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        id_ = self.ids[index].rstrip()
        sample = {'image': self._load_image(id_), 'mask': self._load_target(id_)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.ids)

    @lru_cache
    def _load_image(self, id_: str) -> Tensor:
        """Load a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the image
        """
        filename = os.path.join(self.root, 'output', id_ + '.jpg')
        with Image.open(filename) as img:
            array: np.typing.NDArray[np.int_] = np.array(img)
            tensor = torch.from_numpy(array).float()
            # Convert from HxWxC to CxHxW
            tensor = tensor.permute((2, 0, 1))
            return tensor

    @lru_cache
    def _load_target(self, id_: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            id_: unique ID of the image

        Returns:
            the target mask
        """
        filename = os.path.join(self.root, 'output', id_ + '_m.png')
        with Image.open(filename) as img:
            array: np.typing.NDArray[np.int_] = np.array(img.convert('L'))
            tensor = torch.from_numpy(array).long()
            return tensor

    def _verify_data(self) -> bool:
        """Verify if the images and masks are present."""
        img_query = os.path.join(self.root, 'output', '*_*.jpg')
        mask_query = os.path.join(self.root, 'output', '*_*_m.png')
        images = glob.glob(img_query)
        masks = glob.glob(mask_query)
        return len(images) > 0 and len(images) == len(masks)

    def _extract(self) -> None:
        """Extract the dataset.

        Raises:
            AssertionError: if the checksum of split.py does not match
        """
        super()._extract()

        # Generate train/val/test splits
        # Always check the sha256 of this file before executing
        # to avoid malicious code injection
        with working_dir(self.root):
            with open('split.py') as f:
                split = f.read().encode('utf-8')
                assert hashlib.sha256(split).hexdigest() == self.sha256
                exec(split)
