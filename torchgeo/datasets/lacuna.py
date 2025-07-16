# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Lacuna African Field Boundaries dataset."""

import glob
import os
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import (
    Path,
    download_and_extract_archive,
    extract_archive,
    percentile_normalization,
)


class LacunaAfricanFieldBoundaries(NonGeoDataset):
    r"""Lacuna African Field Boundaries dataset.

    The `Lacuna African Field Boundaries <https://registry.opendata.aws/africa-field-boundary-labels/>`__
    dataset is a dataset for extracting field boundaries from Planet satellite imagery, specifically for
    the African continent.

    Dataset features:

    * 33,746 224 x 224 Planetscope images and masks from Africa
    * four spectral bands - BGRN (blue, green, red, near-infrared)
    * images are reprojected to EPSG:4326 and resampled to 0.000025 degrees (~3m/px)
    * 3-class semantic masks: background, field, field boundary

    Dataset format:

    * rasters are four-channel GeoTiffs with EPSG:4326 spatial reference system
    * masks are single-channel GeoTiffs with EPSG:4326 spatial reference system

    Dataset classes:

    0. background
    1. field
    2. field boundary

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2412.18483

    .. versionadded:: 0.8
    """

    classes = ('Background', 'Field', 'Field Boundary')
    url = 'https://hf.co/datasets/airg/lacuna-field-boundaries/resolve/7fb638c45f22fab28d795ff7660d40ace0003b7a/lacuna-field-boundaries.tar.gz'
    filename = 'lacuna-field-boundaries.tar.gz'
    md5 = '3261a146a7a1452f95820f4e76717754'
    image_dir = 'images'
    mask_dir = 'labels'
    all_bands = ('B04', 'B03', 'B02', 'B05')
    rgb_bands = ('B04', 'B03', 'B02')

    def __init__(
        self,
        root: Path = 'data',
        bands: Sequence[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new LacunaAfricanFieldBoundaries dataset instance.

        Args:
            root: root directory where dataset can be found
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found.
            AssertionError: If number of images and masks do not match
        """
        assert set(bands) <= set(self.all_bands)
        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.band_indices = [self.all_bands.index(b) for b in self.bands]
        self._verify()
        self.images, self.masks = self._load_files(self.root)

    def _load_files(self, root: Path) -> tuple[list[str], list[str]]:
        """Load the image and mask files from the dataset.

        Args:
            root: root directory where dataset can be found

        Returns:
            a tuple of lists containing the paths to the images and masks
        """
        images = sorted(glob.glob(os.path.join(root, self.image_dir, '*.tif')))
        masks = sorted(glob.glob(os.path.join(root, self.mask_dir, '*.tif')))
        assert len(images) == len(masks)
        assert len(images) > 0
        return images, masks

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(self.images[index])
        image = image[self.band_indices]

        mask = self._load_target(self.masks[index])
        sample = {'image': image, 'mask': mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.images)

    def _load_image(self, path: Path) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image

        Returns:
            the image
        """
        with rasterio.open(path) as img:
            array = img.read().astype(np.int32)
            tensor = torch.from_numpy(array).float()
        return tensor

    def _load_target(self, path: Path) -> Tensor:
        """Loads the target mask.

        Args:
            path: path to the mask

        Returns:
            the target mask
        """
        with rasterio.open(path) as img:
            array = img.read().squeeze(axis=0)
            tensor = torch.from_numpy(array).long()
        return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, self.image_dir)) and os.path.exists(
            os.path.join(self.root, self.mask_dir)
        ):
            return

        # Check if the zip file has already been downloaded
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            extract_archive(filepath, self.root)
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        return

    def _download(self) -> None:
        """Download the dataset and extract it."""
        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

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

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        try:
            rgb_indices = [self.bands.index(band) for band in self.rgb_bands]
        except ValueError as e:
            raise RGBBandsMissingError() from e

        image = sample['image'][rgb_indices].permute(1, 2, 0).float().numpy()
        image = percentile_normalization(image, lower=1, upper=99, axis=(0, 1))
        mask = sample['mask'].numpy().astype('uint8').squeeze()

        num_panels = 2
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            predictions = sample['prediction'].numpy().astype('uint8').squeeze()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(mask, vmin=0, vmax=2, cmap='gray', interpolation='none')
        axs[1].axis('off')
        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if showing_predictions:
            axs[2].imshow(
                predictions, vmin=0, vmax=2, cmap='gray', interpolation='none'
            )
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
