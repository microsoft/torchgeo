# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Lacuna African Field Boundaries dataset."""

import glob
import os
from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, percentile_normalization


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

    .. note::

       This dataset can be automatically downloaded using the following bash script:

       .. code-block:: bash

          aws s3 sync --no-sign-request s3://africa-field-boundary-labels/ .

    .. versionadded:: 0.8
    """

    classes = ('Background', 'Field', 'Field Boundary')

    def __init__(
        self,
        root: Path = 'data',
        bands: Literal['rgb', 'all'] = 'all',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    ) -> None:
        """Initialize a new LacunaAfricanFieldBoundaries dataset instance.

        Args:
            root: root directory where dataset can be found
            bands: load all RGBN bands or RGB only. One of 'rgb' or 'all'.
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.images = sorted(glob.glob(os.path.join(root, 'images', '*.tif')))
        self.masks = sorted(glob.glob(os.path.join(root, 'labels', '*.tif')))

        if len(self.images) == 0 or (len(self.images) != len(self.masks)):
            raise DatasetNotFoundError(self)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(self.images[index])

        if self.bands == 'rgb':
            image = image[[3, 2, 1]]

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
        image = sample['image'][:3].permute(1, 2, 0).float().numpy()
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
