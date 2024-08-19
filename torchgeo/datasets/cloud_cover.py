# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Cloud Cover Detection Challenge dataset."""

import os
from collections.abc import Callable, Sequence
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import Path, which


class CloudCoverDetection(NonGeoDataset):
    """Sentinel-2 Cloud Cover Segmentation Dataset.

    This training dataset was generated as part of a `crowdsourcing competition
    <https://www.drivendata.org/competitions/83/cloud-cover/>`_ on DrivenData.org, and
    later on was validated using a team of expert annotators. See `this website
    <https://beta.source.coop/radiantearth/cloud-cover-detection-challenge/>`__
    for dataset details.

    The dataset consists of Sentinel-2 satellite imagery and corresponding cloudy
    labels stored as GeoTiffs. There are 22,728 chips in the training data,
    collected between 2018 and 2020.

    Each chip has:

    * 4 multi-spectral bands from Sentinel-2 L2A product. The four bands are
      [B02, B03, B04, B08] (refer to Sentinel-2 documentation for more
      information about the bands).
    * Label raster for the corresponding source tile representing a binary
      classification for if the pixel is a cloud or not.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.34911/RDNT.HFQ6M7

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.

    .. versionadded:: 0.4
    """

    url = 'https://radiantearth.blob.core.windows.net/mlhub/ref_cloud_cover_detection_challenge_v1/final'
    all_bands = ('B02', 'B03', 'B04', 'B08')
    rgb_bands = ('B04', 'B03', 'B02')
    splits: ClassVar[dict[str, str]] = {'train': 'public', 'test': 'private'}

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        """Initiatlize a CloudCoverDetection instance.

        Args:
            root: root directory where dataset can be found
            split: 'train' or 'test'
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            AssertionError: If *split* or *bands* are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits
        assert set(bands) <= set(self.all_bands)

        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.download = download

        self.csv = os.path.join(self.root, self.split, f'{self.split}_metadata.csv')
        self._verify()

        self.metadata = pd.read_csv(self.csv)

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            length of dataset in integer
        """
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Returns a sample from dataset.

        Args:
            index: index to return

        Returns:
            data and label at given index
        """
        chip_id = self.metadata.iat[index, 0]
        image = self._load_image(chip_id)
        label = self._load_target(chip_id)
        sample = {'image': image, 'mask': label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, chip_id: str) -> Tensor:
        """Load all source images for a chip.

        Args:
            chip_id: ID of the chip.

        Returns:
            a tensor of stacked source image data
        """
        path = os.path.join(self.root, self.split, f'{self.split}_features', chip_id)
        images = []
        for band in self.bands:
            with rasterio.open(os.path.join(path, f'{band}.tif')) as src:
                images.append(src.read(1).astype(np.float32))
        return torch.from_numpy(np.stack(images, axis=0))

    def _load_target(self, chip_id: str) -> Tensor:
        """Load label image for a chip.

        Args:
            chip_id: ID of the chip.

        Returns:
            a tensor of the label image data
        """
        path = os.path.join(self.root, self.split, f'{self.split}_labels')
        with rasterio.open(os.path.join(path, f'{chip_id}.tif')) as src:
            return torch.from_numpy(src.read(1).astype(np.int64))

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if os.path.exists(self.csv):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        directory = os.path.join(self.root, self.split)
        os.makedirs(directory, exist_ok=True)
        url = f'{self.url}/{self.splits[self.split]}'
        azcopy = which('azcopy')
        azcopy('sync', url, directory, '--recursive=true')

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
            time_step: time step at which to access image, beginning with 0
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        if 'prediction' in sample:
            prediction = sample['prediction']
            n_cols = 3
        else:
            n_cols = 2

        image, mask = sample['image'] / 3000, sample['mask']

        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(10, n_cols * 5))

        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].axis('off')
        axs[1].imshow(mask)
        axs[1].axis('off')

        if 'prediction' in sample:
            axs[2].imshow(prediction)
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Prediction')

        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
