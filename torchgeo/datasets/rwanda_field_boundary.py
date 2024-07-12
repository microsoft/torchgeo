# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Rwanda Field Boundary Competition dataset."""

import glob
import os
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import NonGeoDataset
from .utils import which


class RwandaFieldBoundary(NonGeoDataset):
    """Rwanda Field Boundary Competition dataset.

    This dataset contains field boundaries for smallholder farms in eastern Rwanda.
    The Nasa Harvest program funded a team of annotators from TaQadam to label Planet
    imagery for the 2021 growing season for the purpose of conducting the Rwanda Field
    boundary detection Challenge. The dataset includes rasterized labeled field
    boundaries and time series satellite imagery from Planet's NICFI program.
    Planet's basemap imagery is provided for six months (March, April, August, October,
    November and December). Note: only fields that were big enough to be differentiated
    on the Planetscope imagery were labeled, only fields that were fully contained
    within the chips were labeled. The paired dataset is provided in 256x256 chips for a
    total of 70 tiles covering 1532 individual fields.

    The labels are provided as binary semantic segmentation labels:

    0. No field-boundary
    1. Field-boundary

    If you use this dataset in your research, please cite the following:

    * https://doi.org/10.34911/RDNT.G580WW

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.

    .. versionadded:: 0.5
    """

    url = 'https://radiantearth.blob.core.windows.net/mlhub/nasa_rwanda_field_boundary_competition'

    splits = {'train': 57, 'test': 13}
    dates = ('2021_03', '2021_04', '2021_08', '2021_10', '2021_11', '2021_12')
    all_bands = ('B01', 'B02', 'B03', 'B04')
    rgb_bands = ('B03', 'B02', 'B01')
    classes = ['No field-boundary', 'Field-boundary']

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new RwandaFieldBoundary instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
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

        self._verify()

    def __len__(self) -> int:
        """Return the number of chips in the dataset.

        Returns:
            length of the dataset
        """
        return self.splits[self.split]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            a dict containing image and mask at index.
        """
        images = []
        for date in self.dates:
            patches = []
            for band in self.bands:
                path = os.path.join(self.root, 'source', self.split, date)
                with rasterio.open(os.path.join(path, f'{index:02}_{band}.tif')) as src:
                    patches.append(src.read(1).astype(np.float32))
            images.append(patches)
        sample = {'image': torch.from_numpy(np.array(images))}

        if self.split == 'train':
            path = os.path.join(self.root, 'labels', self.split)
            with rasterio.open(os.path.join(path, f'{index:02}.tif')) as src:
                sample['mask'] = torch.from_numpy(src.read(1).astype(np.int64))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the subdirectories already exist and have the correct number of files
        path = os.path.join(self.root, 'source', self.split, '*', '*.tif')
        expected = len(self.dates) * self.splits[self.split] * len(self.all_bands)
        if len(glob.glob(path)) == expected:
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        os.makedirs(self.root, exist_ok=True)
        azcopy = which('azcopy')
        azcopy('sync', self.url, self.root, '--recursive=true')

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        time_step: int = 0,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            time_step: time step at which to access image, beginning with 0
            suptitle: optional string to use as a suptitle

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

        num_time_points = sample['image'].shape[0]
        assert time_step < num_time_points

        image = np.rollaxis(sample['image'][time_step, rgb_indices].numpy(), 0, 3)
        image = np.clip(image / 2000, 0, 1)

        if 'mask' in sample:
            mask = sample['mask'].numpy()
        else:
            mask = np.zeros_like(image)

        num_panels = 2
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            predictions = sample['prediction'].numpy()
            num_panels += 1

        fig, axs = plt.subplots(ncols=num_panels, figsize=(4 * num_panels, 4))

        axs[0].imshow(image)
        axs[0].axis('off')
        if show_titles:
            axs[0].set_title(f't={time_step}')

        axs[1].imshow(mask, vmin=0, vmax=1, interpolation='none')
        axs[1].axis('off')
        if show_titles:
            axs[1].set_title('Mask')

        if showing_predictions:
            axs[2].imshow(predictions, vmin=0, vmax=1, interpolation='none')
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
