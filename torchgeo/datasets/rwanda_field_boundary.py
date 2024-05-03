# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Rwanda Field Boundary Competition dataset."""

import os
from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoDataset
from .utils import (
    DatasetNotFoundError,
    RGBBandsMissingError,
    check_integrity,
    download_radiant_mlhub_collection,
    extract_archive,
)


class RwandaFieldBoundary(NonGeoDataset):
    r"""Rwanda Field Boundary Competition dataset.

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

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    .. versionadded:: 0.5
    """

    dataset_id = 'nasa_rwanda_field_boundary_competition'
    collection_ids = [
        'nasa_rwanda_field_boundary_competition_source_train',
        'nasa_rwanda_field_boundary_competition_labels_train',
        'nasa_rwanda_field_boundary_competition_source_test',
    ]
    number_of_patches_per_split = {'train': 57, 'test': 13}

    filenames = {
        'train_images': 'nasa_rwanda_field_boundary_competition_source_train.tar.gz',
        'test_images': 'nasa_rwanda_field_boundary_competition_source_test.tar.gz',
        'train_labels': 'nasa_rwanda_field_boundary_competition_labels_train.tar.gz',
    }
    md5s = {
        'train_images': '1f9ec08038218e67e11f82a86849b333',
        'test_images': '17bb0e56eedde2e7a43c57aa908dc125',
        'train_labels': '10e4eb761523c57b6d3bdf9394004f5f',
    }

    dates = ('2021_03', '2021_04', '2021_08', '2021_10', '2021_11', '2021_12')

    all_bands = ('B01', 'B02', 'B03', 'B04')
    rgb_bands = ('B03', 'B02', 'B01')

    classes = ['No field-boundary', 'Field-boundary']

    splits = ['train', 'test']

    def __init__(
        self,
        root: str = 'data',
        split: str = 'train',
        bands: Sequence[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        api_key: str | None = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new RwandaFieldBoundary instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "test"
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self._validate_bands(bands)
        assert split in self.splits
        if download and api_key is None:
            raise RuntimeError('Must provide an API key to download the dataset')
        self.root = root
        self.bands = bands
        self.transforms = transforms
        self.split = split
        self.download = download
        self.api_key = api_key
        self.checksum = checksum
        self._verify()

        self.image_filenames: list[list[list[str]]] = []
        self.mask_filenames: list[str] = []
        for i in range(self.number_of_patches_per_split[split]):
            dates = []
            for date in self.dates:
                patch = []
                for band in self.bands:
                    fn = os.path.join(
                        self.root,
                        f'nasa_rwanda_field_boundary_competition_source_{split}',
                        f'nasa_rwanda_field_boundary_competition_source_{split}_{i:02d}_{date}',  # noqa: E501
                        f'{band}.tif',
                    )
                    patch.append(fn)
                dates.append(patch)
            self.image_filenames.append(dates)
            self.mask_filenames.append(
                os.path.join(
                    self.root,
                    f'nasa_rwanda_field_boundary_competition_labels_{split}',
                    f'nasa_rwanda_field_boundary_competition_labels_{split}_{i:02d}',
                    'raster_labels.tif',
                )
            )

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            a dict containing image, mask, transform, crs, and metadata at index.
        """
        img_fns = self.image_filenames[index]
        mask_fn = self.mask_filenames[index]

        imgs = []
        for date_fns in img_fns:
            bands = []
            for band_fn in date_fns:
                with rasterio.open(band_fn) as f:
                    bands.append(f.read(1).astype(np.int32))
            imgs.append(bands)
        img = torch.from_numpy(np.array(imgs))

        sample = {'image': img}

        if self.split == 'train':
            with rasterio.open(mask_fn) as f:
                mask = f.read(1)
            mask = torch.from_numpy(mask)
            sample['mask'] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of chips in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.image_filenames)

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided sequence of bands to load

        Raises:
            ValueError: if an invalid band name is provided
        """
        for band in bands:
            if band not in self.all_bands:
                raise ValueError(f"'{band}' is an invalid band name.")

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the subdirectories already exist and have the correct number of files
        checks = []
        for split, num_patches in self.number_of_patches_per_split.items():
            path = os.path.join(
                self.root, f'nasa_rwanda_field_boundary_competition_source_{split}'
            )
            if os.path.exists(path):
                num_files = len(os.listdir(path))
                # 6 dates + 1 collection.json file
                checks.append(num_files == (num_patches * 6) + 1)
            else:
                checks.append(False)

        if all(checks):
            return

        # Check if tar file already exists (if so then extract)
        have_all_files = True
        for group in ['train_images', 'train_labels', 'test_images']:
            filepath = os.path.join(self.root, self.filenames[group])
            if os.path.exists(filepath):
                if self.checksum and not check_integrity(filepath, self.md5s[group]):
                    raise RuntimeError('Dataset found, but corrupted.')
                extract_archive(filepath)
            else:
                have_all_files = False
        if have_all_files:
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset and extract it."""
        for collection_id in self.collection_ids:
            download_radiant_mlhub_collection(collection_id, self.root, self.api_key)

        for group in ['train_images', 'train_labels', 'test_images']:
            filepath = os.path.join(self.root, self.filenames[group])
            if self.checksum and not check_integrity(filepath, self.md5s[group]):
                raise RuntimeError('Dataset not found or corrupted.')
            extract_archive(filepath, self.root)

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
