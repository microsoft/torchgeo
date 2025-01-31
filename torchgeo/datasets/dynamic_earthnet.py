# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dynamic EarthNet dataset."""

import os
import re
from collections.abc import Callable, Sequence
from typing import ClassVar

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    array_to_tensor,
    download_and_extract_archive,
    extract_archive,
    check_integrity,
)


class DynamicEarthNet(NonGeoDataset):
    """Dynamic EarthNet dataset.

    Dynamic EarthNet is a global multi-modal time-series dataset with and manually annotated land use and land cover labels.
    It covers 75 different areas with daily revisit times over the years 2018 and 2019 around the globe and can be used for
    semantic segmentation and change detection with time-series imagery. The annotated label corresponds to a
    land use and land cover map for the entire month. So depending on the temporal input, the dataset constructs time
    series of images and labels for each area, where the input images cover the entire month and the label corresponds to
    the same month as the first image in the time series.

    The Planet data consists of a 'PF-SR' directory containing daily images. There is an additional 'PF-QA' directory
    containing quality assurance information denoting which parts of the data are raw observations and which parts are
    gap-filled with temporally close observations. The QA product gives the distance and direction to
    the day of the observation. For example, a pixel value of -1 implies that the pixel has been filled from the previous day.
    Only the 'PF-SR' directory is used in this dataset.

    As Sentinel-1 and Sentinel-2 data are composites and potentially have quality issues. See the appendix of the
    original paper for more details.

    Dataset features:

    * 54,750 PlanetFusion images at 3m GSD with daily revisit times
    * 75 areas around the globe with 730 images each
    * Sentinel 1 and 2 monthly composites
    * 1,800 annotated images representintg land use and land cover labels for the first day of each month, so
      24 annotations per area

    Dataset format:

    * Image and label pixel dimensions: 1024x1024
    * Planet imagery in GeoTIFF format
    * Sentinel-1 imagery in GeoTIFF format with 8 channels
    * Sentinel-2 imagery in GeoTIFF format with 12 channels
    * Annotation labels in raster and vector format

    Dataset classes:

    1. impervious surface
    2. agricultural
    3. forest & other vegetation
    4. wetlands
    5. soil
    6. water
    7. snow & ice

    .. note::

        If you choose to return 'daily' data with the *temporal_input* argument, the dataset will return all images for a given month.
        And since months have different number of days, you will need to create a collate function for your accompanying DataLoader.

        Also if you choose additional modalities, there are some missing files in the dataset, so choosing additional modalities
        will reduce the dataset corpus by 62 samples.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2203.12560

    .. versionadded:: 0.7
    """

    valid_splits = 'train'
    valid_temporal_input = ('daily', 'weekly', 'monthly')

    base_url = 'https://hf.co/datasets/torchgeo/dynamic_earthnet/resolve/commit_hash/{}'

    filename_and_md5: ClassVar[dict[str, dict[str, str]]] = {
        'planet': {
            'filename': 'planet_pf_sr.tar.gz',
            'md5': 'd41d8cd98f00b204e9800998ecf8427e',
        },
        'sentinel1': {
            'filename': 'sentinel1.tar.gz',
            'md5': 'd41d8cd98f00b204e9800998ecf8427e',
        },
        'sentinel2': {
            'filename': 'sentinel2.tar.gz',
            'md5': 'd41d8cd98f00b204e9800998ecf8427e',
        },
        'labels': {
            'filename': 'labels.tar.gz',
            'md5': 'd41d8cd98f00b204e9800998ecf8427e',
        },
        'split_info': {
            'filename': 'split_info.tar.gz',
            'md5': 'd41d8cd98f00b204e9800998ecf8427e',
        },
    }

    valid_modalities = ('s1', 's2')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        temporal_input: str = 'daily',
        year_month_start: str = '2018-01',
        year_month_end: str = '2019-12',
        add_modalities: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dynamic EarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: currently only 'train' is available
            temporal_input: one of "daily","weekly", "monthly", where daily will return
                all images for a given month, weekly will return 6 images spaced by 5 days, and
                monthly will return 1 image corresponding to the same day of the month as the label
            year_month_start: start date to include in dataset
            year_month_end: end date to include in dataset
            add_modalities: whether to include addtional modalities of Sentinel-1 and/or Sentinel-2
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: If any arguments are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        # check that the dates are in correct format of YYYY-MM-DD with a regex
        assert re.match(r'\d{4}-\d{2}', year_month_start), (
            'start_date must be in format YYYY-MM'
        )
        assert re.match(r'\d{4}-\d{2}', year_month_end), (
            'end_date must be in format YYYY-MM'
        )
        # check that they fall between 2018-01-01 and 2019-12-31 with datetime
        year_month_start = pd.to_datetime(year_month_start)
        year_month_end = pd.to_datetime(year_month_end)
        assert year_month_start >= pd.to_datetime('2018-01'), (
            'start_date must be after 2018-01'
        )
        assert year_month_end <= pd.to_datetime('2019-12'), (
            'end_date must be before 2019-12'
        )

        assert split in self.valid_splits, f'Split must be one of {self.valid_splits}'
        assert temporal_input in self.valid_temporal_input, (
            f'Temporal type must be one of {self.valid_temporal_input}'
        )

        if add_modalities is not None:
            assert all(
                modality in self.valid_modalities for modality in add_modalities
            ), f'Modalities must be sequence of {self.valid_modalities}'

        self.root = root
        self.split = split
        self.temporal_input = temporal_input
        self.year_month_start = year_month_start
        self.year_month_end = year_month_end
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.add_modalities = add_modalities

        self._verify()

        self.metadata_df = pd.read_parquet(
            os.path.join(self.root, 'split_info', 'splits.parquet')
        )
        self.metadata_df = self.metadata_df[
            self.metadata_df['split'] == split
        ].reset_index(drop=True)

        # filter by dates
        self.metadata_df['year_month'] = pd.to_datetime(self.metadata_df['year_month'])
        self.metadata_df = self.metadata_df[
            (self.metadata_df['year_month'] >= year_month_start)
            & (self.metadata_df['year_month'] <= year_month_end)
        ].reset_index(drop=True)
        # filter for missing
        self.metadata_df = self.metadata_df[
            (self.metadata_df['missing_label'] == False)
        ].reset_index(drop=True)
        # if additional modalities are chosen filter for missing
        if self.add_modalities:
            for modality in self.add_modalities:
                self.metadata_df = self.metadata_df[
                    (self.metadata_df[f'missing_{modality}'] == False)
                ].reset_index(drop=True)

        # collect file paths
        self.sample_file_paths = self.collect_file_paths()

    def collect_file_paths(self) -> list[dict[str, str]]:
        """Collect file paths for the dataset.

        Returns:
            list of dictionaries with file paths
        """
        sample_files: list[dict[str, str]] = []

        def get_image_paths(row) -> dict[str, str]:
            """Get the image paths for a given row in the metadata dataframe according to temporal type.

            Args:
                row: row in the metadata dataframe

            Returns:
                list of image paths and label path
            """
            planet_paths = []
            # removing leading slash from label path for join
            image_path = os.path.join(self.root, row['planet_path'])
            label_path = os.path.join(self.root, row['label_path'])
            date_str_year_month = row['year_month'].strftime('%Y-%m')
            match self.temporal_input:
                case 'daily':
                    # check how many days are in this month and return all days in a month
                    days_in_month = pd.Period(date_str_year_month).days_in_month
                    days = [str(day).zfill(2) for day in range(1, days_in_month + 1)]
                case 'weekly':
                    # https://github.com/aysim/dynnet/blob/1e7d90294b54f52744ae2b35db10b4d0a48d093d/data/utae_dynamicen.py#L87
                    # get 6 images spaced by 5 days
                    days = ['01', '05', '10', '15', '20', '25']
                case 'monthly':
                    # get 1 image corresponding to the same day of the month as the label
                    days = ['01']

            for day in days:
                planet_paths.append(
                    os.path.join(
                        self.root, image_path, f'{date_str_year_month}-{day}.tif'
                    )
                )

            path_dict = {
                'date': date_str_year_month,
                'label_path': os.path.join(self.root, label_path),
                'planet_paths': planet_paths,
            }

            if self.add_modalities:
                for modality in self.add_modalities:
                    match modality:
                        case 's1':
                            path_dict['s1_path'] = os.path.join(
                                self.root, row['s1_path']
                            )
                        case 's2':
                            path_dict['s2_path'] = os.path.join(
                                self.root, row['s2_path']
                            )
            return path_dict

        sample_files = self.metadata_df.apply(get_image_paths, axis=1)
        return sample_files

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            planet image and mask at that index with image of dimension Tx4x1024x1024
            and mask of dimension 1024x1024, optional Sentinel-1 and Sentinel-2 images
            are included under the keys "s1_image" and "s2_image" with dimensions 13x1024x1024
            and 1x1024x1024 respectively
        """
        sample_paths = self.sample_file_paths[index]

        planet_img = torch.stack(
            [self._load_planet_images(path) for path in sample_paths['planet_paths']]
        )
        mask = self._load_mask(sample_paths['label_path'])

        sample = {'image': planet_img, 'mask': mask}

        if 's1_path' in sample_paths:
            sample['s1_image'] = self._load_sentinel(sample_paths['s1_path'])
        if 's2_path' in sample_paths:
            sample['s2_image'] = self._load_sentinel(sample_paths['s2_path'])

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def _load_planet_images(self, path: str) -> Tensor:
        """Load a Planet image from a given path.

        Args:
            path: path to the image

        Returns:
            tensor of the image
        """
        with rasterio.open(path) as src:
            img = src.read()
            # https://github.com/aysim/dynnet/blob/1e7d90294b54f52744ae2b35db10b4d0a48d093d/data/utae_dynamicen.py#L105
            # order of bands is BGRN, so rearrange them to RGBN
            img = img[[2, 1, 0, 3], :, :]
            tensor = torch.from_numpy(img).float()
        return tensor

    def _load_sentinel(self, path: str) -> Tensor:
        """Load a Sentinel image from a given path.

        Args:
            path: path to the image

        Returns:
            tensor of the image
        """
        with rasterio.open(path) as src:
            img = src.read()
            tensor = torch.from_numpy(img).float()
        return tensor

    def _load_mask(self, path: str) -> Tensor:
        """Load a mask from a given path.

        Args:
            path: path to the mask

        Returns:
            tensor of the mask
        """
        import pdb

        pdb.set_trace()
        with rasterio.open(path) as src:
            # mask has separate channel per class
            label = src.read()
            mask = np.zeros((label.shape[1], label.shape[2]), dtype=np.int32)
            # https://github.com/aysim/dynnet/blob/1e7d90294b54f52744ae2b35db10b4d0a48d093d/data/utae_dynamicen.py#L119
            # create single channel mask with class labels and map to 0
            for i in range(7):
                if i == 6:
                    mask[label[i, :, :] == 255] = -1
                else:
                    mask[label[i, :, :] == 255] = i
            tensor = torch.from_numpy(mask).long()
        return tensor

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            length of dataset
        """
        return len(self.sample_file_paths)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # check that directories are there
        exists = []
        for dir, fileinfo in self.filename_and_md5.items():
            exists.append(os.path.exists(os.path.join(self.root, dir)))

        if all(exists):
            return

        # check whether tarballs are there
        exists = []
        for dirname, fileinfo in self.filename_and_md5.items():
            path = os.path.join(self.root, fileinfo['filename'])
            if os.path.exists(path):
                if self.checksum and not check_integrity(path, fileinfo['md5']):
                    raise RuntimeError(f'Dataset {dirname} found, but corrupted.')
                exists.append(True)
                extract_archive(path, self.root)
            else:
                exists.append(False)

        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        # download and extract the dataset
        self._download_and_extract()

    def _download_and_extract(self) -> None:
        """Download and extract the dataset."""
        for dirname, fileinfo in self.filename_and_md5.items():
            download_and_extract_archive(
                self.base_url.format(fileinfo['filename']),
                self.root,
                filename=fileinfo['filename'],
                md5=fileinfo['md5'] if self.checksum else None,
            )

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        pass
