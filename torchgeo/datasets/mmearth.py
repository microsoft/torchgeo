# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MM-Earth Dataset."""

import json
import os
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, lazy_import


class MMEarth(NonGeoDataset):
    """MM-Earth dataset.

    There are three different versions of the dataset, that vary in image size
    and the number of tiles:

    * MMEarth: 128x128 px, 1.2 M tiles, 579 GB
    * MMEarth64: 64x64 px, 1.2 M tiles, 162 GB
    * MMEarth100k: 128x128 px, 100 K tiles, 48 GB

    The dataset consists of 12 modalities:

    * Aster: elevation and slope
    * Biome: 14 terrestrial ecosystem categories
    * ETH Canopy Height: Canopy height and standard deviation
    * Dynamic World: 9 landcover categories
    * Ecoregion: 846 ecoregion categories
    * era5: Climate reanalysis data for temperature mean, min, and max of [year, month, previous month]
        and precipitation total of [year, month, previous month] (counted as separate modalities)
    * ESA World Cover: 11 landcover categories
    * Sentinel-1: VV, VH, HV, HH for ascending/descending orbit
    * Sentinel-2: multi-spectral B1-B12 for L1C/L2A products
    * Geolocation: cyclic encoding of latitude and longitude
    * Date: cyclic encoding of month

    Additionally, there are three masks available as modalities:

    * Sentinel-2 Cloudmask: Sentinel-2 cloud mask
    * Sentinel-2 Cloud probability: Sentinel-2 cloud probability
    * Sentinel-2 SCL: Sentinel-2 scene classification

    that are synchronized across tiles.

    Dataset format:

    * Dataset in single HDF5 file
    * Json files for band statistics, splits, and tile information

    For additional information, as well as bash scripts to
    download the data, please refer to the
    `official repository <https://github.com/vishalned/MMEarth-data?tab=readme-ov-file#data-download>`_.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2405.02771

    .. note::

       This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset

    .. versionadded:: 0.6
    """

    ds_versions = ['MMEarth', 'MMEarth64', 'MMEarth100k']

    filenames = {
        'MMEarth': 'data_1M_v001',
        'MMEarth64': 'data_1M_v001_64',
        'MMEarth100k': 'data_100k_v001',
    }
    splits = ['train', 'val', 'test']

    all_modalities = (
        'aster',
        'biome',
        'canopy_height_eth',
        'dynamic_world',
        'eco_region',
        'era5',
        'esa_worldcover',
        'sentinel1',
        'sentinel2',
        'sentinel2_cloudmask',
        'sentinel2_cloudprod',
        'sentinel2_scl',
    )

    # See https://github.com/vishalned/MMEarth-train/blob/8d6114e8e3ccb5ca5d98858e742dac24350b64fd/MODALITIES.py#L108C1-L160C2
    all_modality_bands = {
        'sentinel2': [
            'B1',
            'B2',
            'B3',
            'B4',
            'B5',
            'B6',
            'B7',
            'B8A',
            'B8',
            'B9',
            'B10',
            'B11',
            'B12',
        ],
        'sentinel2_cloudmask': ['QA60'],
        'sentinel2_cloudprod': ['MSK_CLDPRB'],
        'sentinel2_scl': ['SCL'],
        'sentinel1': [
            'asc_VV',
            'asc_VH',
            'asc_HH',
            'asc_HV',
            'desc_VV',
            'desc_VH',
            'desc_HH',
            'desc_HV',
        ],
        'aster': ['elevation', 'slope'],
        'era5': [
            'prev_month_avg_temp',
            'prev_month_min_temp',
            'prev_month_max_temp',
            'prev_month_total_precip',
            'curr_month_avg_temp',
            'curr_month_min_temp',
            'curr_month_max_temp',
            'curr_month_total_precip',
            'year_avg_temp',
            'year_min_temp',
            'year_max_temp',
            'year_total_precip',
        ],
        'dynamic_world': ['landcover'],
        'canopy_height_eth': ['height', 'std'],
        'lat': ['sin', 'cos'],
        'lon': ['sin', 'cos'],
        'biome': ['biome'],
        'eco_region': ['eco_region'],
        'month': ['sin_month', 'cos_month'],
        'esa_worldcover': ['map'],
    }

    # See https://github.com/vishalned/MMEarth-train/blob/8d6114e8e3ccb5ca5d98858e742dac24350b64fd/MODALITIES.py#L36
    no_data_vals = {
        'sentinel2': 0,
        'sentinel2_cloudmask': 65535,
        'sentinel2_cloudprod': 65535,
        'sentinel2_scl': 255,
        'sentinel1': float('-inf'),
        'aster': float('-inf'),
        'canopy_height_eth': 255,
        'dynamic_world': 0,
        'esa_worldcover': 255,
        'lat': float('-inf'),
        'lon': float('-inf'),
        'month': float('-inf'),
        'era5': float('inf'),
        'biome': 255,
        'eco_region': 65535,
    }

    norm_modes = ['z-score', 'min-max']

    def __init__(
        self,
        root: Path = 'data',
        ds_version: str = 'MMEarth',
        modalities: Sequence[str] = all_modalities,
        modality_bands: dict[str, list[str]] | None = None,
        split: str = 'train',
        normalization_mode: str = 'z-score',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    ) -> None:
        """Initialize the MM-Earth dataset.

        Args:
            root: root directory where dataset can be found
            ds_version: one of "MMEarth", "MMEarth64", or "MMEarth100k"
            modalities: list of modalities to load
            modality_bands: dictionary of modality bands, see `all_modality_bands`
            split: one of "train", "val", or "test"
            normalization_mode: one of "z-score" or "min-max"
            transforms: a function/transform that takes input sample dictionary
                and returns a transformed version

        """
        lazy_import('h5py')

        assert (
            normalization_mode in self.norm_modes
        ), f'Invalid normalization mode: {normalization_mode}, please choose from {self.norm_modes}'
        assert (
            ds_version in self.ds_versions
        ), f'Invalid dataset version: {ds_version}, please choose from {self.ds_versions}'
        assert (
            split in self.splits
        ), f'Invalid split: {split}, please choose from {self.splits}'

        self._validate_modalities(modalities)
        if modality_bands is None:
            modality_bands = {
                modality: self.all_modality_bands[modality] for modality in modalities
            }
        self._validate_modality_bands(modality_bands)

        self.modalities = modalities
        self.modality_bands = modality_bands

        self.root = root
        self.ds_version = ds_version
        self.normalization_mode = normalization_mode
        self.split = split
        self.transforms = transforms

        self.dataset_filename = f'{self.filenames[ds_version]}.h5'
        self.band_stats_filename = f'{self.filenames[ds_version]}_band_stats.json'
        self.splits_filename = f'{self.filenames[ds_version]}_splits.json'
        self.tile_info_filename = f'{self.filenames[ds_version]}_tile_info.json'

        self._verify()

        self.indices = self._load_indices()
        self.band_stats = self._load_normalization_stats()
        self.tile_info = self._load_tile_info()

    def _verify(self) -> None:
        """Verify the dataset.

        Raises:
            AssertionError: if dataset files are not found
        """
        data_dir = os.path.join(self.root, self.filenames[self.ds_version])

        exists = [
            os.path.exists(os.path.join(data_dir, f))
            for f in [
                self.dataset_filename,
                self.band_stats_filename,
                self.splits_filename,
                self.tile_info_filename,
            ]
        ]
        if not all(exists):
            raise DatasetNotFoundError(self)

    def _load_indices(self) -> list[int]:
        """Load the indices for the dataset split.

        Returns:
            list of indices
        """
        with open(
            os.path.join(
                self.root, self.filenames[self.ds_version], self.splits_filename
            )
        ) as f:
            split_indices: dict[str, list[int]] = json.load(f)

        return split_indices[self.split]

    def _load_normalization_stats(self) -> dict[str, dict[str, float]]:
        """Load normalization statistics for each band.

        Returns:
            dictionary containing the normalization statistics
        """
        with open(
            os.path.join(
                self.root, self.filenames[self.ds_version], self.band_stats_filename
            )
        ) as f:
            band_stats = json.load(f)

        return cast(dict[str, dict[str, float]], band_stats)

    def _load_tile_info(self) -> dict[str, dict[str, str]]:
        """Load tile information."""
        with open(
            os.path.join(
                self.root, self.filenames[self.ds_version], self.tile_info_filename
            )
        ) as f:
            tile_info = json.load(f)

        return cast(dict[str, dict[str, str]], tile_info)

    def _validate_modalities(self, modalities: Sequence[str]) -> None:
        """Validate list of modalities.

        Args:
            modalities: user-provided sequence of modalities to load

        Raises:
            AssertionError: if ``modalities`` is not a sequence
            ValueError: if an invalid modality name is provided
        """
        # validate modalities
        assert isinstance(modalities, Sequence), "'modalities' must be a sequence"
        for modality in modalities:
            if modality not in self.all_modalities:
                raise ValueError(f"'{modality}' is an invalid modality name.")

    def _validate_modality_bands(self, modality_bands: dict[str, list[str]]) -> None:
        """Validate modality bands.

        Args:
            modality_bands: user-provided dictionary of modality bands

        Raises:
            AssertionError: if ``modality_bands`` is not a dictionary
            ValueError: if an invalid modality name is provided
            ValueError: if modality bands are invalid
        """
        assert isinstance(modality_bands, dict), "'modality_bands' must be a dictionary"
        # validate modality bands
        for key, vals in modality_bands.items():
            if key not in self.all_modalities:
                raise ValueError(f"'{key}' is an invalid modality name.")
            for val in vals:
                if val not in self.all_modality_bands[key]:
                    raise ValueError(
                        f"'{val}' is an invalid band name for modality '{key}'."
                    )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a sample from the dataset.

        In addition to the modalities, the sample contains the following raw metadata:

        * lat: latitude
        * lon: longitude
        * date: date
        * crs: coordinate reference system
        * tile_id: tile identifier

        Args:
            index: index to return

        Returns:
            dictionary containing the modalities and metadata
            of the sample
        """
        h5py = lazy_import('h5py')

        sample: dict[str, Any] = {}
        ds_index = self.indices[index]

        with h5py.File(
            os.path.join(
                self.root, self.filenames[self.ds_version], self.dataset_filename
            ),
            'r',
        ) as f:
            name = f['metadata'][ds_index][0].decode('utf-8')
            tile_info = self.tile_info[name]
            l2a = tile_info['S2_type'] == 'l2a'
            for modality in self.modalities:
                data = f[modality][ds_index][:]
                sample[modality] = self._process_modality(data, modality, l2a)

            # add additional metadata to the sampl3
            sample['lat'] = tile_info['lat']
            sample['lon'] = tile_info['lon']
            sample['date'] = tile_info['S2_DATE']
            sample['crs'] = tile_info['CRS']
            sample['tile_id'] = name

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _process_modality(
        self, data: 'np.typing.NDArray', modality: str, l2a: bool
    ) -> Tensor:
        """Process a single modality.

        Args:
            data: data to process
            modality: modality name
            l2a: whether the data is from Sentinel-2 L2A

        Returns:
            processed data
        """
        # band selection for modality
        indices = [
            self.all_modality_bands[modality].index(band)
            for band in self.modality_bands[modality]
        ]
        data = data[indices, ...]

        # See https://github.com/vishalned/MMEarth-train/blob/8d6114e8e3ccb5ca5d98858e742dac24350b64fd/mmearth_dataset.py#L69
        if modality == 'dynamic_world':
            # first replacs 0 with nan then assign new labels to have 0-index classes
            data = np.where(data == self.no_data_vals[modality], np.nan, data)
            old_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]
            new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, np.nan]
            for old, new in zip(old_values, new_values):
                data = np.where(data == old, new, data)

            # need to replace nan with a no-data value and get long tensor
            # maybe also 255 like esa_worldcover
            tensor = torch.from_numpy(data)

        elif modality == 'esa_worldcover':
            old_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 255]
            new_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 255]
            for old, new in zip(old_values, new_values):
                data = np.where(data == old, new, data)

            # currently no-data value is still 255
            tensor = torch.from_numpy(data).long()

        elif modality in [
            'aster',
            'canopy_height_eth',
            'sentinel1',
            'sentinel2',
            'era5',
            'lat',
            'lon',
            'month',
        ]:
            data = data.astype(np.float32)
            # See https://github.com/vishalned/MMEarth-train/blob/8d6114e8e3ccb5ca5d98858e742dac24350b64fd/mmearth_dataset.py#L88
            # the modality is called sentinel2 but has different bands stats for l1c and l2a
            if modality == 'sentinel2':
                modality_ = 'sentinel2_l2a' if l2a else 'sentinel2_l1c'
            else:
                modality_ = modality
            data = self._normalize_modality(data, modality_)
            data = np.where(data == self.no_data_vals[modality], np.nan, data)
            tensor = torch.from_numpy(data).float()
        elif modality in ['biome', 'eco_region']:
            data = data.astype(np.int32)
            # no data value also 255 for biome and 65535 for eco_region
            tensor = torch.from_numpy(data).long()
        elif modality in [
            'sentinel2_cloudmask',
            'sentinel2_cloudprod',
            'sentinel2_scl',
        ]:
            tensor = torch.from_numpy(data.astype(np.int32)).long()

        # TODO: tensor might still contain nans, how to handle this?
        return tensor

    def _normalize_modality(
        self, data: 'np.typing.NDArray', modality: str
    ) -> 'np.typing.NDArray':
        """Normalize a single modality.

        Args:
            data: data to normalize
            modality: modality name

        Returns:
            normalized data
        """
        # the modality is called sentinel2 but has different bands stats for l1c and l2a
        if 'sentinel2' in modality:
            indices = [
                self.all_modality_bands['sentinel2'].index(band)
                for band in self.modality_bands['sentinel2']
            ]
        else:
            indices = [
                self.all_modality_bands[modality].index(band)
                for band in self.modality_bands[modality]
            ]

        if self.normalization_mode == 'z-score':
            mean = np.array(self.band_stats[modality]['mean'])[indices, ...]
            std = np.array(self.band_stats[modality]['std'])[indices, ...]
            if data.ndim == 3:
                data = (data - mean[:, None, None]) / std[:, None, None]
            else:
                data = (data - mean) / std
        elif self.normalization_mode == 'min-max':
            min_val = np.array(self.band_stats[modality]['min'])[indices, ...]
            max_val = np.array(self.band_stats[modality]['max'])[indices, ...]
            if data.ndim == 3:
                data = (data - min_val[:, None, None]) / (
                    max_val[:, None, None] - min_val[:, None, None]
                )
            else:
                data = (data - min_val) / (max_val - min_val)

        return data

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.indices)
