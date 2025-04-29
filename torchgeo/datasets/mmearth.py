# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MMEarth Dataset."""

import json
import os
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, lazy_import, percentile_normalization


class MMEarth(NonGeoDataset):
    """MMEarth dataset.

    There are three different versions of the dataset, that vary in image size
    and the number of tiles:

    * MMEarth: 128x128 px, 1.2M tiles, 579 GB
    * MMEarth64: 64x64 px, 1.2M tiles, 162 GB
    * MMEarth100k: 128x128 px, 100K tiles, 48 GB

    The dataset consists of 12 modalities:

    * Aster: elevation and slope
    * Biome: 14 terrestrial ecosystem categories
    * ETH Canopy Height: Canopy height and standard deviation
    * Dynamic World: 9 landcover categories
    * Ecoregion: 846 ecoregion categories
    * ERA5: Climate reanalysis data for temperature mean, min, and max of [year, month, previous month]
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
    * JSON files for band statistics, splits, and tile information

    For additional information, as well as bash scripts to
    download the data, please refer to the
    `official repository <https://github.com/vishalned/MMEarth-data?tab=readme-ov-file#data-download>`_.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2405.02771

    .. note::

       This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset

    .. versionadded:: 0.7
    """

    subsets = ('MMEarth', 'MMEarth64', 'MMEarth100k')

    filenames: ClassVar[dict[str, str]] = {
        'MMEarth': 'data_1M_v001',
        'MMEarth64': 'data_1M_v001_64',
        'MMEarth100k': 'data_100k_v001',
    }

    all_modalities = (
        'aster',
        'biome',
        'canopy_height_eth',
        'dynamic_world',
        'eco_region',
        'era5',
        'esa_worldcover',
        'sentinel1_asc',
        'sentinel1_desc',
        'sentinel2',
        'sentinel2_cloudmask',
        'sentinel2_cloudprod',
        'sentinel2_scl',
    )

    # See https://github.com/vishalned/MMEarth-train/blob/8d6114e8e3ccb5ca5d98858e742dac24350b64fd/MODALITIES.py#L108C1-L160C2
    all_modality_bands: ClassVar[dict[str, list[str]]] = {
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
        'sentinel1_asc': ['VV', 'VH', 'HH', 'HV'],
        'sentinel1_desc': ['VV', 'VH', 'HH', 'HV'],
        'aster': ['b1', 'slope'],  # elevation and slope
        'era5': [
            'prev_temperature_2m',  # previous month avg temp
            'prev_temperature_2m_min',  # previous month min temp
            'prev_temperature_2m_max',  # previous month max temp
            'prev_total_precipitation_sum',  # previous month total precip
            'curr_temperature_2m',  # current month avg temp
            'curr_temperature_2m_min',  # current month min temp
            'curr_temperature_2m_max',  # current month max temp
            'curr_total_precipitation_sum',  # current month total precip
            '0_temperature_2m_mean',  # year avg temp
            '1_temperature_2m_min_min',  # year min temp
            '2_temperature_2m_max_max',  # year max temp
            '3_total_precipitation_sum_sum',  # year total precip
        ],
        'dynamic_world': ['label'],
        'canopy_height_eth': ['height', 'std'],
        'lat': ['sin', 'cos'],
        'lon': ['sin', 'cos'],
        'biome': ['biome'],
        'eco_region': ['eco_region'],
        'month': ['sin_month', 'cos_month'],
        'esa_worldcover': ['Map'],
    }

    # See https://github.com/vishalned/MMEarth-train/blob/8d6114e8e3ccb5ca5d98858e742dac24350b64fd/MODALITIES.py#L36
    no_data_vals: ClassVar[dict[str, int | float]] = {
        'sentinel2': 0,
        'sentinel2_cloudmask': 65535,
        'sentinel2_cloudprod': 65535,
        'sentinel2_scl': 255,
        'sentinel1_asc': float('-inf'),
        'sentinel1_desc': float('-inf'),
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

    norm_modes = ('z-score', 'min-max')

    modality_category_name: ClassVar[dict[str, str]] = {
        'sentinel1_asc': 'image_',
        'sentinel1_desc': 'image_',
        'sentinel2': 'image_',
        'sentinel2_cloudmask': 'mask_',
        'sentinel2_cloudprod': 'mask_',
        'sentinel2_scl': 'mask_',
        'aster': 'image_',
        'era5': '',
        'canopy_height_eth': 'image_',
        'dynamic_world': 'mask_',
        'esa_worldcover': 'mask_',
    }

    def __init__(
        self,
        root: Path = 'data',
        subset: str = 'MMEarth',
        modalities: Sequence[str] = all_modalities,
        modality_bands: dict[str, list[str]] | None = None,
        normalization_mode: str = 'z-score',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    ) -> None:
        """Initialize the MMEarth dataset.

        Args:
            root: root directory where dataset can be found
            subset: one of "MMEarth", "MMEarth64", or "MMEarth100k"
            modalities: list of modalities to load
            modality_bands: dictionary of modality bands, see
            normalization_mode: one of "z-score" or "min-max"
            transforms: a function/transform that takes input sample dictionary
                and returns a transformed version

        Raises:
            AssertionError: if *normalization_mode* or *subset*
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        lazy_import('h5py')

        assert normalization_mode in self.norm_modes, (
            f'Invalid normalization mode: {normalization_mode}, please choose from {self.norm_modes}'
        )
        assert subset in self.subsets, (
            f'Invalid dataset version: {subset}, please choose from {self.subsets}'
        )

        self._validate_modalities(modalities)
        self.modalities = modalities
        if modality_bands is None:
            modality_bands = {
                modality: self.all_modality_bands[modality] for modality in modalities
            }
        self._validate_modality_bands(modality_bands)
        self.modality_bands = modality_bands

        self.root = root
        self.subset = subset
        self.normalization_mode = normalization_mode
        self.split = 'train'
        self.transforms = transforms

        self.dataset_filename = f'{self.filenames[subset]}.h5'
        self.band_stats_filename = f'{self.filenames[subset]}_band_stats.json'
        self.splits_filename = f'{self.filenames[subset]}_splits.json'
        self.tile_info_filename = f'{self.filenames[subset]}_tile_info.json'

        self._verify()

        self.indices = self._load_indices()
        self.band_stats = self._load_normalization_stats()
        self.tile_info = self._load_tile_info()

    def _verify(self) -> None:
        """Verify the dataset."""
        data_dir = os.path.join(self.root, self.filenames[self.subset])

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
            os.path.join(self.root, self.filenames[self.subset], self.splits_filename)
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
                self.root, self.filenames[self.subset], self.band_stats_filename
            )
        ) as f:
            band_stats = json.load(f)

        return cast(dict[str, dict[str, float]], band_stats)

    def _load_tile_info(self) -> dict[str, dict[str, str]]:
        """Load tile information.

        Returns:
            dictionary containing tile information
        """
        with open(
            os.path.join(
                self.root, self.filenames[self.subset], self.tile_info_filename
            )
        ) as f:
            tile_info = json.load(f)

        return cast(dict[str, dict[str, str]], tile_info)

    def _validate_modalities(self, modalities: Sequence[str]) -> None:
        """Validate list of modalities.

        Args:
            modalities: user-provided sequence of modalities to load

        Raises:
            AssertionError: if ``modalities`` is not a sequence or an
                invalid modality name is provided
        """
        # validate modalities
        assert isinstance(modalities, Sequence), "'modalities' must be a sequence"
        if not set(modalities) <= set(self.all_modalities):
            raise ValueError(
                f'{set(modalities) - set(self.all_modalities)} is an invalid modality.'
            )

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
            # check that the modality name is also specified in modalities
            if key not in self.modalities:
                raise ValueError(f"'{key}' is an invalid modality name.")
            for val in vals:
                if val not in self.all_modality_bands[key]:
                    raise ValueError(
                        f"'{val}' is an invalid band name for modality '{key}'."
                    )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a sample from the dataset.

        Normalization is applied to the data with chosen ``normalization_mode``.
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
        ds_index = self.indices[index]

        # expose sample retrieval to separate function to allow for different index sampling strategies
        # in subclasses
        sample = self._retrieve_sample(ds_index)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def get_sample_specific_band_names(
        self, tile_info: dict[str, Any]
    ) -> dict[str, list[str]]:
        """Retrieve the sample specific band names.

        Args:
            tile_info: tile information for a sample

        Returns:
            dictionary containing the specific band names for each modality
        """
        date_str = tile_info['S2_DATE']
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        curr_month_str = date_obj.strftime('%Y%m')
        # set to first day of month and subtract one day to get previous month
        prev_month_obj = date_obj.replace(day=1) - timedelta(days=1)
        prev_month_str = prev_month_obj.strftime('%Y%m')

        specific_modality_bands = {}
        for modality, bands in self.modality_bands.items():
            if modality == 'era5':
                # replace date with the 'prev' and 'curr' strings for generality
                bands = [band.replace(prev_month_str, 'prev') for band in bands]
                bands = [band.replace(curr_month_str, 'curr') for band in bands]
            specific_modality_bands[modality] = bands

        return specific_modality_bands

    def get_intersection_dict(self, tile_info: dict[str, Any]) -> dict[str, list[str]]:
        """Get intersection of requested and available bands.

        Args:
            tile_info: tile information for a sample

        Returns:
            Dictionary with intersected keys and lists.
        """
        sample_specific_band_names = self.get_sample_specific_band_names(tile_info)
        # used the chosen modality bands to get the intersection with available bands
        intersection_dict = {}
        for modality in self.all_modalities:
            if modality in sample_specific_band_names:
                intersected_list = [
                    band
                    for band in self.all_modality_bands[modality]
                    if band in sample_specific_band_names[modality]
                ]
                if intersected_list:
                    intersection_dict[modality] = intersected_list

        return intersection_dict

    def _retrieve_sample(self, ds_index: int) -> dict[str, Any]:
        """Retrieve a sample from the dataset.

        Args:
            ds_index: index inside the hdf5 dataset file

        Returns:
            dictionary containing the modalities and metadata
            of the sample
        """
        h5py = lazy_import('h5py')
        sample: dict[str, Any] = {}
        with h5py.File(
            os.path.join(self.root, self.filenames[self.subset], self.dataset_filename),
            'r',
        ) as f:
            name = f['metadata'][ds_index][0].decode('utf-8')
            tile_info: dict[str, Any] = self.tile_info[name]
            # need to find the intersection of requested and available bands
            intersection_dict = self.get_intersection_dict(tile_info)
            for modality, bands in intersection_dict.items():
                if 'sentinel1' in modality:
                    data = f['sentinel1'][ds_index][:]
                else:
                    data = f[modality][ds_index][:]

                tensor = self._preprocess_modality(data, modality, tile_info, bands)
                modality_name = self.modality_category_name.get(modality, '') + modality
                sample[modality_name] = tensor

            # add the sensor and bands actually available
            sample['avail_bands'] = intersection_dict

            # add additional metadata to the sample
            sample['lat'] = tile_info['lat']
            sample['lon'] = tile_info['lon']
            sample['date'] = tile_info['S2_DATE']
            sample['crs'] = tile_info['CRS']
            sample['tile_id'] = name

        return sample

    def _select_indices_for_modality(
        self, modality: str, bands: list[str]
    ) -> list[int]:
        """Select bands for a modality.

        Args:
            modality: modality name
            bands: bands aviailable for the modality

        Returns:
            list of band indices
        """
        # need to handle sentinel1 descending separately, because ascending
        # and descending are stored under the same modality
        if modality == 'sentinel1_desc':
            indices = [
                self.all_modality_bands['sentinel1_desc'].index(band) + 4
                for band in bands
            ]
        # the modality is called sentinel2 but has different bands stats for l1c and l2a
        # but common indices
        elif modality in ['sentinel2_l1c', 'sentinel2_l2a']:
            indices = [
                self.all_modality_bands['sentinel2'].index(band) for band in bands
            ]
        else:
            indices = [self.all_modality_bands[modality].index(band) for band in bands]
        return indices

    def _preprocess_modality(
        self,
        data: 'np.typing.NDArray[Any]',
        modality: str,
        tile_info: dict[str, Any],
        bands: list[str],
    ) -> Tensor:
        """Preprocess a single modality.

        Args:
            data: data to process
            modality: modality name
            tile_info: tile information
            bands: available bands for the modality

        Returns:
            processed data
        """
        # band selection for modality
        indices = self._select_indices_for_modality(modality, bands)
        data = data[indices, ...]

        # See https://github.com/vishalned/MMEarth-train/blob/8d6114e8e3ccb5ca5d98858e742dac24350b64fd/mmearth_dataset.py#L69
        if modality == 'dynamic_world':
            # first replace 0 with nan then assign new labels to have 0-index classes
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
            'sentinel1_asc',
            'sentinel1_desc',
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
                modality_ = (
                    'sentinel2_l2a'
                    if tile_info['S2_type'] == 'l2a'
                    else 'sentinel2_l1c'
                )
            else:
                modality_ = modality
            data = self._normalize_modality(data, modality_, bands)
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
        self, data: 'np.typing.NDArray[Any]', modality: str, bands: list[str]
    ) -> 'np.typing.NDArray[np.float64]':
        """Normalize a single modality.

        Args:
            data: data to normalize
            modality: modality name
            bands: available bands for the modality

        Returns:
            normalized data
        """
        indices = self._select_indices_for_modality(modality, bands)

        if 'sentinel1' in modality:
            modality = 'sentinel1'

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
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """
        return len(self.indices)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset as shown in fig. 2 from https://arxiv.org/pdf/2405.02771.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        color_map = {
            'esa_worldcover': {
                0: [0, 100, 0],  # Tree cover
                1: [255, 187, 34],  # Shrubland
                2: [255, 255, 76],  # Grassland
                3: [240, 150, 255],  # Cropland
                4: [250, 0, 0],  # Built-up
                5: [180, 180, 180],  # Bare/sparse vegetation
                6: [240, 240, 240],  # Snow and Ice
                7: [0, 100, 200],  # Permanent water bodies
                8: [0, 150, 160],  # Herbaceous wetland
                9: [0, 207, 117],  # Mangroves
                10: [250, 230, 160],  # Moss and lichen
                255: [0, 0, 0],  # No-data value
            },
            'dynamic_world': {
                0: [65, 155, 223],  # #419BDF - Water
                1: [57, 125, 73],  # #397D49 - Trees
                2: [136, 176, 83],  # #88B053 - Grass
                3: [122, 135, 198],  # #7A87C6 - Flooded vegetation
                4: [228, 150, 53],  # #E49635 - Crops
                5: [223, 195, 90],  # #DFC35A - Shrub & Scrub
                6: [196, 40, 27],  # #C4281B - Built Area
                7: [165, 155, 143],  # #A59B8F - Bare ground
                8: [179, 159, 225],  # #B39FE1 - Snow & Ice
            },
        }

        images = []
        titles = []

        keys_to_plot = [
            'image_sentinel2',
            'image_sentinel1_asc',
            'image_aster',
            'mask_esa_worldcover',
            'mask_dynamic_world',
            'image_canopy_height_eth',
        ]

        avail_bands_dict = dict(sample['avail_bands'])
        for key in keys_to_plot:
            val = sample[key]
            modalities_name = key.split('_', 1)[1]
            match modalities_name:
                case 'sentinel2':
                    norm_img = percentile_normalization(val[[3, 2, 1]].numpy())
                    images.append(rearrange(norm_img, 'c h w -> h w c'))

                    titles.append('Sentinel-2 RGB')
                case 'esa_worldcover':
                    tensor_np = val.squeeze().numpy()
                    rgb_image = np.zeros(
                        (tensor_np.shape[0], tensor_np.shape[1], 3), dtype=np.uint8
                    )
                    for value, color in color_map[modalities_name].items():
                        mask = tensor_np == value
                        rgb_image[mask] = color

                    images.append(rgb_image)
                    titles.append(modalities_name.replace('_', ' ').title())
                case 'dynamic_world':
                    tensor_np = val.squeeze().numpy()
                    rgb_image = np.zeros(
                        (tensor_np.shape[0], tensor_np.shape[1], 3), dtype=np.uint8
                    )
                    for value, color in color_map[modalities_name].items():
                        mask = tensor_np == value
                        rgb_image[mask] = color

                    images.append(rgb_image)
                    titles.append(modalities_name.replace('_', ' ').title())
                case _:
                    band_val = val[0].numpy()
                    norm_img = percentile_normalization(band_val)
                    images.append(norm_img)

                    modalities_name = key.split('_', 1)[1]
                    band_name = avail_bands_dict[modalities_name][0]
                    titles.append(
                        (modalities_name.replace('_', ' ').title()) + ' ' + band_name
                    )
        fig, ax = plt.subplots(1, 6, figsize=(12, 4), squeeze=False)
        axes = ax.flatten()

        for i, (image, title) in enumerate(zip(images, titles)):
            axes[i].imshow(image)
            axes[i].axis('off')

            if show_titles:
                title_words = title.split(' ')
                title_word_len = len(title_words)
                if title_word_len > 2:
                    title = (
                        str.join(' ', title_words[:2])
                        + '\n'
                        + str.join(' ', title_words[2:])
                    )
                axes[i].set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()

        return fig
