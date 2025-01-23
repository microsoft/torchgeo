# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""AI4Artic Sea Ice Dataset."""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, ClassVar, cast

import numpy as np
import torch
from torch import Tensor
import xarray as xr

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, download_url, extract_archive, check_integrity


class AI4ArcticSeaIce(NonGeoDataset):
    """AI4Artic Sea Ice Dataset.

    The Sea Ice Challenge Dataset contains Sentinel-1 SAR imagery, passive microwave radiometer observations
    from AMSR2, and numerical weather prediction data from the ECMWF Reanalysis v5 (ERA5) dataset - all
    gridded to match the Sentinel-1 SAR scenes geometrically. As label data, the dataset contains ice charts
    manually produced by the ice analysts at the Greenland Ice Service and the Canadian Ice Service.

    Dataset features:

    * Dual-polarization SAR (HH, HV) imagery for each patch.
    * Sea Ice Concentration (SIC): the percentage ratio of sea ice to open water for an area,
        discretized into 11 10% bins ranging from 0% to 100%.
    * Stage Of Development (SOD): type of sea ice, as proxy for ice thickness and
        ease of traversing with 6 classes
    * Floe size (FLOE): Classifying or segmenting distinct ice floes based on size, shape,
          or other geometric properties.

    Dataset format:

    * each sample scene is stored in a separate .nc file
    * pixel dimension of varying sizes up to ~5000pxx5000px
    * 80m resolution

    Geographical variables:

    * distance-to-land layer (distance_map)

    SAR variables:

    * Sentinel-1 backscatter intensity (dB) in HH polarization (nersc_sar_primary)
    * Sentinel-1 backscatter intensity (dB) in HV polarization (nersc_sar_secondary)
    * Sentinel-1 incidence angle (sar_incidenceangle)

    Weather variables:

    * eastward wind component at 10m (u10m_rotated)
    * northward wind component at 10m (v10m_rotated)
    * ERA5 2m air temperature (t2m)
    * ERA5 skin temperature (skt)
    * ERA5 total column water vapor (tcwv)
    * ERA5 total column liquid water (tclw)

    Advanced Microwave Scanning Radiometer 2 (AMSR2) variables:

    * 6.9 GHz Brightness Temperature (btemp_6_9h, btemp_6_9v)
    * 7.3 GHz Brightness Temperature (btemp_7_3h, btemp_7_3v)
    * 10.7 GHz Brightness Temperature (btemp_10_7h, btemp_10_7v)
    * 18.7 GHz Brightness Temperature (btemp_18_7h, btemp_18_7v)
    * 23.8 GHz Brightness Temperature (btemp_23_8h, btemp_23_8v)
    * 36.5 GHz Brightness Temperature (btemp_36_5h, btemp_36_5v)
    * 89.0 GHz Brightness Temperature (btemp_89_0h, btemp_89_0v)

    Sea Ice Concentration (SIC) classes:

    * 0: 0%
    * 1: 0-10%
    * 2: 10-20%
    * 3: 20-30%
    * 4: 30-40%
    * 5: 40-50%
    * 6: 50-60%
    * 7: 60-70%
    * 8: 70-80%
    * 9: 80-90%
    * 10: 90-100%

    Stage of Development (SOD) classes:

    * 0: Open-water
    * 1: New ice
    * 2: Young ice
    * 3: Thin First-year ice
    * 4: Thick First-year ice
    * 5: Old ice (older than 1 year)

    Floe size (FLOE) classes:

    * 0: Open-water
    * 1: Cake ice
    * 2: Small floe
    * 3: Medium floe
    * 4: Big floe
    * 5: Vast floe
    * 6: Bergs (variants of icebergs and glacier ice)

    files:
    Danish Meteorological Institute (DMI) and the Canadian Ice Service (CIS)
    dmi_prep: data by DMI
    cis_prep: data by CIS
    dmi_prep_referece: contains SIC, SOD, FLOE
    cis_prep_reference: contains SIC, SOD, FLOE

    Dataset format:

    * Dataset in separate .nc files

    If you use this dataset in your research, please cite the following paper:

    * https://data.dtu.dk/articles/dataset/Ready-To-Train_AI4Arctic_Sea_Ice_Challenge_Dataset/21316608

    .. note::

        This dataset requires the following additional libraries to be installed:

        * `xarray <https://docs.xarray.dev/en/stable/getting-started-guide/installing.html>`_
        * `netcdf4 <https://unidata.github.io/netcdf4-python/>`_

    .. versionadded:: 0.7

    # Variables in the ASID3 challenge ready-to-train dataset
    """

    url = 'https://huggingface.co/datasets/torchgeo/ai4artic-sea-ice-challenge/resolve/main/{}'

    files = [
        {'name': 'metadata.csv', 'md5': '4b610118c2d182325ec7599434b37deb'},
        {'name': 'train.tar.gzaa', 'md5': '847ea12d0a5100f0a00af4bb110404b4'},
        {'name': 'train.tar.gzab', 'md5': '3f4770c586487dc681d1d216c7003f2c'},
        {'name': 'test.tar.gz', 'md5': 'bca98ec6734783aa6f005382549a0d21'},
    ]

    splits = ('train', 'test')

    # https://github.com/astokholm/AI4ArcticSeaIceChallenge/blob/4d5e3bc85e681f6c56821d96f2ebfcf4ed58b495/utils.py#L68
    SIC_GROUPS = {
        0: 0,
        1: 10,
        2: 20,
        3: 30,
        4: 40,
        5: 50,
        6: 60,
        7: 70,
        8: 80,
        9: 90,
        10: 100,
    }

    SOD_GROUPS = {
        0: 'Open water',
        1: 'New Ice',
        2: 'Young ice',
        3: 'Thin FYI',
        4: 'Thick FYI',
        5: 'Old ice',
    }

    FLOE_GROUPS = {
        0: 'Open water',
        1: 'Cake Ice',
        2: 'Small floe',
        3: 'Medium floe',
        4: 'Big floe',
        5: 'Vast floe',
        6: 'Bergs',
    }

    valid_sar_vars = ('nersc_sar_primary', 'nersc_sar_secondary', 'sar_incidenceangle')
    valid_geo_vars = ('distance_map',)
    valid_amsr2_vars = (
        'btemp_6_9h',
        'btemp_6_9v',
        'btemp_7_3h',
        'btemp_7_3v',
        'btemp_10_7h',
        'btemp_10_7v',
        'btemp_18_7h',
        'btemp_18_7v',
        'btemp_23_8h',
        'btemp_23_8v',
        'btemp_36_5h',
        'btemp_36_5v',
        'btemp_89_0h',
        'btemp_89_0v',
    )
    valid_weather_vars = ('u10m_rotated', 'v10m_rotated', 't2m', 'skt', 'tcwv', 'tclw')

    valid_target_vars = ('SOD', 'SIC', 'FLOE')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        target_var: str = 'SOD',
        geo_var: str | None = None,
        amsr2_vars: Sequence[str] | None = None,
        weather_vars: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize the AI4Artic Sea Ice dataset.

        Args:
            root: root directory where the dataset can be found
            split: The split of the dataset. Either 'train' or 'test'.
            target_var: Target variable to be the label mask
            geo_var: Geographical variables to include in the dataset, only option is 'distance_map'
            amsr2_vars: AMSR2 channels to include in the dataset
            weather_vars: Environmental variables to include in the dataset
            transforms: a function/transform that takes input sample dictionary
                and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: If  *split* is not one of 'train' or 'test', or if selected variables are not valid.
            DatasetNotFoundError: If the dataset is not found and *download* is False.
            DependencyNotFoundError: If xarray is not installed.
        """
        assert target_var in self.valid_target_vars, (
            f'Invalid target variable selected. Must be one of {self.valid_target_vars}'
        )
        if geo_var is not None:
            assert geo_var == 'distance_map', (
                f"Invalid geographical variable selected. Only 'distance_map' is supported."
            )

        if amsr2_vars is not None:
            assert all(var in self.valid_amsr2_vars for var in amsr2_vars), (
                f'Invalid AMSR2 variables selected. Must be a subset of {self.valid_amsr2_vars}'
            )

        if weather_vars is not None:
            assert all(var in self.valid_weather_vars for var in weather_vars), (
                f'Invalid weather variables selected. Must be a subset of {self.valid_weather_vars}'
            )

        assert split in self.splits, (
            f"Split '{split}' not supported, must be one of {self.splits}"
        )

        self.target_var = target_var
        self.geo_var = geo_var
        self.amsr2_vars = amsr2_vars
        self.weather_vars = weather_vars

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        # metadata df
        self.metadata_df = pd.read_csv(os.path.join(self.root, 'metadata.csv'))
        self.metadata_df = self.metadata_df[
            self.metadata_df['split'] == self.split
        ].reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get the sample at the given index.

        Args:
            idx: index of the sample to return

        Returns:
            A dictionary containing the sample data, split into the following keys by data type:
            * 'image': SAR data stacked hh, hv in that order
            * 'geo': Geographical data
            * 'amsr2': AMSR2 data
            * 'weather': Weather data
            * 'mask': Chosen target data
        """
        df_row = self.metadata_df.iloc[idx]

        # load data
        sample = self._load_data(os.path.join(self.root, df_row['input_path']))

        # load target
        sample['mask'] = self._load_label(os.path.join(self.root, df_row['input_path']))
        if self.transforms is not None:
            sample = self.transforms(sample)

        # crop bottom right corner of the image
        # sample["image"] = sample["image"][:, -1024:, -1024:]
        # sample["mask"] = sample["mask"][-1024:, -1024:]
        # crop bottom left corner
        # sample["image"] = sample["image"][:, -1024:, :1024]
        # sample["mask"] = sample["mask"][-1024:, :1024]

        return sample

    def _load_data(self, path: str) -> dict[str, Tensor]:
        """Load the data from the given path.

        Args:
            input_path: path to the data file

        Returns:
            A dictionary containing the data, split into the following keys followed by var name if specified:
            * 'image': SAR data stacked hh, hv in that order
            * 'geo': Geographical data
            * 'amsr2': AMSR2 data
            * 'weather': Weather data
        """
        sample: dict[str, Tensor] = {}

        input_data = xr.open_dataset(path)

        # load s1 vars
        hh = torch.from_numpy(input_data['nersc_sar_primary'].values)
        hv = torch.from_numpy(input_data['nersc_sar_secondary'].values)

        # NaN values in SAR data have value 2
        sample['image'] = torch.stack([hh, hv], dim=0)

        if self.geo_var is not None:
            sample['geo'] = torch.from_numpy(input_data[self.geo_var].values)

        if self.amsr2_vars is not None:
            data = np.stack([input_data[var].values for var in self.amsr2_vars])
            sample['amsr2'] = torch.from_numpy(data)

        if self.weather_vars is not None:
            data = np.stack([input_data[var].values for var in self.weather_vars])
            sample['weather'] = torch.from_numpy(data)

        input_data.close()

        return sample

    def _load_label(self, path: str) -> Tensor:
        """Load the label from the given path.

        Args:
            path: path to the label file

        Returns:
            A tensor containing the label data
        """
        # in test directory label is under a separate file
        if self.split == 'test':
            # append 'reference' to the input path to get the reference file
            path = path.replace('.nc', '_reference.nc')

        target_data = xr.open_dataset(path)
        # NaN values in target data have value 255
        tensor = torch.from_numpy(target_data[self.target_var].values).long()
        target_data.close()

        return tensor

    def _verify(self) -> None:
        """Verify integrity of the dataset."""
        # check if metadata file exists
        exists = []
        if os.path.exists(os.path.join(self.root, 'metadata.csv')):
            df = pd.read_csv(os.path.join(self.root, 'metadata.csv'))
            for i, row in df.iterrows():
                exists.append(
                    os.path.exists(os.path.join(self.root, row['input_path']))
                )
        else:
            exists.append(False)

        if all(exists):
            return

        # check presence of tarball files
        exists = [
            os.path.exists(os.path.join(self.root, file['name'])) for file in self.files
        ]
        if all(exists):
            return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download_data()
        self._extract_data()

    def _download_data(self) -> None:
        """Download data."""
        for file in self.files:
            download_url(
                self.url.format(file['name']),
                self.root,
                md5=file['md5'] if self.checksum else None,
            )

    def _extract_data(self) -> None:
        """Extract the dataset."""
        # Concatenate the train tarballs together
        chunk_size = 2**15  # same as torchvision
        path = os.path.join(self.root, 'train.tar.gz')
        with open(path, 'wb') as f:
            for split in ['aa', 'ab']:
                with open(os.path.join(self.root, f'train.tar.gz{split}'), 'rb') as g:
                    while chunk := g.read(chunk_size):
                        f.write(chunk)
        extract_archive(path, self.root)

        # Extract test tarball
        extract_archive(os.path.join(self.root, 'test.tar.gz'), self.root)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`CaFFe.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        if 'prediction' in sample:
            ncols = 3
        else:
            ncols = 2

        class_mapping = getattr(self, f'{self.target_var}_GROUPS')
        # add 255 for NaN values
        class_mapping[255] = 'NaN'

        num_classes = len(class_mapping)

        fig, axs = plt.subplots(1, ncols, figsize=(15, 7))

        # Plot SAR image (HH channel) with proper normalization
        hh_image = sample['image'][0].numpy()
        vmin, vmax = np.nanpercentile(hh_image, (2, 98))  # robust normalization
        axs[0].imshow(hh_image, cmap='gray', vmin=vmin, vmax=vmax)
        axs[0].axis('off')
        if show_titles:
            axs[0].set_title('SAR HH Channel')

        # Create colormap with transparent color for NaN
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        # colors = np.vstack((colors, [1, 1, 1, 0]))  # add transparent for NaN
        cmap = plt.cm.colors.ListedColormap(colors)

        # Plot mask with proper handling of NaN values
        # import pdb
        # pdb.set_trace()
        mask = sample['mask'].numpy()
        # mask_ma = ma.masked_where(mask == 255, mask)  # mask NaN values
        axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=num_classes)
        if show_titles:
            axs[1].set_title(f'{self.target_var} Mask')
        axs[1].axis('off')

        if 'prediction' in sample:
            prediction = sample['prediction'].numpy()
            # pred_ma = ma.masked_where(prediction == 255, prediction)
            axs[2].imshow(prediction, cmap=cmap)
            if show_titles:
                axs[2].set_title('Prediction Mask')
            axs[2].axis('off')

        # create legend with class names
        # import pdb
        # pdb.set_trace()
        legend_elements = [
            Patch(facecolor=colors[i], label=list(class_mapping.values())[i])
            for i in range(num_classes)
        ]
        fig.legend(
            handles=legend_elements,
            loc='center right',
            bbox_to_anchor=(0.98, 0.5),
            title=self.target_var,
        )

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
