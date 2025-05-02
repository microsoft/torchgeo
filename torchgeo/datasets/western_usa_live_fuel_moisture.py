# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Western USA Live Fuel Moisture Dataset."""

import glob
import json
import os
from collections.abc import Callable, Iterable
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, which


class WesternUSALiveFuelMoisture(NonGeoDataset):
    """Western USA Live Fuel Moisture Dataset.

    This tabular style dataset contains fuel moisture
    (mass of water in vegetation) and remotely sensed variables
    in the western United States. It contains 2615 datapoints and 138
    variables. For more details see the
    `dataset page <https://beta.source.coop/stanford/sar-moisture-conent/>`_.

    Dataset Format:

    * .geojson file for each datapoint

    Dataset Features:

    * 138 remote sensing derived variables, some with a time dependency
    * 2615 datapoints with regression target of predicting fuel moisture

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.rse.2020.111797

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.

    .. versionadded:: 0.5
    """

    url = 'https://radiantearth.blob.core.windows.net/mlhub/su-sar-moisture-content'

    label_name = 'percent(t)'

    all_variable_names = (
        # "date",
        'slope(t)',
        'elevation(t)',
        'canopy_height(t)',
        'forest_cover(t)',
        'silt(t)',
        'sand(t)',
        'clay(t)',
        'vv(t)',
        'vh(t)',
        'red(t)',
        'green(t)',
        'blue(t)',
        'swir(t)',
        'nir(t)',
        'ndvi(t)',
        'ndwi(t)',
        'nirv(t)',
        'vv_red(t)',
        'vv_green(t)',
        'vv_blue(t)',
        'vv_swir(t)',
        'vv_nir(t)',
        'vv_ndvi(t)',
        'vv_ndwi(t)',
        'vv_nirv(t)',
        'vh_red(t)',
        'vh_green(t)',
        'vh_blue(t)',
        'vh_swir(t)',
        'vh_nir(t)',
        'vh_ndvi(t)',
        'vh_ndwi(t)',
        'vh_nirv(t)',
        'vh_vv(t)',
        'slope(t-1)',
        'elevation(t-1)',
        'canopy_height(t-1)',
        'forest_cover(t-1)',
        'silt(t-1)',
        'sand(t-1)',
        'clay(t-1)',
        'vv(t-1)',
        'vh(t-1)',
        'red(t-1)',
        'green(t-1)',
        'blue(t-1)',
        'swir(t-1)',
        'nir(t-1)',
        'ndvi(t-1)',
        'ndwi(t-1)',
        'nirv(t-1)',
        'vv_red(t-1)',
        'vv_green(t-1)',
        'vv_blue(t-1)',
        'vv_swir(t-1)',
        'vv_nir(t-1)',
        'vv_ndvi(t-1)',
        'vv_ndwi(t-1)',
        'vv_nirv(t-1)',
        'vh_red(t-1)',
        'vh_green(t-1)',
        'vh_blue(t-1)',
        'vh_swir(t-1)',
        'vh_nir(t-1)',
        'vh_ndvi(t-1)',
        'vh_ndwi(t-1)',
        'vh_nirv(t-1)',
        'vh_vv(t-1)',
        'slope(t-2)',
        'elevation(t-2)',
        'canopy_height(t-2)',
        'forest_cover(t-2)',
        'silt(t-2)',
        'sand(t-2)',
        'clay(t-2)',
        'vv(t-2)',
        'vh(t-2)',
        'red(t-2)',
        'green(t-2)',
        'blue(t-2)',
        'swir(t-2)',
        'nir(t-2)',
        'ndvi(t-2)',
        'ndwi(t-2)',
        'nirv(t-2)',
        'vv_red(t-2)',
        'vv_green(t-2)',
        'vv_blue(t-2)',
        'vv_swir(t-2)',
        'vv_nir(t-2)',
        'vv_ndvi(t-2)',
        'vv_ndwi(t-2)',
        'vv_nirv(t-2)',
        'vh_red(t-2)',
        'vh_green(t-2)',
        'vh_blue(t-2)',
        'vh_swir(t-2)',
        'vh_nir(t-2)',
        'vh_ndvi(t-2)',
        'vh_ndwi(t-2)',
        'vh_nirv(t-2)',
        'vh_vv(t-2)',
        'slope(t-3)',
        'elevation(t-3)',
        'canopy_height(t-3)',
        'forest_cover(t-3)',
        'silt(t-3)',
        'sand(t-3)',
        'clay(t-3)',
        'vv(t-3)',
        'vh(t-3)',
        'red(t-3)',
        'green(t-3)',
        'blue(t-3)',
        'swir(t-3)',
        'nir(t-3)',
        'ndvi(t-3)',
        'ndwi(t-3)',
        'nirv(t-3)',
        'vv_red(t-3)',
        'vv_green(t-3)',
        'vv_blue(t-3)',
        'vv_swir(t-3)',
        'vv_nir(t-3)',
        'vv_ndvi(t-3)',
        'vv_ndwi(t-3)',
        'vv_nirv(t-3)',
        'vh_red(t-3)',
        'vh_green(t-3)',
        'vh_blue(t-3)',
        'vh_swir(t-3)',
        'vh_nir(t-3)',
        'vh_ndvi(t-3)',
        'vh_ndwi(t-3)',
        'vh_nirv(t-3)',
        'vh_vv(t-3)',
        'lat',
        'lon',
    )

    def __init__(
        self,
        root: Path = 'data',
        input_features: Iterable[str] = all_variable_names,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new Western USA Live Fuel Moisture Dataset.

        Args:
            root: root directory where dataset can be found
            input_features: which input features to include
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            AssertionError: if ``input_features`` contains invalid variable names
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert set(input_features) <= set(self.all_variable_names)

        self.root = root
        self.input_features = input_features
        self.transforms = transforms
        self.download = download

        self._verify()

        self.dataframe = self._load_data()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            input features and target at that index
        """
        data = self.dataframe.iloc[index, :]

        sample = {
            'input': torch.tensor(
                data.drop([self.label_name]).values, dtype=torch.float32
            ),
            'label': torch.tensor(data[self.label_name], dtype=torch.float32),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_data(self) -> pd.DataFrame:
        """Load data from individual files into pandas dataframe.

        Returns:
            the features and label
        """
        data_rows = []
        for path in sorted(self.files):
            with open(path) as f:
                content = json.load(f)
                data_dict = content['properties']
                data_dict['lon'] = content['geometry']['coordinates'][0]
                data_dict['lat'] = content['geometry']['coordinates'][1]
                data_rows.append(data_dict)

        df = pd.DataFrame(data_rows)
        df = df[[*self.input_features, self.label_name]]
        return df

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        file_glob = os.path.join(self.root, '**', 'feature_*.geojson')
        self.files = glob.glob(file_glob, recursive=True)
        if self.files:
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self.files = glob.glob(file_glob, recursive=True)

    def _download(self) -> None:
        """Download the dataset and extract it."""
        os.makedirs(self.root, exist_ok=True)
        azcopy = which('azcopy')
        azcopy('sync', self.url, self.root, '--recursive=true')

    def plot(
        self,
        sample: dict[str, Any],
        variables_to_plot: list[str] = ['vv', 'vh', 'ndvi', 'ndwi', 'nirv'],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a time series visualization of the LFMC sample.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            variables_to_plot: a list of valid variable to be drawn in the plot
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for the Figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.8
        """
        input_data = sample['input'].numpy()

        # Time points to display on x-axis
        time_labels = ['t', 't-1', 't-2', 't-3']

        fig, axs = plt.subplots(
            len(variables_to_plot),
            1,
            figsize=(6, 1.5 * len(variables_to_plot)),
            sharex=True,
        )

        # Handle single subplot case
        if len(variables_to_plot) == 1:
            axs = [axs]

        for i, var_base_name in enumerate(variables_to_plot):
            values = []

            # Extract data for each time point (t, t-1, t-2, t-3)
            for t_label in time_labels:
                full_var_name = f'{var_base_name}({t_label})'
                var_position = self.all_variable_names.index(full_var_name)
                values.append(input_data[var_position])

            axs[i].plot(range(len(time_labels)), values, 'o-')
            axs[i].grid(True, alpha=0.3)

            if show_titles:
                axs[i].set_title(f'{var_base_name.upper()}')

        axs[-1].set_xticks(range(len(time_labels)))
        axs[-1].set_xticklabels(time_labels)

        # add coordinate and label information below the plot
        lon = input_data[-2]
        lat = input_data[-1]
        lfmc_value = sample['label'].item()

        axs[-1].text(
            x=0.5,
            y=-0.6,
            s=f'Live Fuel Moisture Content\nat {lon:.4f}, {lat:.4f}: {lfmc_value:.2f}%',
            ha='center',
            transform=axs[-1].transAxes,
        )

        if suptitle is not None:
            fig.suptitle(t=suptitle, y=1.5, fontsize=12, transform=axs[0].transAxes)

        fig.tight_layout()

        return fig
