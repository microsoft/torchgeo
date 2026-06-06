# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Western USA Live Fuel Moisture Dataset."""

import glob
import json
import os
from collections.abc import Callable, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, which


class WesternUSALiveFuelMoisture(NonGeoDataset):
    """Western USA Live Fuel Moisture Dataset.

    This tabular style dataset contains fuel moisture
    (mass of water in vegetation) and remotely sensed variables
    in the western United States. It contains 2615 datapoints, each with 34
    variables observed at 4 time steps. For more details see the
    `dataset page <https://source.coop/stanford/sar-moisture-conent>`_.

    Dataset Format:

    * .geojson file for each datapoint

    Dataset Features:

    * 34 remote sensing derived variables, each observed at 4 time steps
      (``t``, ``t-1``, ``t-2``, ``t-3``)
    * 2615 datapoints with regression target of predicting fuel moisture

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.rse.2020.111797

    .. note::

       This dataset requires the following additional library to be installed:

       * `azcopy <https://github.com/Azure/azure-storage-azcopy>`_: to download the
         dataset from Source Cooperative.

    .. versionadded:: 0.5

    .. versionchanged:: 0.10
       ``input`` is now returned as a ``T x C`` time-series tensor (time steps by
       variables) instead of a flat vector, the point ``lat`` and ``lon`` are
       returned under their own keys, and ``input_features`` now selects base
       variable names, without the time-step suffix.
    """

    url = 'https://radiantearth.blob.core.windows.net/mlhub/su-sar-moisture-content'

    label_name = 'percent(t)'

    time_steps = ('t', 't-1', 't-2', 't-3')

    variable_names = (
        'slope',
        'elevation',
        'canopy_height',
        'forest_cover',
        'silt',
        'sand',
        'clay',
        'vv',
        'vh',
        'red',
        'green',
        'blue',
        'swir',
        'nir',
        'ndvi',
        'ndwi',
        'nirv',
        'vv_red',
        'vv_green',
        'vv_blue',
        'vv_swir',
        'vv_nir',
        'vv_ndvi',
        'vv_ndwi',
        'vv_nirv',
        'vh_red',
        'vh_green',
        'vh_blue',
        'vh_swir',
        'vh_nir',
        'vh_ndvi',
        'vh_ndwi',
        'vh_nirv',
        'vh_vv',
    )

    def __init__(
        self,
        root: Path = 'data',
        input_features: Iterable[str] = variable_names,
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new Western USA Live Fuel Moisture Dataset.

        Args:
            root: root directory where dataset can be found
            input_features: which base variables to include, without the time-step
                suffix (e.g. ``'ndvi'``); each one is returned across all *time_steps*
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            AssertionError: if ``input_features`` contains invalid variable names
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert set(input_features) <= set(self.variable_names)

        self.root = root
        self.input_features = tuple(input_features)
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

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            input time series, point coordinates, and target at that index
        """
        data = self.dataframe.iloc[index]

        sample = {
            'input': torch.tensor(
                [
                    [data[f'{name}({step})'] for name in self.input_features]
                    for step in self.time_steps
                ],
                dtype=torch.float32,
            ),
            'lon': torch.tensor(data['lon'], dtype=torch.float32),
            'lat': torch.tensor(data['lat'], dtype=torch.float32),
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

        columns = [
            f'{name}({step})'
            for step in self.time_steps
            for name in self.input_features
        ]
        columns += ['lon', 'lat', self.label_name]
        return pd.DataFrame(data_rows)[columns]

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
        sample: Sample,
        variables_to_plot: list[str] | None = None,
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
        if not variables_to_plot:
            variables_to_plot = list(self.input_features)
        else:
            variables_to_plot = [
                v for v in variables_to_plot if v in self.input_features
            ]
            if not variables_to_plot:
                raise ValueError(
                    'None of the requested variables are in input_features: '
                    f'{self.input_features}'
                )

        input_data = sample['input'].numpy()

        # Time points to display on x-axis
        time_labels = list(self.time_steps)

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
            # Extract the variable's value at each time step (t, t-1, t-2, t-3)
            position = self.input_features.index(var_base_name)
            values = input_data[:, position]

            axs[i].plot(range(len(time_labels)), values, 'o-')
            axs[i].grid(True, alpha=0.3)

            if show_titles:
                axs[i].set_title(f'{var_base_name.upper()}')

        axs[-1].set_xticks(range(len(time_labels)))
        axs[-1].set_xticklabels(time_labels)

        # add coordinate and label information below the plot
        lon = sample['lon'].item()
        lat = sample['lat'].item()
        lfmc_value = sample['label'].item()

        fig.supxlabel(
            f'Live Fuel Moisture Content\nat {lon:.4f}, {lat:.4f}: {lfmc_value:.2f}%'
        )

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()

        return fig
