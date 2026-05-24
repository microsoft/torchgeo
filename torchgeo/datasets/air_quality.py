# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Air Quality dataset."""

import math
import pathlib
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample


class AirQuality(NonGeoDataset):
    """Air Quality dataset.

    The `Air Quality dataset <https://archive.ics.uci.edu/dataset/360/air+quality>`_
    from the UCI Machine Learning Repository is a multivariate time
    series dataset containing air quality measurements from an Italian
    city.

    Dataset Format:

    * .csv file containing date, time and air quality measurements

    Dataset Features:

    * hourly averaged sensor responses and reference analyzer ground truth over one year
      (2004-2005)
    * contains missing features, gap filled using linear interpolation

    .. note:: There are actually two different versions of this dataset with major
       formatting differences, including comma-delimited vs. semicolon-delimited,
       empty rows and columns, and differences in datetime formatting. This dataset
       currently only supports the comma-delimited version.

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1016/J.SNB.2007.09.060

    .. versionadded:: 0.10
    """

    url = 'https://archive.ics.uci.edu/static/public/360/data.csv'
    data_file_name = 'data.csv'

    def __init__(
        self,
        root: Path = 'data',
        *,
        input_steps: int = 3,
        target_steps: int = 1,
        input_features: Sequence[str] | None = None,
        target_features: Sequence[str] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: Root directory where dataset can be found.
            input_steps: Number of input time steps to use.
            target_steps: Number of target time steps to use.
            input_features: List of input features to load
                (uses all features by default).
            target_features: List of target features to load
                (uses all features by default).
            download: If True, download dataset and store it in the root directory.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = pathlib.Path(root)
        self.download = download
        self.input_steps = input_steps
        self.target_steps = target_steps
        self.input_features = input_features
        self.target_features = target_features
        self._load_data()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.input_data) - self.input_steps - self.target_steps + 1

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data at that index.
        """
        input = self.input_data.iloc[index : index + self.input_steps]
        target = self.target_data.iloc[
            index + self.input_steps : index + self.input_steps + self.target_steps
        ]

        return {
            'input': torch.tensor(input.values, dtype=torch.float32),
            'target': torch.tensor(target.values, dtype=torch.float32),
        }

    def _parse_datetime(self, data: pd.DataFrame) -> None:
        """Parse datetime columns into cyclical features.

        Args:
            data: Raw data.
        """
        if {'Date', 'Time'} <= set(data.columns):
            dt = pd.to_datetime(data['Date'] + ' ' + data['Time']).dt
            doy = 2 * np.pi * dt.dayofyear / 365.25  # ty: ignore[unsupported-operator]
            hod = 2 * np.pi * dt.hour / 24  # ty: ignore[unsupported-operator]
            data['sin(DOY)'] = np.sin(doy)
            data['cos(DOY)'] = np.cos(doy)
            data['sin(HOD)'] = np.sin(hod)
            data['cos(HOD)'] = np.cos(hod)

        data.drop(columns=['Date', 'Time'], inplace=True, errors='ignore')

    def _load_data(self) -> None:
        """Load the dataset into a pandas dataframe."""
        filepath = self.root / self.data_file_name
        if filepath.is_file():
            pass
        elif self.download:
            filepath = self.url
        else:
            raise DatasetNotFoundError(self)

        # Load twice in case target_features is not a subset of input_features
        kwargs = {'na_values': -200}
        self.input_data = pd.read_csv(filepath, usecols=self.input_features, **kwargs)  # ty: ignore[no-matching-overload]
        self.target_data = pd.read_csv(filepath, usecols=self.target_features, **kwargs)  # ty: ignore[no-matching-overload]

        # Encode cyclic features
        self._parse_datetime(self.input_data)
        self._parse_datetime(self.target_data)

        # Interpolate missing values using linear interpolation
        self.input_data.interpolate(inplace=True)
        self.target_data.interpolate(inplace=True)

    def plot(self, sample: Sample, features: Sequence[str] | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            features: List of features to plot (defaults to *target_features*).

        Returns:
            A matplotlib Figure with the plotted sample.
        """
        ylabel = {
            'CO(GT)': 'CO (mg/m$^3$)',
            'PT08.S1(CO)': 'CO',
            'NMHC(GT)': 'NMHC (μg/m$^3$)',
            'C6H6(GT)': 'C$_6$H$_6$ (μg/m$^3$)',
            'PT08.S2(NMHC)': 'NHMC',
            'NOx(GT)': 'NO$_x$ (ppb)',
            'PT08.S3(NOx)': 'NO$_x$',
            'NO2(GT)': 'NO$_2$ (μg/m$^3$)',
            'PT08.S4(NO2)': 'NO$_2$',
            'PT08.S5(O3)': 'O$_3$',
            'T': 'Temperature (°C)',
            'RH': 'Relative Humidity (%)',
            'AH': 'Absolute Humidity',
        }

        input_steps = range(len(sample['input']))
        target_steps = range(
            len(sample['input']), len(sample['input']) + len(sample['target'])
        )

        features = features or self.target_data.columns
        n_features = len(features)

        ncols = math.ceil(math.sqrt(n_features))
        nrows = math.ceil(n_features / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False
        )
        axes = axes.ravel()

        # For each axis, feature...
        for ax, feature in zip(axes, features):
            ax.set_title(feature)

            # Input data
            if feature in self.input_data:
                idx = self.input_data.columns.get_loc(feature)
                data = sample['input'][:, idx]
                ax.plot(input_steps, data, label='Input', marker='o')

            # Target data
            if feature in self.target_data:
                idx = self.target_data.columns.get_loc(feature)
                data = sample['target'][:, idx]
                ax.plot(target_steps, data, label='Target', marker='x')

                # Predicted data
                if 'prediction' in sample:
                    data = sample['prediction'][:, idx]
                    ax.plot(target_steps, data, label='Prediction', marker='^')

            ax.legend()
            if feature in ylabel:
                ax.set_ylabel(ylabel[feature])

        # Hide unused axes
        for ax in axes[n_features:]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig
