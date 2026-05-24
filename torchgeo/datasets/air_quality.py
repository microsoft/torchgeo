# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Air Quality dataset."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import math
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
        num_input_steps: int = 3,
        num_target_steps: int = 1,
        features: list[str] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            num_input_steps: Number of input time steps to use.
            num_target_steps: Number of target time steps to use.
            features: Optional list of feature names to keep. If None, all features are used.
            download: if True, download dataset and store it in the root directory

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.download = download
        self.num_input_steps = num_input_steps
        self.num_target_steps = num_target_steps
        self.features = features
        self.data = self._load_data()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.data) - self.num_input_steps - self.num_target_steps + 1

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data at that index
        """
        input = self.data.iloc[index : index + self.num_input_steps]
        target = self.data.iloc[
            index + self.num_input_steps : index
            + self.num_input_steps
            + self.num_target_steps
        ]

        return {
            'input': torch.tensor(input.values, dtype=torch.float32),
            'target': torch.tensor(target.values, dtype=torch.float32),
        }

    def _load_data(self) -> pd.DataFrame:
        """Load the dataset into a pandas dataframe.

        Returns:
            Dataframe containing the data.
        """
        pathname = os.path.join(self.root, self.data_file_name)
        if os.path.exists(pathname):
            df = pd.read_csv(pathname, na_values=['-200'])
        elif not self.download:
            raise DatasetNotFoundError(self)
        else:
            df = pd.read_csv(self.url, na_values=['-200'])

        # Drop Date and Time, not yet using these inputs
        df.drop(columns=['Date', 'Time'], inplace=True)

        # Drop NMHC(GT) column which has mostly missing values
        df.drop(columns=['NMHC(GT)'], inplace=True)

        # Interpolate missing values
        df.interpolate(inplace=True)

        if self.features is not None:
            invalid = set(self.features) - set(df.columns)
            if invalid:
                raise ValueError(f'Requested features not available in dataset: {invalid}')
            df = df[self.features]

        self.feature_names = list(df.columns)

        return df

    def plot(self, sample: Sample, features: list[str] | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            features: optional list of feature names to plot.
                If None, all features are plotted.

        Returns:
            a matplotlib Figure with the plotted sample
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

        x_in = sample['input']
        x_out = sample['target']

        # Normalize feature selection
        features = features or self.feature_names
        feature_indices = [self.feature_names.index(f) for f in features]

        n_features = len(features)
        ncols = math.ceil(math.sqrt(n_features))
        nrows = math.ceil(n_features / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 3 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        input_steps = range(len(x_in))
        target_steps = range(len(x_in), len(x_in) + len(x_out))

        for ax, idx, feature in zip(axes, feature_indices, features):
            ax.plot(input_steps, x_in[:, idx], label='Input', marker='o')
            ax.plot(target_steps, x_out[:, idx], label='Target', marker='x')

            ax.set_title(feature)
            ax.set_ylabel(ylabel.get(feature, feature))
            ax.legend()

        # Hide unused axes
        for ax in axes[n_features:]:
            ax.set_visible(False)

        fig.tight_layout()
        return fig
