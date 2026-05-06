# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Air Quality dataset."""

import os

import matplotlib.pyplot as plt
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

    * hourly averaged sensor responses and reference analyzer ground truth over one year (2004-2005)
    * has missing features

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1016/J.SNB.2007.09.060

    .. versionadded:: 0.10
    """

    url = 'https://archive.ics.uci.edu/static/public/360/data.csv'
    data_file_name = 'data.csv'

    def __init__(
        self,
        root: Path = 'data',
        download: bool = False,
        num_input_steps: int = 3,
        num_target_steps: int = 1,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            download: if True, download dataset and store it in the root directory
            num_input_steps: Number of input time steps to use.
            num_target_steps: Number of target time steps to use.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.download = download
        self.num_input_steps = num_input_steps
        self.num_target_steps = num_target_steps
        self.data = self._load_data()
        self.feature_names = [c for c in self.data.columns if c != 'datetime']

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.data) - (self.num_input_steps + self.num_target_steps)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data at that index
        """
        x_input = self.data.iloc[index : index + self.num_input_steps]
        y_target = self.data.iloc[
            index + self.num_input_steps : index
            + self.num_input_steps
            + self.num_target_steps
        ]

        return {
            'x_input': torch.tensor(x_input.values, dtype=torch.float32),
            'y_target': torch.tensor(y_target.values, dtype=torch.float32),
        }

    def _load_data(self) -> pd.DataFrame:
        """Load the dataset into a pandas dataframe.

        Returns:
            Dataframe containing the data.
        """
        pathname = os.path.join(self.root, self.data_file_name)
        if os.path.exists(pathname):
            df = pd.read_csv(pathname, na_values=-200)
        elif not self.download:
            raise DatasetNotFoundError(self)
        else:
            df = pd.read_csv(self.url, na_values=-200)

        # Combine Date and Time into a single numeric column
        df.drop(columns=['Date', 'Time'], inplace=True)

        # Interpolate missing values
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.interpolate(method='linear', limit_direction='both')
        df = df.ffill().bfill()
        df.reset_index(drop=True, inplace=True)

        return df

    def plot(self, sample: Sample) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`

        Returns:
            a matplotlib Figure with the plotted sample
        """
        input = sample['x_input'].numpy()
        target = sample['y_target'].numpy()

        num_features = input.shape[1]
        fig: Figure
        fig, axes = plt.subplots(num_features, 1, figsize=(10, 2 * num_features))
        if num_features == 1:
            axes = [axes]

        input_steps = range(self.num_input_steps)
        target_steps = range(
            self.num_input_steps, self.num_input_steps + self.num_target_steps
        )

        for i, ax in enumerate(axes):
            feature_name = self.feature_names[i]

            ax.plot(input_steps, input[:, i], label='Input', marker='o')
            ax.plot(
                target_steps, target[:, i], label='Target', marker='x', linestyle='--'
            )

            ax.set_title(feature_name)
            ax.legend()
            ax.set_xlabel('Time step')

        plt.tight_layout()
        return fig
