# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Air Quality dataset."""

import os

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

    .. versionadded:: 0.9
    """

    url = 'https://archive.ics.uci.edu/static/public/360/data.csv'
    data_file_name = 'data.csv'

    def __init__(
        self,
        root: Path = 'data',
        download: bool = False,
        num_past_steps: int = 3,
        num_future_steps: int = 1,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            download: if True, download dataset and store it in the root directory
            num_past_steps: Number of past time steps to use.
            num_future_steps: Number of future time steps to use.

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.download = download
        self.num_past_steps = num_past_steps
        self.num_future_steps = num_future_steps
        self.data = self._load_data()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.data) - (self.num_past_steps + self.num_future_steps)

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data at that index
        """
        past_targets = self.data.iloc[index : index + self.num_past_steps]
        future_targets = self.data.iloc[
            index + self.num_past_steps : index
            + self.num_past_steps
            + self.num_future_steps
        ]

        return {
            'past_targets': torch.tensor(past_targets.values, dtype=torch.float32),
            'future_targets': torch.tensor(future_targets.values, dtype=torch.float32),
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
        df['datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'],
            format='%m/%d/%Y %H:%M:%S',  # month/day/year
        )
        df.drop(columns=['Date', 'Time'], inplace=True)

        # Convert datetime64 to float (Unix timestamp in seconds) so it can become a Tensor
        df['datetime'] = df['datetime'].astype('int64') / 1e9

        # Drop rows with any remaining NaNs
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def plot(self, sample: Sample) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`

        Returns:
            a matplotlib Figure with the plotted sample
        """
        import matplotlib.pyplot as plt

        past = sample['past_targets'].numpy()
        future = sample['future_targets'].numpy()

        num_features = past.shape[1]
        fig: Figure
        fig, axes = plt.subplots(num_features, 1, figsize=(10, 2 * num_features))
        if num_features == 1:
            axes = [axes]

        past_steps = range(self.num_past_steps)
        future_steps = range(
            self.num_past_steps, self.num_past_steps + self.num_future_steps
        )

        for i, ax in enumerate(axes):
            ax.plot(past_steps, past[:, i], label='Past', marker='o')
            ax.plot(
                future_steps, future[:, i], label='Future', marker='x', linestyle='--'
            )
            ax.set_title(f'Feature {i}')
            ax.legend()
            ax.set_xlabel('Time step')

        plt.tight_layout()
        return fig
