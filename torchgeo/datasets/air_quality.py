# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Air Quality dataset."""

import os

import pandas as pd

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path


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

    .. versionadded:: 0.7
    """

    url = 'https://archive.ics.uci.edu/static/public/360/data.csv'
    data_file_name = 'data.csv'

    def __init__(self, root: Path = 'data', download: bool = False) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            download: if True, download dataset and store it in the root directory
        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.download = download
        self.data = self._load_data()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index: int) -> pd.Series:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data at that index
        """
        return self.data.iloc[index]

    def _load_data(self) -> pd.DataFrame:
        """Load the dataset into a pandas dataframe.

        Returns:
            Dataframe containing the data.
        """
        # Check if the file already exists
        pathname = os.path.join(self.root, self.data_file_name)
        if os.path.exists(pathname):
            return pd.read_csv(pathname)

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        return pd.read_csv(self.url)
