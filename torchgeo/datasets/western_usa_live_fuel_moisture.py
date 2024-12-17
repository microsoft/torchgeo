# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Western USA Live Fuel Moisture Dataset."""

import glob
import json
import os
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns

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
        x_feature: str = None,
        y_feature: str = None,
        kind: str = "scatter",
        title: str = None,
        save_path: str = None,
    ) -> None:
        """Plot features or relationships within the dataset.

        Args:
            x_feature: Name of the feature to plot on the x-axis.
            y_feature: Name of the feature to plot on the y-axis. 
                       Defaults to the label if not specified.
            kind: Type of plot ('scatter', 'hist', 'box', or 'geo').
            title: Title of the plot.
            save_path: If provided, save the plot to the given path.
        """
        if x_feature not in self.input_features:
            raise ValueError(f"'{x_feature}' is not a valid input feature.")
        if y_feature is None:
            y_feature = self.label_name
        if y_feature not in self.input_features and y_feature != self.label_name:
            raise ValueError(f"'{y_feature}' is not a valid feature or label.")

        plt.figure(figsize=(10, 6))

        if kind == "scatter":
            # Scatter plot for feature relationships
            sns.scatterplot(
                x=self.dataframe[x_feature],
                y=self.dataframe[y_feature],
                alpha=0.7,
            )
            plt.xlabel(x_feature)
            plt.ylabel(y_feature)
            plt.title(title or f"Scatter plot: {x_feature} vs {y_feature}")

        elif kind == "hist":
            # Histogram for a single feature
            sns.histplot(self.dataframe[x_feature], kde=True, bins=30, color="blue")
            plt.xlabel(x_feature)
            plt.title(title or f"Distribution of {x_feature}")

        elif kind == "box":
            # Boxplot for feature distributions
            sns.boxplot(y=self.dataframe[x_feature])
            plt.title(title or f"Boxplot of {x_feature}")

        elif kind == "geo":
            # Spatial scatter plot using latitude and longitude
            if "lat" not in self.input_features or "lon" not in self.input_features:
                raise ValueError("Latitude ('lat') and longitude ('lon') must be input features for geo plots.")
            sns.scatterplot(
                x=self.dataframe["lon"],
                y=self.dataframe["lat"],
                hue=self.dataframe[self.label_name],
                palette="viridis",
                alpha=0.7,
            )
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(title or "Geographic Distribution of Fuel Moisture")

        else:
            raise ValueError(f"Plot kind '{kind}' is not supported.")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
