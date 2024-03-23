# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Western USA Live Fuel Moisture Dataset."""

import glob
import json
import os
from typing import Any, Callable, Optional

import pandas as pd
import torch
from torch import Tensor

from .geo import NonGeoDataset
from .utils import (
    DatasetNotFoundError,
    download_radiant_mlhub_collection,
    extract_archive,
)


class WesternUSALiveFuelMoisture(NonGeoDataset):
    """Western USA Live Fuel Moisture Dataset.

    This tabular style dataset contains fuel moisture
    (mass of water in vegetation) and remotely sensed variables
    in the western United States. It contains 2615 datapoints and 138
    variables. For more details see the
    `dataset page <https://mlhub.earth/data/su_sar_moisture_content_main>`_.

    Dataset Format:

    * .geojson file for each datapoint

    Dataset Features:

    * 138 remote sensing derived variables, some with a time dependency
    * 2615 datapoints with regression target of predicting fuel moisture

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.rse.2020.111797

    .. note::

       This dataset requires the following additional library to be installed:

       * `radiant-mlhub <https://pypi.org/project/radiant-mlhub/>`_ to download the
         imagery and labels from the Radiant Earth MLHub

    .. versionadded:: 0.5
    """

    collection_id = "su_sar_moisture_content"

    md5 = "a6c0721f06a3a0110b7d1243b18614f0"

    label_name = "percent(t)"

    all_variable_names = [
        # "date",
        "slope(t)",
        "elevation(t)",
        "canopy_height(t)",
        "forest_cover(t)",
        "silt(t)",
        "sand(t)",
        "clay(t)",
        "vv(t)",
        "vh(t)",
        "red(t)",
        "green(t)",
        "blue(t)",
        "swir(t)",
        "nir(t)",
        "ndvi(t)",
        "ndwi(t)",
        "nirv(t)",
        "vv_red(t)",
        "vv_green(t)",
        "vv_blue(t)",
        "vv_swir(t)",
        "vv_nir(t)",
        "vv_ndvi(t)",
        "vv_ndwi(t)",
        "vv_nirv(t)",
        "vh_red(t)",
        "vh_green(t)",
        "vh_blue(t)",
        "vh_swir(t)",
        "vh_nir(t)",
        "vh_ndvi(t)",
        "vh_ndwi(t)",
        "vh_nirv(t)",
        "vh_vv(t)",
        "slope(t-1)",
        "elevation(t-1)",
        "canopy_height(t-1)",
        "forest_cover(t-1)",
        "silt(t-1)",
        "sand(t-1)",
        "clay(t-1)",
        "vv(t-1)",
        "vh(t-1)",
        "red(t-1)",
        "green(t-1)",
        "blue(t-1)",
        "swir(t-1)",
        "nir(t-1)",
        "ndvi(t-1)",
        "ndwi(t-1)",
        "nirv(t-1)",
        "vv_red(t-1)",
        "vv_green(t-1)",
        "vv_blue(t-1)",
        "vv_swir(t-1)",
        "vv_nir(t-1)",
        "vv_ndvi(t-1)",
        "vv_ndwi(t-1)",
        "vv_nirv(t-1)",
        "vh_red(t-1)",
        "vh_green(t-1)",
        "vh_blue(t-1)",
        "vh_swir(t-1)",
        "vh_nir(t-1)",
        "vh_ndvi(t-1)",
        "vh_ndwi(t-1)",
        "vh_nirv(t-1)",
        "vh_vv(t-1)",
        "slope(t-2)",
        "elevation(t-2)",
        "canopy_height(t-2)",
        "forest_cover(t-2)",
        "silt(t-2)",
        "sand(t-2)",
        "clay(t-2)",
        "vv(t-2)",
        "vh(t-2)",
        "red(t-2)",
        "green(t-2)",
        "blue(t-2)",
        "swir(t-2)",
        "nir(t-2)",
        "ndvi(t-2)",
        "ndwi(t-2)",
        "nirv(t-2)",
        "vv_red(t-2)",
        "vv_green(t-2)",
        "vv_blue(t-2)",
        "vv_swir(t-2)",
        "vv_nir(t-2)",
        "vv_ndvi(t-2)",
        "vv_ndwi(t-2)",
        "vv_nirv(t-2)",
        "vh_red(t-2)",
        "vh_green(t-2)",
        "vh_blue(t-2)",
        "vh_swir(t-2)",
        "vh_nir(t-2)",
        "vh_ndvi(t-2)",
        "vh_ndwi(t-2)",
        "vh_nirv(t-2)",
        "vh_vv(t-2)",
        "slope(t-3)",
        "elevation(t-3)",
        "canopy_height(t-3)",
        "forest_cover(t-3)",
        "silt(t-3)",
        "sand(t-3)",
        "clay(t-3)",
        "vv(t-3)",
        "vh(t-3)",
        "red(t-3)",
        "green(t-3)",
        "blue(t-3)",
        "swir(t-3)",
        "nir(t-3)",
        "ndvi(t-3)",
        "ndwi(t-3)",
        "nirv(t-3)",
        "vv_red(t-3)",
        "vv_green(t-3)",
        "vv_blue(t-3)",
        "vv_swir(t-3)",
        "vv_nir(t-3)",
        "vv_ndvi(t-3)",
        "vv_ndwi(t-3)",
        "vv_nirv(t-3)",
        "vh_red(t-3)",
        "vh_green(t-3)",
        "vh_blue(t-3)",
        "vh_swir(t-3)",
        "vh_nir(t-3)",
        "vh_ndvi(t-3)",
        "vh_ndwi(t-3)",
        "vh_nirv(t-3)",
        "vh_vv(t-3)",
        "lat",
        "lon",
    ]

    def __init__(
        self,
        root: str = "data",
        input_features: list[str] = all_variable_names,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        download: bool = False,
        api_key: Optional[str] = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Western USA Live Fuel Moisture Dataset.

        Args:
            root: root directory where dataset can be found
            input_features: which input features to include
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``input_features`` contains invalid variable names
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        super().__init__()

        self.root = root
        self.transforms = transforms
        self.checksum = checksum
        self.download = download
        self.api_key = api_key

        self._verify()

        assert all(
            input in self.all_variable_names for input in input_features
        ), "Invalid input variable name."
        self.input_features = input_features

        self.collection = self._retrieve_collection()

        self.dataframe = self._load_data()

    def _retrieve_collection(self) -> list[str]:
        """Retrieve dataset collection that maps samples to paths.

        Returns:
            list of sample paths
        """
        return glob.glob(
            os.path.join(self.root, self.collection_id, "**", "labels.geojson")
        )

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

        sample: dict[str, Tensor] = {
            "input": torch.tensor(
                data.drop([self.label_name]).values, dtype=torch.float32
            ),
            "label": torch.tensor(data[self.label_name], dtype=torch.float32),
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
        for path in self.collection:
            with open(path) as f:
                content = json.load(f)
                data_dict = content["properties"]
                data_dict["lon"] = content["geometry"]["coordinates"][0]
                data_dict["lat"] = content["geometry"]["coordinates"][1]
                data_rows.append(data_dict)

        df: pd.DataFrame = pd.DataFrame(data_rows)
        df = df[self.input_features + [self.label_name]]
        return df

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, self.collection_id)
        if os.path.exists(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.collection_id) + ".tar.gz"
        if os.path.exists(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _extract(self) -> None:
        """Extract the dataset."""
        pathname = os.path.join(self.root, self.collection_id) + ".tar.gz"
        extract_archive(pathname, self.root)

    def _download(self, api_key: Optional[str] = None) -> None:
        """Download the dataset and extract it.

        Args:
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
        """

        ##-- feb2024
        ## azcopy sync https://radiantearth.blob.core.windows.net/mlhub/su-sar-moisture-content . --recursive=true

        download_radiant_mlhub_collection(self.collection_id, self.root, api_key)
        filename = os.path.join(self.root, self.collection_id) + ".tar.gz"
        extract_archive(filename, self.root)
