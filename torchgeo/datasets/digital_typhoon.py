# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Digital Typhoon dataset."""

import os
from collections.abc import Sequence
from typing import Any, Callable, Optional

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoDataset
from .utils import check_integrity, download_url, extract_archive


class DigitalTyphoonAnalysis(NonGeoDataset):
    """Digital Tyhphoon Dataset for Analysis Task.


    .. versionadded:: 0.6
    """

    valid_tasks = ["classification", "regression"]
    aux_file_name = "aux_data.csv"

    valid_features = [
        "year",
        "month",
        "day",
        "hour",
        "grade",
        "lat",
        "lng",
        "pressure",
        "wind",
        "dir50",
        "long50",
        "short50",
        "dir30",
        "long30",
        "short30",
        "landfall",
        "intp",
    ]

    def __init__(
        self,
        root: str = "data",
        task: str = "regression",
        features: Sequence[str] = ["wind"],
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Digital Typhoon Analysis dataset instance.

        Args:
            root: root directory where dataset can be found
            task: whether to load "regression" or "classification" labels
            features: which auxiliary features to return
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``task`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match
        """
        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        assert task in self.valid_tasks, f"Please choose one of {self.valid_tasks}"
        self.task = task

        assert set(features).issubset(set(self.valid_features))
        self.features = features

        # get all the hf file paths and save them to a dictionary with corresponding id
        self.aux_df = pd.read_csv(os.path.join(root, self.aux_file_name))

        # processing based on different tasks here?

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, and metadata at that index
        """
        sample_entry = self.aux_df.iloc[index]

        sample = {
            "image": self._load_image(
                os.path.join(
                    self.root,
                    "image",
                    str(sample_entry["id"]),
                    sample_entry["image_path"],
                )
            )
        }
        sample.update(
            self._load_features(
                os.path.join(self.root, "metadata", str(sample_entry["id"]) + ".csv"),
                sample_entry["image_path"],
            )
        )

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """REturn the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.aux_df)

    def _load_image(self, filepath: str) -> Tensor:
        """Load a single image.

        Args:
            filepath: path of the image file to load

        Returns:
            the image
        """
        with h5py.File(filepath, "r") as h5f:
            # tensor with added channel dimension
            tensor = torch.from_numpy(h5f["Infrared"][:]).unsqueeze(0)
        return tensor

    def _load_features(self, filepath: str, image_path: str) -> dict[str, Any]:
        """Load features for the corresponding image.

        Args:
            filepath: path of the feature file to load
            image_path: image path for the unique image for which to retrieve features

        Returns:
            features for image
        """
        feature_df = pd.read_csv(filepath)
        feature_df = feature_df[feature_df["file_1"] == image_path]
        feature_dict = {
            name: torch.tensor(feature_df[name].item()) for name in self.features
        }
        return feature_dict
