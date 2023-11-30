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
    valid_splits = ["train", "test"]

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
        sequence_length: int = 3,
        split: str = "train",
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
        self.sequence_length = sequence_length

        assert task in self.valid_tasks, f"Please choose one of {self.valid_tasks}"
        self.task = task

        assert split in self.valid_splits, f"Please choose one of {self.valid_splits}"

        assert set(features).issubset(set(self.valid_features))
        self.features = features

        self.aux_df = pd.read_csv(os.path.join(root, self.aux_file_name))
        self.aux_df["datetime"] = pd.to_datetime(
            self.aux_df[["year", "month", "day", "hour"]]
        )

        self.aux_df = self.aux_df.sort_values(["year", "month", "day", "hour"])
        self.aux_df["seq_id"] = self.aux_df.groupby(["id"]).cumcount()

        # Compute the hour difference between consecutive images per typhoon id
        self.aux_df["hour_diff"] = (
            self.aux_df.sort_values(["id", "datetime"])
            .groupby("id")["datetime"]
            .diff()
            .dt.total_seconds()
            / 3600
        )

        # 0 hour difference is for the last time step of each typhoon sequence and want to keep only images that have max 1 hour difference
        self.aux_df = self.aux_df[self.aux_df["hour_diff"] <= 1]
        # Filter out all ids that only have less than sequence_length entries
        self.aux_df = self.aux_df.groupby("id").filter(
            lambda x: len(x) >= self.sequence_length
        )

        def get_subsequences(df: pd.DataFrame, k: int) -> list[dict[str, list[int]]]:
            """Generate all possible subsequences of length k for a given group.

            Args:
                df: grouped dataframe of a single typhoon
                k: length of the subsequences to generate

            Returns:
                list of all possible subsequences of length k for a given typhoon id
            """
            # generate possible subsquences of length k for each group
            subsequences = [
                {"id": df["id"].iloc[0], "seq_id": list(range(i, i + k))}
                for i in range(len(df) - k + 1)
            ]
            return [
                subseq
                for subseq in subsequences
                if set(subseq["seq_id"]).issubset(df["seq_id"])
            ]

        # self.sample_sequences = self.aux_df.groupby('id').apply(get_subsequences, k=self.sequence_length).tolist()
        self.sample_sequences: list[dict[str, list[int]]] = [
            item
            for sublist in self.aux_df.groupby("id")
            .apply(get_subsequences, k=self.sequence_length)
            .tolist()
            for item in sublist
        ]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data, labels, and metadata at that index
        """
        sample_entry = self.sample_sequences[index]
        sample_df = self.aux_df[
            (self.aux_df["id"] == sample_entry["id"])
            & (self.aux_df["seq_id"].isin(sample_entry["seq_id"]))
        ]

        sample = {"image": self._load_image(sample_df)}
        # load features of the last image in the sequence
        sample.update(
            self._load_features(
                os.path.join(
                    self.root, "metadata", str(sample_df.iloc[-1]["id"]) + ".csv"
                ),
                sample_df.iloc[-1]["image_path"],
            )
        )

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.sample_sequences)

    def _load_image(self, sample_df: pd.DataFrame) -> Tensor:
        """Load a single image.

        Args:
            sample_df: df holding all information necessary to load the
                consecutive images in the sequence

        Returns:
            concatenation of all images in the sequence over channel dimension
        """

        def load_image_tensor(id: str, filepath: str) -> Tensor:
            full_path = os.path.join(self.root, "image", id, filepath)
            with h5py.File(full_path, "r") as h5f:
                # tensor with added channel dimension
                tensor = torch.from_numpy(h5f["Infrared"][:]).unsqueeze(0)
            return tensor

        # tensor of shape [sequence_length, height, width]
        tensor = torch.cat(
            [
                load_image_tensor(str(id), filepath)
                for id, filepath in zip(sample_df["id"], sample_df["image_path"])
            ]
        )
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
