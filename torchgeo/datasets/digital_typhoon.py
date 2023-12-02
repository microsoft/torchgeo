# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Digital Typhoon dataset."""

import glob
import os
import tarfile
from collections.abc import Sequence
from typing import Any, Callable, Optional

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoDataset
from .utils import DatasetNotFoundError, download_url, percentile_normalization


class DigitalTyphoonAnalysis(NonGeoDataset):
    """Digital Tyhphoon Dataset for Analysis Task.

    This dataset contains typhoon-centered images, derived from hourly infrared channel
    images captured by meteorological satellites. It incorporates data from multiple
    generations of the Himawari weather satellite, dating back to 1978. These images
    have been transformed into brightness temperatures and adjusted for varying
    satellite sensor readings, yielding a consistent spatio-temporal dataset that
    covers over four decades.

    See `the Digital Typhoon website
    <http://agora.ex.nii.ac.jp/digital-typhoon/dataset/>`_
    for more information about the dataset.

    Dataset features:

    * infrared channel images from the Himawari weather satellite (512x512 px)
      at 5km spatial resolution
    * auxiliary features such as wind speed, pressure, and more that can be used
      for regression or classification tasks
    * 1,099 typhoons and 189,364 image

    Dataset format:

    * hdf5 files containing the infrared channel images
    * .csv files containing the metadata for each image

    If you use this dataset in your research, please cite the following papers:

    * https://doi.org/10.20783/DIAS.664

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

    url = "https://huggingface.co/datasets/torchgeo/digital_typhoon/resolve/main/WP.tar.gz{0}"  # noqa: E501

    md5sums = {
        "aa": "3af98052aed17e0ddb1e94caca2582e2",
        "ab": "2c5d25455ac8aef1de33fe6456ab2c8d",
    }

    data_root = "WP"

    def __init__(
        self,
        root: str = "data",
        task: str = "regression",
        features: Sequence[str] = ["wind"],
        target: list[str] = ["wind"],
        sequence_length: int = 3,
        min_feature_value: Optional[dict[str, float]] = None,
        max_feature_value: Optional[dict[str, float]] = None,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Digital Typhoon Analysis dataset instance.

        Args:
            root: root directory where dataset can be found
            task: whether to load "regression" or "classification" labels
            features: which auxiliary features to return
            target: which auxiliary features to use as targets
            sequence_length: length of the sequence to return
            min_feature_value: minimum value for each feature
            max_feature_value: maximum value for each feature
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

        self.min_feature_value = min_feature_value
        self.max_feature_value = max_feature_value

        assert task in self.valid_tasks, f"Please choose one of {self.valid_tasks}"
        self.task = task

        assert set(features).issubset(set(self.valid_features))
        self.features = features

        assert set(target).issubset(set(self.valid_features))
        self.target = target

        self._verify()

        self.aux_df = pd.read_csv(
            os.path.join(root, self.data_root, self.aux_file_name)
        )
        self.aux_df["datetime"] = pd.to_datetime(
            self.aux_df[["year", "month", "day", "hour"]]
        )

        self.aux_df = self.aux_df.sort_values(["year", "month", "day", "hour"])
        self.aux_df["seq_id"] = self.aux_df.groupby(["id"]).cumcount()

        # Compute the hour difference between consecutive images per typhoon id
        self.aux_df["hour_diff_consecutive"] = (
            self.aux_df.sort_values(["id", "datetime"])
            .groupby("id")["datetime"]
            .diff()
            .dt.total_seconds()
            / 3600
        )

        # Compute the hour difference between the first and second entry
        self.aux_df["hour_diff_to_next"] = (
            self.aux_df.sort_values(["id", "datetime"])
            .groupby("id", group_keys=False)["datetime"]
            .apply(lambda x: (x - x.shift(-1)).abs().dt.total_seconds() / 3600)
        )

        self.aux_df["hour_diff"] = self.aux_df["hour_diff_consecutive"].combine_first(
            self.aux_df["hour_diff_to_next"]
        )
        self.aux_df.drop(
            ["hour_diff_consecutive", "hour_diff_to_next"], axis=1, inplace=True
        )

        # 0 hour difference is for the last time step of each typhoon sequence and want
        # to keep only images that have max 1 hour difference
        self.aux_df = self.aux_df[self.aux_df["hour_diff"] <= 1]
        # Filter out all ids that only have less than sequence_length entries
        self.aux_df = self.aux_df.groupby("id").filter(
            lambda x: len(x) >= self.sequence_length
        )

        # Filter aux_df according to min_target_value
        if self.min_feature_value is not None:
            for feature, min_value in self.min_feature_value.items():
                self.aux_df = self.aux_df[self.aux_df[feature] >= min_value]

        # Filter aux_df according to max_target_value
        if self.max_feature_value is not None:
            for feature, max_value in self.max_feature_value.items():
                self.aux_df = self.aux_df[self.aux_df[feature] <= max_value]

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
                    self.root,
                    self.data_root,
                    "metadata",
                    str(sample_df.iloc[-1]["id"]) + ".csv",
                ),
                sample_df.iloc[-1]["image_path"],
            )
        )

        # torchgeo expects a single label
        sample["label"] = torch.Tensor([sample[target] for target in self.target])

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
            full_path = os.path.join(self.root, self.data_root, "image", id, filepath)
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

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        exists = []
        path = os.path.join(self.root, self.data_root, "image", "*", "*.h5")
        if glob.glob(path):
            exists.append(True)
        else:
            exists.append(False)

        # check if aux.csv file exists
        exists.append(
            os.path.exists(os.path.join(self.root, self.data_root, self.aux_file_name))
        )
        if all(exists):
            return

        # Check if the tar.gz files have already been downloaded
        exists = []
        for suffix in self.md5sums.keys():
            path = os.path.join(self.root, f"{self.data_root}.tar.gz{suffix}")
            exists.append(os.path.exists(path))

        if all(exists):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download amd extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for suffix, md5 in self.md5sums.items():
            download_url(
                self.url.format(suffix), self.root, md5=md5 if self.checksum else None
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        # Extract tarball
        for suffix in self.md5sums.keys():
            with tarfile.open(
                os.path.join(self.root, f"{self.data_root}.tar.gz{suffix}")
            ) as tar:
                tar.extractall(path=self.root)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image, label = sample["image"], sample["label"]

        image = percentile_normalization(image)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = sample["prediction"]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image.permute(1, 2, 0))
        ax.axis("off")

        if show_titles:
            title_dict = {
                label_name: label[idx].item()
                for idx, label_name in enumerate(self.target)
            }
            title = f"Label: {title_dict}"
            if showing_predictions:
                title_dict = {
                    label_name: prediction[idx].item()
                    for idx, label_name in enumerate(self.target)
                }
                title += f"\nPrediction: {title_dict}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
