# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SustainBench Crop Yield dataset."""

import os
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_url, extract_archive


class SustainBenchCropYield(NonGeoDataset):
    """SustainBench Crop Yield Dataset.

    This dataset contains MODIS band histograms and soybean yield
    estimates for selected counties in the USA, Argentina and Brazil.
    The dataset is part of the
    `SustainBench <https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/crop_yield.html>`_
    datasets for tackling the UN Sustainable Development Goals (SDGs).

    Dataset Format:

    * .npz files of stacked samples

    Dataset Features:

    * input histogram of 7 surface reflectance and 2 surface temperature
      bands from MODIS pixel values in 32 ranges across 32 timesteps
      resulting in 32x32x9 input images
    * regression target value of soybean yield in metric tonnes per
      harvested hectare

    If you use this dataset in your research, please cite:

    * https://doi.org/10.1145/3209811.3212707
    * https://doi.org/10.1609/aaai.v31i1.11172

    .. versionadded:: 0.5
    """  # noqa: E501

    valid_countries = ["usa", "brazil", "argentina"]

    md5 = "c2794e59512c897d9bea77b112848122"

    url = "https://drive.google.com/file/d/1odwkI1hiE5rMZ4VfM0hOXzlFR4NbhrfU/view?usp=share_link"  # noqa: E501

    dir = "soybeans"

    valid_splits = ["train", "dev", "test"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        countries: list[str] = ["usa"],
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "dev", or "test"
            countries: which countries to include in the dataset
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            AssertionError: if ``countries`` contains invalid countries or if ``split``
                is invalid
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        assert set(countries).issubset(
            self.valid_countries
        ), f"Please choose a subset of these valid countried: {self.valid_countries}."
        self.countries = countries

        assert (
            split in self.valid_splits
        ), f"Pleas choose one of these valid data splits {self.valid_splits}."
        self.split = split

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()
        self.collection = self.retrieve_collection()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.collection)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        input_file_path, sample_idx = self.collection[index]

        sample: dict[str, Tensor] = {
            "image": self._load_image(input_file_path, sample_idx)
        }
        sample.update(self._load_features(input_file_path, sample_idx))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: str, sample_idx: int) -> Tensor:
        """Load input image.

        Args:
            path: path to input npz collection
            sample_idx: what sample to index from the npz collection

        Returns:
            input image as tensor
        """
        arr = np.load(path)["data"][sample_idx]
        # return [channel, height, width]
        return torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32)

    def _load_features(self, path: str, sample_idx: int) -> dict[str, Tensor]:
        """Load features value.

        Args:
            path: path to image npz collection
            sample_idx: what sample to index from the npz collection

        Returns:
            target regression value
        """
        target_file_path = path.replace("_hists", "_yields")
        target = np.load(target_file_path)["data"][sample_idx]

        years_file_path = path.replace("_hists", "_years")
        year = int(np.load(years_file_path)["data"][sample_idx])

        ndvi_file_path = path.replace("_hists", "_ndvi")
        ndvi = np.load(ndvi_file_path)["data"][sample_idx]

        features = {
            "label": torch.tensor(target).to(torch.float32),
            "year": torch.tensor(year),
            "ndvi": torch.from_numpy(ndvi).to(dtype=torch.float32),
        }
        return features

    def retrieve_collection(self) -> list[tuple[str, int]]:
        """Retrieve the collection.

        Returns:
            path and index to dataset samples
        """
        collection = []
        for country in self.countries:
            file_path = os.path.join(
                self.root, self.dir, country, f"{self.split}_hists.npz"
            )
            npz_file = np.load(file_path)
            num_data_points = npz_file["data"].shape[0]
            for idx in range(num_data_points):
                collection.append((file_path, idx))

        return collection

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, self.dir)
        if os.path.exists(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.dir) + ".zip"
        if os.path.exists(pathname):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                f"Dataset not found in `root={self.root}` and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset and extract it.

        Raises:
            RuntimeError: if download doesn't work correctly or checksums don't match
        """
        download_url(
            self.url,
            self.root,
            filename=self.dir,
            md5=self.md5 if self.checksum else None,
        )
        self._extract()

    def _extract(self) -> None:
        """Extract the dataset."""
        zipfile_path = os.path.join(self.root, self.dir) + ".zip"
        extract_archive(zipfile_path, self.root)

    def plot(
        self,
        sample: dict[str, Any],
        band_idx: int = 0,
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            band_idx: which of the nine histograms to index
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        """
        image, label = sample["image"], sample["label"].item()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = sample["prediction"].item()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.imshow(image.permute(1, 2, 0)[:, :, band_idx])
        ax.axis("off")

        if show_titles:
            title = f"Label: {label:.3f}"
            if showing_predictions:
                title += f"\nPrediction: {prediction:.3f}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
