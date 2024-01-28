# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""FireRisk dataset."""

import os
from typing import Callable, Optional, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoClassificationDataset
from .utils import DatasetNotFoundError, download_url, extract_archive


class FireRisk(NonGeoClassificationDataset):
    """FireRisk dataset.

    The `FireRisk <https://github.com/CharmonyShen/FireRisk>`__
    dataset is a dataset for remote sensing fire risk classification.

    Dataset features:

    * 91,872 images with 1 m per pixel resolution (320x320 px)
    * 70,331 and 21,541 train and val images, respectively
    * three spectral bands - RGB
    * 7 fire risk classes
    * images extracted from NAIP tiles

    Dataset format:

    * images are three-channel pngs

    Dataset classes:

    0. high
    1. low
    2. moderate
    3. non-burnable
    4. very_high
    5. very_low
    6. water

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2303.07035

    .. versionadded:: 0.5
    """

    url = "https://drive.google.com/file/d/1J5GrJJPLWkpuptfY_kgqkiDtcSNP88OP"
    md5 = "a77b9a100d51167992ae8c51d26198a6"
    filename = "FireRisk.zip"
    directory = "FireRisk"
    splits = ["train", "val"]
    classes = [
        "High",
        "Low",
        "Moderate",
        "Non-burnable",
        "Very_High",
        "Very_Low",
        "Water",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new FireRisk dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "val"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits
        self.root = root
        self.split = split
        self.download = download
        self.checksum = checksum
        self._verify()

        super().__init__(
            root=os.path.join(root, self.directory, self.split), transforms=transforms
        )

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        path = os.path.join(self.root, self.directory)
        if os.path.exists(path):
            return

        # Check if zip file already exists (if so then extract)
        filepath = os.path.join(self.root, self.filename)
        if os.path.exists(filepath):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`NonGeoClassificationDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample["image"].permute((1, 2, 0)).numpy()
        label = cast(int, sample["label"].item())
        label_class = self.classes[label]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = cast(int, sample["prediction"].item())
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Label: {label_class}"
            if showing_predictions:
                title += f"\nPrediction: {prediction_class}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
