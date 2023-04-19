# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SKy Images and Photovoltaic Power Dataset (SKIPP'D)."""

import os
from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from .geo import NonGeoDataset
from .utils import download_url, extract_archive


class SKIPPD(NonGeoDataset):
    """SKy Images and Photovoltaic Power Dataset (SKIPP'D).

    The `SKIPP'D dataset <https://purl.stanford.edu/dj417rh1007>`_
    contains ground-based fish-eye photos of the sky for solar
    forecasting tasks.

    Dataset Format:

    * .hdf5 file containing images and labels
    * .npy files with corresponding datetime timestamps

    Dataset Features:

    * fish-eye RGB images (64x64px)
    * power output measurements from 30-kW rooftop PV array
    * 1-min interval across 3 years (2017-2019)
    * 349,372 images under the split key *trainval*
    * 14,003 images under the split key *test*

    If you use this dataset in your research, please cite:

    * https://doi.org/10.48550/arXiv.2207.00913

    .. versionadded:: 0.5
    """

    url = "https://stacks.stanford.edu/object/dj417rh1007"
    md5 = "b38d0f322aaeb254445e2edd8bc5d012"

    img_file_name = "2017_2019_images_pv_processed.hdf5"

    data_dir = "dj417rh1007"

    valid_splits = ["trainval", "test"]

    dateformat = "%m/%d/%Y, %H:%M:%S"

    def __init__(
        self,
        root: str = "data",
        split: str = "trainval",
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "trainval", or "test"
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            AssertionError: if ``countries`` contains invalid countries
            ImportError: if h5py is not installed
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        assert (
            split in self.valid_splits
        ), f"Pleas choose one of these valid data splits {self.valid_splits}."
        self.split = split

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        try:
            import h5py  # noqa: F401
        except ImportError:
            raise ImportError(
                "h5py is not installed and is required to use this dataset"
            )

        self._verify()

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        import h5py

        with h5py.File(
            os.path.join(self.root, self.data_dir, self.img_file_name), "r"
        ) as f:
            num_datapoints: int = f[self.split]["pv_log"].shape[0]

        return num_datapoints

    def __getitem__(self, index: int) -> dict[str, Union[str, Tensor]]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        sample: dict[str, Union[str, Tensor]] = {"image": self._load_image(index)}
        sample.update(self._load_features(index))

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, index: int) -> Tensor:
        """Load the input image.

        Args:
            index: index of image to load

        Returns:
            image tensor at index
        """
        import h5py

        with h5py.File(
            os.path.join(self.root, self.data_dir, self.img_file_name), "r"
        ) as f:
            arr = f[self.split]["images_log"][index, :, :, :]

        # put channel first
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32)
        return tensor

    def _load_features(self, index: int) -> dict[str, Union[str, Tensor]]:
        """Load label.

        Args:
            index: index of label to load

        Returns:
            label tensor at index
        """
        import h5py

        with h5py.File(
            os.path.join(self.root, self.data_dir, self.img_file_name), "r"
        ) as f:
            label = f[self.split]["pv_log"][index]

        path = os.path.join(self.root, self.data_dir, f"times_{self.split}.npy")
        datestring = np.load(path, allow_pickle=True)[index].strftime(self.dateformat)

        # put channel first
        features: dict[str, Union[str, Tensor]] = {
            "label": torch.tensor(label, dtype=torch.float32),
            "date": datestring,
        }
        return features

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the extracted files already exist
        pathname = os.path.join(self.root, self.data_dir)
        if os.path.exists(pathname):
            return

        # Check if the zip files have already been downloaded
        pathname = os.path.join(self.root, self.data_dir) + ".zip"
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
            filename=self.data_dir,
            md5=self.md5 if self.checksum else None,
        )
        self._extract()

    def _extract(self) -> None:
        """Extract the dataset."""
        zipfile_path = os.path.join(self.root, self.data_dir) + ".zip"
        extract_archive(zipfile_path, self.root)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
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

        ax.imshow(image.permute(1, 2, 0) / 255)
        ax.axis("off")

        if show_titles:
            title = f"Label: {label:.3f}"
            if showing_predictions:
                title += f"\nPrediction: {prediction:.3f}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
