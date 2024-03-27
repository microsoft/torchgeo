# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ChaBuD dataset."""

import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoDataset
from .utils import DatasetNotFoundError, download_url, percentile_normalization


class ChaBuD(NonGeoDataset):
    """ChaBuD dataset.

    `ChaBuD <https://huggingface.co/spaces/competitions/ChaBuD-ECML-PKDD2023>`__
    is a dataset for Change detection for Burned area Delineation and is used
    for the ChaBuD ECML-PKDD 2023 Discovery Challenge.

    Dataset features:

    * Sentinel-2 multispectral imagery
    * binary masks of burned areas
    * 12 multispectral bands
    * 356 pairs of pre and post images with 10 m per pixel resolution (512x512 px)

    Dataset format:

    * single hdf5 dataset containing images and masks

    Dataset classes:

    0. no change
    1. burned area

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1016/j.rse.2021.112603

    .. note::

       This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset

    .. versionadded:: 0.6
    """

    all_bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
    rgb_bands = ["B04", "B03", "B02"]
    folds = {"train": [1, 2, 3, 4], "val": [0]}
    url = "https://hf.co/datasets/chabud-team/chabud-ecml-pkdd2023/resolve/de222d434e26379aa3d4f3dd1b2caf502427a8b2/train_eval.hdf5"  # noqa: E501
    filename = "train_eval.hdf5"
    md5 = "15d78fb825f9a81dad600db828d22c08"

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: list[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new ChaBuD dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train" or "val"
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: If ``split`` or ``bands`` arguments are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.folds
        assert set(bands) <= set(self.all_bands)

        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.filepath = os.path.join(root, self.filename)
        self.band_indices = [self.all_bands.index(b) for b in bands]

        self._verify()

        try:
            import h5py  # noqa: F401
        except ImportError:
            raise ImportError(
                "h5py is not installed and is required to use this dataset"
            )

        self.uuids = self._load_uuids()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            sample containing image and mask
        """
        image = self._load_image(index)
        mask = self._load_target(index)

        sample = {"image": image, "mask": mask}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.uuids)

    def _load_uuids(self) -> list[str]:
        """Return the image uuids for the given split.

        Returns:
            the image uuids
        """
        import h5py

        uuids = []
        with h5py.File(self.filepath, "r") as f:
            for k, v in f.items():
                if v.attrs["fold"] in self.folds[self.split] and "pre_fire" in v:
                    uuids.append(k)

        uuids = sorted(uuids)
        return uuids

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        import h5py

        uuid = self.uuids[index]
        with h5py.File(self.filepath, "r") as f:
            pre_array = f[uuid]["pre_fire"][:]
            post_array = f[uuid]["post_fire"][:]

        # index specified bands and concatenate
        pre_array = pre_array[..., self.band_indices]
        post_array = post_array[..., self.band_indices]
        array = np.concatenate([pre_array, post_array], axis=-1).astype(np.float32)

        tensor = torch.from_numpy(array)
        # Convert from HxWxC to CxHxW
        tensor = tensor.permute((2, 0, 1))
        return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask
        """
        import h5py

        uuid = self.uuids[index]
        with h5py.File(self.filepath, "r") as f:
            array = f[uuid]["mask"][:].astype(np.int32).squeeze(axis=-1)

        tensor = torch.from_numpy(array)
        tensor = tensor.to(torch.long)
        return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if os.path.exists(self.filepath):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        if not os.path.exists(self.filepath):
            download_url(
                self.url,
                self.root,
                filename=self.filename,
                md5=self.md5 if self.checksum else None,
            )

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        mask = sample["mask"].numpy()
        image_pre = sample["image"][: len(self.bands)][rgb_indices].numpy()
        image_post = sample["image"][len(self.bands) :][rgb_indices].numpy()
        image_pre = percentile_normalization(image_pre)
        image_post = percentile_normalization(image_post)

        ncols = 3

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = sample["prediction"]
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 5))

        axs[0].imshow(np.transpose(image_pre, (1, 2, 0)))
        axs[0].axis("off")
        axs[1].imshow(np.transpose(image_post, (1, 2, 0)))
        axs[1].axis("off")
        axs[2].imshow(mask)
        axs[2].axis("off")

        if showing_predictions:
            axs[3].imshow(prediction)
            axs[3].axis("off")

        if show_titles:
            axs[0].set_title("Image Pre")
            axs[1].set_title("Image Post")
            axs[2].set_title("Mask")
            if showing_predictions:
                axs[3].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
