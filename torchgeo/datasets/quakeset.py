# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""QuakeSet dataset."""

import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .geo import NonGeoDataset
from .utils import DatasetNotFoundError, download_url, percentile_normalization


class QuakeSet(NonGeoDataset):
    """QuakeSet dataset.

    `QuakeSet <https://huggingface.co/datasets/DarthReca/quakeset>`__
    is a dataset for Earthquake Change Detection and Magnitude Estimation and is used
    for the Seismic Monitoring and Analysis (SMAC) ECML-PKDD 2024 Discovery Challenge.

    Dataset features:

    * Sentinel-1 SAR imagery
    * before/pre/post imagery of areas affected by earthquakes
    * 2 multispectral bands (VV/VH)
    * 356 pairs of pre and post images with 5 m per pixel resolution (512x512 px)

    Dataset format:

    * single hdf5 dataset containing images, magnitudes, hypercenters, and splits

    Dataset classes:

    0. unaffected area
    1. earthquake affected area

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2403.18116

    .. note::

       This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset

    .. versionadded:: 0.6
    """

    all_bands = ["VV", "VH"]
    filename = "earthquakes.h5"
    url = "https://hf.co/datasets/DarthReca/quakeset/resolve/main/earthquakes.h5",
    md5 = "76fc7c76b7ca56f4844d852e175e1560"
    splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: list[str] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new QuakeSet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: If ``split`` or ``bands`` arguments are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.splits
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

        self.data = self._load_data()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            sample containing image and mask
        """
        image = self._load_image(index)
        label = torch.tensor(self.data[index]["label"])
        magnitude = torch.tensor(self.data[index]["magnitude"])

        sample = {"image": image, "label": label, "magnitude": magnitude}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.data)

    def _load_data(self) -> list[dict[str, str | tuple[str, str], int | float]]:
        """Return the metadata for a given split.

        Returns:
            the sample keys, patches, images, labels, and magnitudes
        """
        import h5py

        f = h5py.File(self.filepath)

        data = []
        for k in sorted(f.keys()):
            if f[k].attrs["split"] != self.split:
                continue

            for patch in sorted(f[k].keys()):
                if patch not in ["x", "y"]:
                    # positive sample
                    magnitude = float(f[k].attrs["magnitude"])
                    data.append(dict(key=k, patch=patch, images=("pre", "post"), label=1, magnitude=magnitude))

                    # hard negative sample
                    if "before" in f[k][patch].keys():
                        data.append(dict(key=k, patch=patch, images=("before", "pre"), label=0, magnitude=0.0))
        f.close()
        return data

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        import h5py

        key = self.data[index]["key"]
        patch = self.data[index]["patch"]
        images = self.data[index]["images"]

        with h5py.File(self.filepath) as f:
            pre_array = f[key][patch][images[0]][:]
            post_array = f[key][patch][images[1]][:]

        # index specified bands and concatenate
        pre_array = pre_array[..., self.band_indices]
        post_array = post_array[..., self.band_indices]
        array = np.concatenate([pre_array, post_array], axis=-1).astype(np.float32)

        tensor = torch.from_numpy(array)
        # Convert from HxWxC to CxHxW
        tensor = tensor.permute((2, 0, 1))
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
