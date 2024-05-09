# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""QuakeSet dataset."""

import os
from collections.abc import Callable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import download_url, lazy_import, percentile_normalization


class QuakeSet(NonGeoDataset):
    """QuakeSet dataset.

    `QuakeSet <https://huggingface.co/datasets/DarthReca/quakeset>`__
    is a dataset for Earthquake Change Detection and Magnitude Estimation and is used
    for the Seismic Monitoring and Analysis (SMAC) ECML-PKDD 2024 Discovery Challenge.

    Dataset features:

    * Sentinel-1 SAR imagery
    * before/pre/post imagery of areas affected by earthquakes
    * 2 SAR bands (VV/VH)
    * 3,327 pairs of pre and post images with 5 m per pixel resolution (512x512 px)
    * 2 classification labels (unaffected / affected by earthquake)
    * pre/post image pairs represent earthquake affected areas
    * before/pre image pairs represent hard negative unaffected areas
    * earthquake magnitudes for each sample

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

    filename = 'earthquakes.h5'
    url = 'https://hf.co/datasets/DarthReca/quakeset/resolve/bead1d25fb9979dbf703f9ede3e8b349f73b29f7/earthquakes.h5'
    md5 = '76fc7c76b7ca56f4844d852e175e1560'
    splits = {'train': 'train', 'val': 'validation', 'test': 'test'}
    classes = ['unaffected_area', 'earthquake_affected_area']

    def __init__(
        self,
        root: str = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new QuakeSet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: If ``split`` argument is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
            MissingDependencyError: If h5py is not installed.
        """
        lazy_import('h5py')

        assert split in self.splits

        self.root = root
        self.split = split
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.filepath = os.path.join(root, self.filename)
        self._verify()
        self.data = self._load_data()

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            sample containing image and mask
        """
        image = self._load_image(index)
        label = torch.tensor(self.data[index]['label'])
        magnitude = torch.tensor(self.data[index]['magnitude'])

        sample = {'image': image, 'label': label, 'magnitude': magnitude}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.data)

    def _load_data(self) -> list[dict[str, Any]]:
        """Return the metadata for a given split.

        Returns:
            the sample keys, patches, images, labels, and magnitudes
        """
        h5py = lazy_import('h5py')
        data = []
        with h5py.File(self.filepath) as f:
            for k in sorted(f.keys()):
                if f[k].attrs['split'] != self.splits[self.split]:
                    continue

                for patch in sorted(f[k].keys()):
                    if patch not in ['x', 'y']:
                        # positive sample
                        magnitude = float(f[k].attrs['magnitude'])
                        data.append(
                            dict(
                                key=k,
                                patch=patch,
                                images=('pre', 'post'),
                                label=1,
                                magnitude=magnitude,
                            )
                        )

                        # hard negative sample
                        if 'before' in f[k][patch].keys():
                            data.append(
                                dict(
                                    key=k,
                                    patch=patch,
                                    images=('before', 'pre'),
                                    label=0,
                                    magnitude=0.0,
                                )
                            )
        return data

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        h5py = lazy_import('h5py')

        key = self.data[index]['key']
        patch = self.data[index]['patch']
        images = self.data[index]['images']

        with h5py.File(self.filepath) as f:
            pre_array = f[key][patch][images[0]][:]
            pre_array = np.nan_to_num(pre_array, nan=0)
            post_array = f[key][patch][images[1]][:]
            post_array = np.nan_to_num(post_array, nan=0)
            array = np.concatenate([pre_array, post_array], axis=-1)
            array = array.astype(np.float32)

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
        image = sample['image'].permute((1, 2, 0)).numpy()
        label = cast(int, sample['label'].item())
        label_class = self.classes[label]

        # Create false color image for image1
        vv = percentile_normalization(image[..., 0]) + 1e-16
        vh = percentile_normalization(image[..., 1]) + 1e-16
        fci1 = np.stack([vv, vh, vv / vh], axis=-1).clip(0, 1)

        # Create false color image for image2
        vv = percentile_normalization(image[..., 2]) + 1e-16
        vh = percentile_normalization(image[..., 3]) + 1e-16
        fci2 = np.stack([vv, vh, vv / vh], axis=-1).clip(0, 1)

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = cast(int, sample['prediction'].item())
            prediction_class = self.classes[prediction]

        ncols = 2
        fig, axs = plt.subplots(
            nrows=1, ncols=ncols, figsize=(ncols * 5, 10), sharex=True
        )

        axs[0].imshow(fci1)
        axs[0].axis('off')
        axs[0].set_title('Image Pre')
        axs[1].imshow(fci2)
        axs[1].axis('off')
        axs[1].set_title('Image Post')

        if show_titles:
            title = f'Label: {label_class}'
            if 'magnitude' in sample:
                magnitude = cast(float, sample['magnitude'].item())
                title += f' | Magnitude: {magnitude:.2f}'
            if showing_predictions:
                title += f'\nPrediction: {prediction_class}'
            fig.supxlabel(title, y=0.22)

        if suptitle is not None:
            fig.suptitle(suptitle, y=0.8)

        fig.tight_layout()

        return fig
