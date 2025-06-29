# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CaBuAr dataset."""

import os
from collections.abc import Callable
from typing import ClassVar

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, download_url, lazy_import, percentile_normalization


class CaBuAr(NonGeoDataset):
    """CaBuAr dataset.

    `CaBuAr <https://huggingface.co/datasets/DarthReca/california_burned_areas>`__
    is a dataset for Change detection for Burned area Delineation and part of
    the splits are used for the ChaBuD ECML-PKDD 2023 Discovery Challenge.

    Dataset features:

    * Sentinel-2 multispectral imagery
    * binary masks of burned areas
    * 12 multispectral bands
    * 424 pairs of pre and post images with 20 m per pixel resolution (512x512 px)

    Dataset format:

    * single hdf5 dataset containing images and masks

    Dataset classes:

    0. no change
    1. burned area

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/MGRS.2023.3292467

    .. note::

       This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset

    .. versionadded:: 0.6
    """

    all_bands = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B8A',
        'B09',
        'B11',
        'B12',
    )
    rgb_bands = ('B04', 'B03', 'B02')
    folds: ClassVar[dict[str, list[object]]] = {
        'train': [1, 2, 3, 4],
        'val': [0],
        'test': ['chabud'],
    }
    urls = (
        'https://huggingface.co/datasets/DarthReca/california_burned_areas/resolve/main/raw/patched/512x512.hdf5',
        'https://huggingface.co/datasets/DarthReca/california_burned_areas/resolve/main/raw/patched/chabud_test.h5',
    )
    filenames = ('512x512.hdf5', 'chabud_test.h5')
    md5s = ('15d78fb825f9a81dad600db828d22c08', 'a70bb7e4a2788657c2354c4c3d9296fe')

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        bands: tuple[str, ...] = all_bands,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CaBuAr dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", "test"
            bands: the subset of bands to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: If ``split`` or ``bands`` arguments are invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
            DependencyNotFoundError: If h5py is not installed.
        """
        lazy_import('h5py')

        assert split in self.folds
        assert set(bands) <= set(self.all_bands)

        # Set the file index based on the split
        file_index = 1 if split == 'test' else 0

        self.root = root
        self.split = split
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.filepath = os.path.join(root, self.filenames[file_index])
        self.band_indices = [self.all_bands.index(b) for b in bands]

        self._verify()

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

        sample = {'image': image, 'mask': mask}

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
        h5py = lazy_import('h5py')
        uuids = []
        with h5py.File(self.filepath, 'r') as f:
            for k, v in f.items():
                if v.attrs['fold'] in self.folds[self.split] and 'pre_fire' in v.keys():
                    uuids.append(k)
        return sorted(uuids)

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the image
        """
        h5py = lazy_import('h5py')
        uuid = self.uuids[index]
        with h5py.File(self.filepath, 'r') as f:
            pre_array = f[uuid]['pre_fire'][:]
            post_array = f[uuid]['post_fire'][:]

        # index specified bands and concatenate
        pre_array = pre_array[..., self.band_indices]
        post_array = post_array[..., self.band_indices]
        array = np.stack([pre_array, post_array]).astype(np.float32)
        tensor = torch.from_numpy(array)
        tensor = einops.rearrange(tensor, 't h w c -> t c h w')
        return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target mask
        """
        h5py = lazy_import('h5py')
        uuid = self.uuids[index]
        with h5py.File(self.filepath, 'r') as f:
            array = f[uuid]['mask'][:].astype(np.int32).squeeze(axis=-1)

        tensor = torch.from_numpy(array)
        tensor = tensor.to(torch.long)
        return tensor

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        exists = []
        for filename in self.filenames:
            filepath = os.path.join(self.root, filename)
            exists.append(os.path.exists(filepath))

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        for url, filename, md5 in zip(self.urls, self.filenames, self.md5s):
            filepath = os.path.join(self.root, filename)
            if not os.path.exists(filepath):
                download_url(
                    url,
                    self.root,
                    filename=filename,
                    md5=md5 if self.checksum else None,
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

        mask = sample['mask'].numpy()
        image_pre = sample['image'][0][rgb_indices].numpy()
        image_post = sample['image'][1][rgb_indices].numpy()
        image_pre = percentile_normalization(image_pre)
        image_post = percentile_normalization(image_post)

        ncols = 3

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            prediction = sample['prediction']
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(10, ncols * 5))

        axs[0].imshow(einops.rearrange(image_pre, 'c h w -> h w c'))
        axs[0].axis('off')
        axs[1].imshow(einops.rearrange(image_post, 'c h w -> h w c'))
        axs[1].axis('off')
        axs[2].imshow(mask)
        axs[2].axis('off')

        if showing_predictions:
            axs[3].imshow(prediction)
            axs[3].axis('off')

        if show_titles:
            axs[0].set_title('Image Pre')
            axs[1].set_title('Image Post')
            axs[2].set_title('Mask')
            if showing_predictions:
                axs[3].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
