# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench abstract base class."""

import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import pandas as pd
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from torch import Tensor

from torchgeo.datasets.geo import NonGeoDataset

from ..errors import DatasetNotFoundError, RGBBandsMissingError
from ..utils import (
    Path,
    download_and_extract_archive,
    extract_archive,
    percentile_normalization,
)


class CopernicusBenchBase(NonGeoDataset, ABC):
    """Abstract base class for all Copernicus-Bench datasets.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.11849

    .. versionadded:: 0.7
    """

    @property
    @abstractmethod
    def url(self) -> str:
        """Download URL."""

    @property
    @abstractmethod
    def md5(self) -> str:
        """MD5 checksum."""

    @property
    @abstractmethod
    def zipfile(self) -> str:
        """Zip file name."""

    @property
    @abstractmethod
    def directory(self) -> str:
        """Subdirectory containing split files."""

    @property
    def filename(self) -> str:
        """Filename format of split files."""
        return '{}.csv'

    @property
    @abstractmethod
    def all_bands(self) -> tuple[str, ...]:
        """All spectral channels."""

    @property
    @abstractmethod
    def rgb_bands(self) -> tuple[str, ...]:
        """Red, green, and blue spectral channels."""

    @property
    @abstractmethod
    def cmap(self) -> str | Colormap:
        """Matplotlib color map."""

    @property
    @abstractmethod
    def classes(self) -> tuple[str, ...]:
        """List of classes for classification and semantic segmentation."""

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new CopernicusBenchBase instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'train', 'val', or 'test'.
            bands: Sequence of band names to load (defaults to all bands).
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.split = split
        self.bands = bands or self.all_bands
        self.band_indices = [self.all_bands.index(i) + 1 for i in self.bands]
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        filepath = os.path.join(root, self.directory, self.filename.format(split))
        self.files = pd.read_csv(filepath, header=None)[0]

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        sample = self._load_image(index) | self._load_target(index)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    @abstractmethod
    def _load_image(self, index: int) -> dict[str, Tensor]:
        """Load an image.

        Args:
            index: Index to return.

        Returns:
            An image sample.
        """

    @abstractmethod
    def _load_target(self, index: int) -> dict[str, Tensor]:
        """Load a target label or mask.

        Args:
            index: Index to return.

        Returns:
            A target sample.
        """

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        path = os.path.join(self.root, self.directory, self.filename.format(self.split))
        if os.path.exists(path):
            return

        # Check if the zip file already exists (if so then extract)
        if os.path.exists(os.path.join(self.root, self.zipfile)):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, self.zipfile))

    def _download(self) -> None:
        """Download the dataset."""
        md5 = self.md5 if self.checksum else None
        download_and_extract_archive(self.url, self.root, md5=md5)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        ncols = 1
        if 'mask' in sample:
            ncols += 1
            if 'prediction' in sample:
                ncols += 1

        fig, ax = plt.subplots(ncols=ncols, squeeze=False)

        image = sample['image'][rgb_indices].numpy()
        image = rearrange(image, 'c h w -> h w c')
        image = percentile_normalization(image)
        ax[0, 0].imshow(image)
        ax[0, 0].axis('off')
        if show_titles:
            ax[0, 0].set_title('Image')

        if 'mask' in sample:
            kwargs = {
                'cmap': self.cmap,
                'vmin': 0,
                'vmax': len(self.classes) - 1,
                'interpolation': 'none',
            }
            mask = sample['mask']
            ax[0, 1].imshow(mask, **kwargs)
            ax[0, 1].axis('off')
            if show_titles:
                ax[0, 1].set_title('Mask')

            if 'prediction' in sample:
                prediction = sample['prediction']
                ax[0, 2].imshow(prediction, **kwargs)
                ax[0, 2].axis('off')
                if show_titles:
                    ax[0, 2].set_title('Prediction')

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
