import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import pandas as pd
import rasterio as rio
from torch import Tensor

from torchgeo.datasets.geo import NonGeoDataset

from .utils import Path


class CopernicusBenchBase(NonGeoDataset, ABC):
    """Abstract base class for all Copernicus-Bench datasets.

    .. versionadded:: 0.7
    """

    url = 'https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/86342afa2409e49d80688fe00c05201c0f46569b/{}/{}'

    @property
    @abstractmethod
    def directory(self) -> str:
        """URL directory."""

    @property
    @abstractmethod
    def filename(self) -> str:
        """URL filename."""

    @property
    @abstractmethod
    def checksum(self) -> str:
        """MD5 checksum."""

    @property
    @abstractmethod
    def all_bands(self) -> tuple[str]:
        """All spectral channels."""

    @property
    @abstractmethod
    def rgb_bands(self) -> tuple[str]:
        """Red, green, and blue spectral channels."""

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
            bands: Sequence of band names to load (defauts to all bands).
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).
        """
        self.root = root
        self.split = split
        self.bands = bands or self.all_bands
        self.band_indices = [self.all_bands.index(i) + 1 for i in self.bands]
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.subdir = os.path.join(root, self.filename[: self.filename.index('.')])
        filepath = os.path.join(self.subdir, f'{split}.csv')
        self.files = pd.read_csv(os.path.join(filepath, header=None))[0]

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

    def _load_image(self, index: int) -> dict[str, Tensor]:
        """Load an image.

        Args:
            index: Index to return.

        Returns:
            An image sample.
        """
        sample: dict[str, Tensor] = {}
        with rio.open(self.files[index]) as f:
            sample['image'] = f.read(self.band_indices)

        return sample

    @abstractmethod
    def _load_target(self, index: int) -> dict[str, Tensor]:
        """Load a target label or mask.

        Args:
            index: Index to return.

        Returns:
            A target sample.
        """
