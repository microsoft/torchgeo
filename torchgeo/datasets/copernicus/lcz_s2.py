# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench LCZ-S2 dataset."""

import os
from collections.abc import Callable, Sequence
from typing import ClassVar, Literal

import torch
from einops import rearrange
from torch import Tensor

from ..errors import DatasetNotFoundError
from ..utils import Path, download_url, lazy_import
from .base import CopernicusBenchBase


class CopernicusBenchLCZS2(CopernicusBenchBase):
    """Copernicus-Bench LCZ-S2 dataset.

    LCZ-S2 is a multi-class scene classification dataset derived from So2Sat-LCZ42,
    a large-scale local climate zone classification dataset.

    If you use this dataset in your research, please cite the following papers:

    * https://arxiv.org/abs/2503.11849
    * https://doi.org/10.1109/MGRS.2020.2964708

    .. note::

       This dataset requires the following additional library to be installed:

       * `<https://pypi.org/project/h5py/>`_ to load the dataset.

    .. versionadded:: 0.7
    """

    url = 'https://hf.co/datasets/wangyi111/Copernicus-Bench/resolve/9d252acd3aa0e3da3128e05c6f028647f0e48e5f/l3_lcz_s2/lcz_{}.h5'
    md5s: ClassVar[dict[str, str]] = {
        'train': 'e0b10cdb7f12e053cda8dd3ff12dbd9e',
        'val': 'be3b503dba5a1405ec6d5a770c2bee33',
        'test': '4e95788c72a421d636f6f8dc7623d116',
    }
    filename = 'lcz_{}.h5'
    all_bands = ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12')
    rgb_bands = ('B04', 'B03', 'B02')
    classes = (
        'Compact high rise',
        'Compact mid rise',
        'Compact low rise',
        'Open high rise',
        'Open mid rise',
        'Open low rise',
        'Lightweight low rise',
        'Large low rise',
        'Sparsely built',
        'Heavy industry',
        'Dense trees',
        'Scattered trees',
        'Bush, scrub',
        'Low plants',
        'Bare rock or paved',
        'Bare soil or sand',
        'Water',
    )

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
        h5py = lazy_import('h5py')

        self.root = root
        self.split = split
        self.bands = bands or self.all_bands
        self.band_indices = [self.all_bands.index(i) for i in self.bands]
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        self._verify()

        self.filepath = os.path.join(root, self.filename.format(split))
        with h5py.File(self.filepath, 'r') as f:
            self.length: int = f['label'].shape[0]

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return self.length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        h5py = lazy_import('h5py')

        with h5py.File(self.filepath, 'r') as f:
            sen2 = f['sen2'][index][:, :, self.band_indices]
            sen2 = rearrange(sen2, 'h w c -> c h w')
            label = f['label'][index].argmax()

        sample = {'image': torch.from_numpy(sen2), 'label': torch.tensor(label)}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if os.path.exists(os.path.join(self.root, self.filename.format(self.split))):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download and extract the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        md5 = self.md5s[self.split] if self.checksum else None
        download_url(self.url.format(self.split), self.root, md5=md5)
