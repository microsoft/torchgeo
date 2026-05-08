# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

# Adapted from https://github.com/microsoft/satclip.
# Copyright (c) Microsoft Corporation.

"""S2-100k pre-training dataset from SatCLIP paper."""

import pathlib
from collections.abc import Callable
from typing import Literal

import einops
import pandas as pd
import rasterio as rio
import torch
from matplotlib import pyplot as plt

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import (
    Path,
    Sample,
    array_to_tensor,
    download_and_extract_archive,
    download_url,
    extract_archive,
)


class S2100k(NonGeoDataset):
    """S2-100K dataset.

    The `S2-100k dataset <https://hf.co/datasets/kklmmr/s2-100k>`__
    contains 100,000 256x256 patches of 12 band Sentinel imagery sampled randomly from
    Sentinel 2 scenes on the Microsoft Planetary Computer that have <20% cloud cover,
    intersect land, and were captured between 2021-01-01 and 2023-05-17 (there are
    2,359,972 such scenes).

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1609/aaai.v39i4.32457

    .. versionadded:: 0.10
    """

    url = 'https://hf.co/datasets/kklmmr/s2-100k/resolve/fbdbb78ba57d22d5b6be203913f1e6020c2b4e0a'
    index_sha256 = '9fdcdec776b331fcc2d9ab5af18355efc5bd0716df33ab78e1ff03f60cf343ad'
    data_sha256 = 'da1bae4e9dd44fb00e5f1fc537b752f5025ac908a73b5a9e24ff90bcbdd56edb'

    def __init__(
        self,
        root: Path = 'data',
        *,
        mode: Literal['both', 'points'] = 'both',
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new S2100K dataset instance.

        Args:
            root: Root directory where dataset can be found.
            mode: Which data to return (options are "both" or "points"), useful for
                embedding locations without loading images.
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            AssertionError: If *mode* argument is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert mode in {'both', 'points'}

        self.root = pathlib.Path(root)
        self.transforms = transforms
        self.mode = mode
        self.download = download
        self.checksum = checksum

        self._verify()

        self.index = pd.read_csv(self.root / 'index.csv')

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Dictionary with "image" and "point" keys where point is in (lon, lat)
            format.
        """
        row = self.index.iloc[index]

        point = torch.tensor([row['lon'], row['lat']])
        sample = {'point': point}

        if self.mode == 'both':
            with rio.open(self.root / 'images' / row['fn']) as f:
                sample['image'] = array_to_tensor(f.read()).float()

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.index)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Index
        filename = 'index.csv'
        if (self.root / filename).is_file():
            pass
        elif self.download:
            url = f'{self.url}/{filename}'
            sha256 = self.index_sha256 if self.checksum else None
            download_url(url, self.root, sha256=sha256)
        else:
            raise DatasetNotFoundError(self)

        # Data
        if (self.root / 'images' / 'patch_0.tif').is_file():
            return

        path = self.root / 'satclip.tar'
        if path.is_file():
            extract_archive(path, self.root / 'images')
        elif self.download:
            url = f'{self.url}/satclip.tar'
            sha256 = self.data_sha256 if self.checksum else None
            download_and_extract_archive(url, self.root / 'images', sha256=sha256)
        else:
            raise DatasetNotFoundError(self)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        image = sample['image'][[3, 2, 1]]
        image = einops.rearrange(image, 'c h w -> h w c')
        image = torch.clamp(image / 4000, 0, 1)

        fig, ax = plt.subplots(figsize=(4, 4))

        ax.imshow(image)
        ax.axis('off')

        if show_titles:
            ax.set_title(f'({sample["point"][0]:0.4f}, {sample["point"][1]:0.4f})')

        if suptitle is not None:
            fig.suptitle(suptitle)

        return fig
