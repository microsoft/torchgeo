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

    The `S2-100k dataset <https://huggingface.co/datasets/davanstrien/satclip>`__
    contains 100,000 256x256 patches of 12 band Sentinel imagery sampled randomly from
    Sentinel 2 scenes on the Microsoft Planetary Computer that have <20% cloud cover,
    intersect land, and were captured between 2021-01-01 and 2023-05-17 (there are
    2,359,972 such scenes).

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1609/aaai.v39i4.32457

    .. versionadded:: 0.10
    """

    url = 'https://huggingface.co/datasets/davanstrien/satclip/resolve/b7043ae65ce4de69fb6f18d88cfa041f6b8fcd7c'

    # index.csv
    sha256 = '9fdcdec776b331fcc2d9ab5af18355efc5bd0716df33ab78e1ff03f60cf343ad'

    # data_*.tar.xz
    sha256s = (
        'c29355d0729c373ab8be3bc9bea915699db1daa6cd86b39e5843f28b9d1421c7',
        '802a294fbff6f7d0d70b5b3a251703793de5f47fa578f0fe92bc9b951d58f7f3',
        '0ed8343b08959ebd5b41519db15d6999833ffb423393888060fa2adef6ad5960',
        '4583ee1414d6ed3c0acf0cfd1958460eabd38348214cdfc93c9e1b76ef22403d',
        '8bea9f49cba4a0339ba6b00a97c40cc4bbd59935115354a6de123d5e52179fb8',
        '7c15f63af8b1e0c65e9f53efe7efab465b676c668f1eceabd039d3dd358cbea7',
        'f28b9d9f97eeb38a45d17250a2d28de99aacb93851fca9aed1cb62fb47c3eb80',
        '1c419e0a44877844c79eaedf227b13c4854f1acddd959ac7109c73627311b39f',
        'bc923a3420623925afb024edcadcb8295ebeecb8337726b05493068951462cde',
        'a3399adc2b99528d2e64ebb74cefef74f70226f9392fa29cd9214c2ffbfcdd8b',
    )

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

        self.df = pd.read_csv(self.root / 'index.csv')

    def __getitem__(self, index: int) -> Sample:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Dictionary with "image" and "point" keys where point is in (lon, lat)
            format.
        """
        row = self.df.iloc[index]

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
        return len(self.df)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # index.csv
        filename = 'index.csv'
        if (self.root / filename).is_file():
            pass
        elif self.download:
            url = f'{self.url}/{filename}'
            sha256 = self.sha256 if self.checksum else None
            download_url(url, self.root, sha256=sha256)
        else:
            raise DatasetNotFoundError(self)

        # data_*.tar.xz
        if (self.root / 'images' / 'patch_0.tif').is_file():
            return

        for i, sha256 in enumerate(self.sha256s, start=1):
            path = self.root / 'images' / f'data_{i}.tar.xz'
            if path.is_file():
                extract_archive(path)
            elif self.download:
                url = f'{self.url}/images/data_{i}.tar.xz'
                sha256 = sha256 if self.checksum else None
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
