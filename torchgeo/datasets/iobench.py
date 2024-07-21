# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""I/O benchmark dataset."""

import glob
import os
from collections.abc import Callable, Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .cdl import CDL
from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import IntersectionDataset
from .landsat import Landsat9
from .utils import Path, download_url, extract_archive


class IOBench(IntersectionDataset):
    """I/O Bench dataset.

    I/O Bench is a dataset designed to benchmark the I/O performance of TorchGeo.
    It contains a single Landsat 9 scene and CDL file from 2023, and consists of
    the following splits

    * original: the original files as downloaded
      from USGS Earth Explorer and USDA CropScape
    * raw: the same files with compression and with
      CDL clipped to the bounds of the Landsat scene
    * preprocessed: the same files with compression,
      reprojected to the same CRS, as COGs, with TAP

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1145/3557915.3560953

    .. versionadded:: 0.6
    """

    url = 'https://hf.co/datasets/torchgeo/io/resolve/c9d9d268cf0b61335941bdc2b6963bf16fc3a6cf/{}.tar.gz'  # noqa: E501

    md5s = {
        'original': 'e3a908a0fd1c05c1af2f4c65724d59b3',
        'raw': 'e9603990441007ce7bba73bb8ba7d217',
        'preprocessed': '9801f1240b238cb17525c865e413d1fd',
    }

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'preprocessed',
        crs: CRS | None = None,
        res: float | None = None,
        bands: Sequence[str] | None = Landsat9.default_bands + ['SR_QA_AEROSOL'],
        classes: list[int] = [0],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new IOBench instance.

        Args:
            root: Root directory where dataset can be found.
            split: One of 'original', 'raw', or 'preprocessed'.
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: Resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found).
            bands: Bands to return (defaults to all bands).
            classes: List of classes to include, the rest will be mapped to 0.
            transforms: A function/transform that takes an input sample
                and returns a transformed version.
            cache: If True, cache file handle to speed up repeated sampling.
            download: If True, download dataset and store it in the root directory.
            checksum: If True, check the MD5 of the downloaded files (may be slow).

        Raises:
            AssertionError: If *split* argument is invalid.
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert split in self.md5s

        self.root = root
        self.split = split
        self.download = download
        self.checksum = checksum

        self._verify()

        root = os.path.join(root, split)
        self.landsat = Landsat9(root, crs, res, bands, transforms, cache)
        self.cdl = CDL(root, crs, res, [2023], classes, transforms, cache)

        super().__init__(self.landsat, self.cdl)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        count = 0
        for filename_glob in [Landsat9.filename_glob[:6], CDL.filename_glob]:
            pathname = os.path.join(self.root, self.split, '**', filename_glob)
            count += len(glob.glob(pathname, recursive=True))

        if count == 9:
            return

        # Check if the tar files have already been downloaded
        if glob.glob(os.path.join(self.root, f'{self.split}.tar.gz')):
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url.format(self.split),
            self.root,
            md5=self.md5s[self.split] if self.checksum else None,
        )

    def _extract(self) -> None:
        """Extract the dataset."""
        extract_archive(os.path.join(self.root, f'{self.split}.tar.gz'), self.root)

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`IntersectionDataset.__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        rgb_indices = []
        for band in self.landsat.rgb_bands:
            if band in self.landsat.bands:
                rgb_indices.append(self.landsat.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = sample['image'][rgb_indices].permute(1, 2, 0).float()
        mask = sample['mask'].squeeze()

        image = (image - image.min()) / (image.max() - image.min())
        mask = self.cdl.ordinal_cmap[mask]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(image)
        axes[1].imshow(mask, interpolation='none')

        axes[0].axis('off')
        axes[1].axis('off')

        if show_titles:
            axes[0].set_title('Image')
            axes[1].set_title('Mask')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
