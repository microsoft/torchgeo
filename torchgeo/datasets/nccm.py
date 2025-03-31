# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Northeastern China Crop Map Dataset."""

from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import BoundingBox, Path, download_url


class NCCM(RasterDataset):
    """The Northeastern China Crop Map Dataset.

    Link: https://www.nature.com/articles/s41597-021-00827-9

    This dataset produced annual 10-m crop maps of the
    major crops (maize, soybean, and rice)
    in Northeast China from 2017 to 2019, using hierarchial mapping strategies,
    random forest classifiers, interpolated and
    smoothed 10-day Sentinel-2 time series data and
    optimized features from spectral, temporal and
    textural characteristics of the land surface.
    The resultant maps have high overall accuracies (OA)
    based on ground truth data. The dataset contains information
    specific to three years: 2017, 2018, 2019.

    The dataset contains 5 classes:

    0. paddy rice
    1. maize
    2. soybean
    3. others crops and lands
    4. nodata

    Dataset format:

    * Three .TIF files containing the labels
    * JavaScript code to download images from the dataset.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1038/s41597-021-00827-9

    .. versionadded:: 0.6
    """

    filename_regex = r'CDL(?P<date>\d{4})_clip'
    filename_glob = 'CDL*.*'

    date_format = '%Y'
    is_image = False
    urls: ClassVar[dict[int, str]] = {
        2019: 'https://figshare.com/ndownloader/files/25070540',
        2018: 'https://figshare.com/ndownloader/files/25070624',
        2017: 'https://figshare.com/ndownloader/files/25070582',
    }
    md5s: ClassVar[dict[int, str]] = {
        2019: '0d062bbd42e483fdc8239d22dba7020f',
        2018: 'b3bb4894478d10786aa798fb11693ec1',
        2017: 'd047fbe4a85341fa6248fd7e0badab6c',
    }
    fnames: ClassVar[dict[int, str]] = {
        2019: 'CDL2019_clip.tif',
        2018: 'CDL2018_clip1.tif',
        2017: 'CDL2017_clip.tif',
    }

    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {
        0: (0, 255, 0, 255),
        1: (255, 0, 0, 255),
        2: (255, 255, 0, 255),
        3: (128, 128, 128, 255),
        15: (255, 255, 255, 255),
    }

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        years: list[int] = [2019],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new dataset.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres) format. If a
                single float is provided, it is used for both the x and y resolution.
                (defaults to the resolution of the first file found)
            years: list of years for which to use nccm layers
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert set(years) <= self.md5s.keys(), (
            'NCCM data product only exists for the following years: '
            f'{list(self.md5s.keys())}.'
        )
        self.paths = paths
        self.years = years
        self.download = download
        self.checksum = checksum
        self.ordinal_map = torch.full((max(self.cmap.keys()) + 1,), 4, dtype=self.dtype)
        self.ordinal_cmap = torch.zeros((5, 4), dtype=torch.uint8)

        self._verify()
        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

        for i, (k, v) in enumerate(self.cmap.items()):
            self.ordinal_map[k] = i
            self.ordinal_cmap[i] = torch.tensor(v)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        sample = super().__getitem__(query)
        sample['mask'] = self.ordinal_map[sample['mask']]
        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the files already exist
        if self.files:
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        for year in self.years:
            download_url(
                self.urls[year],
                self.paths,
                filename=self.fnames[year],
                md5=self.md5s[year] if self.checksum else None,
            )

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`NCCM.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        mask = sample['mask'].squeeze()
        ncols = 1

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            pred = sample['prediction'].squeeze()
            ncols = 2

        fig, axs = plt.subplots(
            nrows=1, ncols=ncols, figsize=(ncols * 4, 4), squeeze=False
        )

        axs[0, 0].imshow(self.ordinal_cmap[mask], interpolation='none')
        axs[0, 0].axis('off')

        if show_titles:
            axs[0, 0].set_title('Mask')

        if showing_predictions:
            axs[0, 1].imshow(self.ordinal_cmap[pred], interpolation='none')
            axs[0, 1].axis('off')
            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
