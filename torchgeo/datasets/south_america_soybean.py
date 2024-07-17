# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""South America Soybean Dataset."""

from collections.abc import Callable, Iterable
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import download_url


class SouthAmericaSoybean(RasterDataset):
    """South America Soybean Dataset.

    This dataset produced annual 30-m soybean maps of South America from 2001 to 2021.

    Link: https://www.nature.com/articles/s41893-021-00729-z

    Dataset contains 2 classes:

    0. other
    1. soybean

    Dataset Format:

    * 21 .tif files


    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1038/s41893-021-00729-z

    .. versionadded:: 0.6
    """

    filename_glob = 'SouthAmerica_Soybean_*.*'
    filename_regex = r'SouthAmerica_Soybean_(?P<year>\d{4})'

    date_format = '%Y'
    is_image = False
    url = 'https://glad.umd.edu/projects/AnnualClassMapsV1/SouthAmerica_Soybean_{}.tif'

    md5s = {
        2021: 'edff3ada13a1a9910d1fe844d28ae4f',
        2020: '0709dec807f576c9707c8c7e183db31',
        2019: '441836493bbcd5e123cff579a58f5a4f',
        2018: '503c2d0a803c2a2629ebbbd9558a3013',
        2017: '4d0487ac1105d171e5f506f1766ea777',
        2016: '770c558f6ac40550d0e264da5e44b3e',
        2015: '6beb96a61fe0e9ce8c06263e500dde8f',
        2014: '824ff91c62a4ba9f4ccfd281729830e5',
        2013: '0263e19b3cae6fdaba4e3b450cef985e',
        2012: '9f3a71097c9836fcff18a13b9ba608b2',
        2011: 'b73352ebea3d5658959e9044ec526143',
        2010: '9264532d36ffa93493735a6e44caef0d',
        2009: '341387c1bb42a15140c80702e4cca02d',
        2008: '96fc3f737ab3ce9bcd16cbf7761427e2',
        2007: 'bb8549b6674163fe20ffd47ec4ce8903',
        2006: 'eabaa525414ecbff89301d3d5c706f0b',
        2005: '89faae27f9b5afbd06935a465e5fe414',
        2004: 'f9882ca9c70e054e50172835cb75a8c3',
        2003: 'cad5ed461ff4ab45c90177841aaecad2',
        2002: '8a4a9dcea54b3ec7de07657b9f2c0893',
        2001: '2914b0af7590a0ca4dfa9ccefc99020f',
    }

    def __init__(
        self,
        paths: str | Iterable[str] = 'data',
        crs: CRS | None = None,
        res: float | None = None,
        years: list[int] = [2021],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            years: list of years for which to use the South America Soybean layer
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.paths = paths
        self.download = download
        self.checksum = checksum
        self.years = years
        self._verify()

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        if self.files:
            return
        assert isinstance(self.paths, str)

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()

    def _download(self) -> None:
        """Download the dataset."""
        for year in self.years:
            download_url(
                self.url.format(year),
                self.paths,
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
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
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

        axs[0, 0].imshow(mask, interpolation='none')
        axs[0, 0].axis('off')

        if show_titles:
            axs[0, 0].set_title('Mask')

        if showing_predictions:
            axs[0, 1].imshow(pred, interpolation='none')
            axs[0, 1].axis('off')
            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
