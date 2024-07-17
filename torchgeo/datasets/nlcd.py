# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NLCD dataset."""

import glob
import os
from collections.abc import Callable, Iterable
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import BoundingBox, download_url, extract_archive


class NLCD(RasterDataset):
    """National Land Cover Database (NLCD) dataset.

    The `NLCD dataset
    <https://www.usgs.gov/centers/eros/science/national-land-cover-database>`_
    is a land cover product that covers the United States and Puerto Rico. The current
    implementation supports maps for the continental United States only. The product is
    a joint effort between the United States Geological Survey
    (`USGS <https://www.usgs.gov/>`_) and the Multi-Resolution Land Characteristics
    Consortium (`MRLC <https://www.mrlc.gov/>`_) which released the first product
    in 2001 with new updates every five years since then.

    The dataset contains the following 17 classes:

    0. Background
    #. Open Water
    #. Perennial Ice/Snow
    #. Developed, Open Space
    #. Developed, Low Intensity
    #. Developed, Medium Intensity
    #. Developed, High Intensity
    #. Barren Land (Rock/Sand/Clay)
    #. Deciduous Forest
    #. Evergreen Forest
    #. Mixed Forest
    #. Shrub/Scrub
    #. Grassland/Herbaceous
    #. Pasture/Hay
    #. Cultivated Crops
    #. Woody Wetlands
    #. Emergent Herbaceous Wetlands

    Detailed descriptions of the classes can be found
    `here <https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description>`__.

    Dataset format:

    * single channel .img file with integer class labels

    If you use this dataset in your research, please use the corresponding citation:

    * 2001: https://doi.org/10.5066/P9MZGHLF
    * 2006: https://doi.org/10.5066/P9HBR9V3
    * 2011: https://doi.org/10.5066/P97S2IID
    * 2016: https://doi.org/10.5066/P96HHBIE
    * 2019: https://doi.org/10.5066/P9KZCM54

    .. versionadded:: 0.5
    """  # noqa: E501

    filename_glob = 'nlcd_*_land_cover_l48_*.img'
    filename_regex = (
        r'nlcd_(?P<date>\d{4})_land_cover_l48_(?P<publication_date>\d{8})\.img'
    )
    zipfile_glob = 'nlcd_*_land_cover_l48_*.zip'
    date_format = '%Y'
    is_image = False

    url = 'https://s3-us-west-2.amazonaws.com/mrlc/nlcd_{}_land_cover_l48_20210604.zip'

    md5s = {
        2001: '538166a4d783204764e3df3b221fc4cd',
        2006: '67454e7874a00294adb9442374d0c309',
        2011: 'ea524c835d173658eeb6fa3c8e6b917b',
        2016: '452726f6e3bd3f70d8ca2476723d238a',
        2019: '82851c3f8105763b01c83b4a9e6f3961',
    }

    cmap = {
        0: (0, 0, 0, 0),
        11: (70, 107, 159, 255),
        12: (209, 222, 248, 255),
        21: (222, 197, 197, 255),
        22: (217, 146, 130, 255),
        23: (235, 0, 0, 255),
        24: (171, 0, 0, 255),
        31: (179, 172, 159, 255),
        41: (104, 171, 95, 255),
        42: (28, 95, 44, 255),
        43: (181, 197, 143, 255),
        52: (204, 184, 121, 255),
        71: (223, 223, 194, 255),
        81: (220, 217, 57, 255),
        82: (171, 108, 40, 255),
        90: (184, 217, 235, 255),
        95: (108, 159, 184, 255),
    }

    def __init__(
        self,
        paths: str | Iterable[str] = 'data',
        crs: CRS | None = None,
        res: float | None = None,
        years: list[int] = [2019],
        classes: list[int] = list(cmap.keys()),
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
            years: list of years for which to use nlcd layer
            classes: list of classes to include, the rest will be mapped to 0
                (defaults to all classes)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 after downloading files (may be slow)

        Raises:
            AssertionError: if ``years`` or ``classes`` are invalid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        assert set(years) <= self.md5s.keys(), (
            'NLCD data product only exists for the following years: '
            f'{list(self.md5s.keys())}.'
        )
        assert (
            set(classes) <= self.cmap.keys()
        ), f'Only the following classes are valid: {list(self.cmap.keys())}.'
        assert 0 in classes, 'Classes must include the background class: 0'

        self.paths = paths
        self.years = years
        self.classes = classes
        self.download = download
        self.checksum = checksum
        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=self.dtype)
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)

        self._verify()

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])

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
        # Check if the extracted files already exist
        if self.files:
            return

        # Check if the zip files have already been downloaded
        exists = []
        for year in self.years:
            zipfile_year = self.zipfile_glob.replace('*', str(year), 1)
            assert isinstance(self.paths, str)
            pathname = os.path.join(self.paths, '**', zipfile_year)
            if glob.glob(pathname, recursive=True):
                exists.append(True)
                self._extract()
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise DatasetNotFoundError(self)

        # Download the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        for year in self.years:
            download_url(
                self.url.format(year),
                self.paths,
                md5=self.md5s[year] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        for year in self.years:
            zipfile_name = self.zipfile_glob.replace('*', str(year), 1)
            assert isinstance(self.paths, str)
            pathname = os.path.join(self.paths, '**', zipfile_name)
            extract_archive(glob.glob(pathname, recursive=True)[0], self.paths)

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
