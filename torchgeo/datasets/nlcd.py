# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""NLCD dataset."""

import os
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import BoundingBox, Path, download_url


class NLCD(RasterDataset):
    """Annual National Land Cover Database (NLCD) dataset.

    The `Annual NLCD products
    <https://www.usgs.gov/centers/eros/science/annual-national-land-cover-database>`_
    is an annual land cover product for the conterminous U.S. initially covering the period
    from 1985 to 2023. The product is a joint effort between the United States Geological Survey
    (`USGS <https://www.usgs.gov/>`_) and the Multi-Resolution Land Characteristics
    Consortium (`MRLC <https://www.mrlc.gov/>`_).

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

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.5066/P94UXNTS

    .. versionadded:: 0.5
    """

    filename_glob = 'Annual_NLCD_LndCov_*_CU_C1V0.tif'
    filename_regex = r'Annual_NLCD_LndCov_(?P<date>\d{4})_CU_C1V0\.tif'
    date_format = '%Y'
    is_image = False

    url = 'https://s3-us-west-2.amazonaws.com/mrlc/Annual_NLCD_LndCov_{}_CU_C1V0.tif'

    md5s: ClassVar[dict[int, str]] = {
        1985: 'a2e1c5f0b34e9b15a63a9dc10e8d3ec2',
        1986: 'da1d08ca51ac43abc14711c8d6139f1d',
        1987: '2cb85e8f077c227605cd7bac62a72a75',
        1988: 'b20fb987cc30926d2d125d045e02626d',
        1989: 'dbe851cbea34d0a57c2a94eb745a1267',
        1990: '1927e0e040b9ff513ff039749b64919b',
        1991: 'eca73474843d6c58693eba62d70e507c',
        1992: '8beda41ba79000f55a8e9358ba3fa5a4',
        1993: '1a023552967cdac1111e9968ea62c879',
        1994: 'acc30ce4f6cdd78af5f7887d17ac4de3',
        1995: 'f728e8fc231b2e8e74a14201f500543a',
        1996: 'd2580904244f89b20d6258150fbf4161',
        1997: 'fec4e08032e162f2cc7dbe019d042609',
        1998: '87ea19434de96ea99cd5d7991042816c',
        1999: 'd4133737f20e75f3bd3a5baa32a668da',
        2000: 'e20b61bb2e7f4034a33c9fd536798a01',
        2001: 'b1f46ace9aedd17a89efab489cb67bc3',
        2002: '57bf60d7cd473096af3bb125391bde63',
        2003: '5e346854da9abf739152e85fee4c7aff',
        2004: '13136f271f53a454358eb7ec12bda686',
        2005: 'f00b66b57a23eb49a077e88704964a91',
        2006: '074ba90de5e62a37a5f001b7572f6baa',
        2007: 'cdef29a191cf165baaae80857ce5a980',
        2008: 'da907c76a1f12739333148504fd111c9',
        2009: '47890b306b875e681990b3db0c709da3',
        2010: '9a81f405f9e2f45d581078afd53c2d4b',
        2011: '13f4ef40b204aa1108dc0599d9546701',
        2012: '66b33146f9a9d9491be10c59c51e3e33',
        2013: 'f8d230f7dea493c47fbc74984ff856cc',
        2014: '68eb07ce86c1f7c2546ec43c2f9f7029',
        2015: 'f5a1b59fe54a70752f544c06cb965be4',
        2016: 'f0c2e74824fc281a57821e28e2c7fe6e',
        2017: 'a0aa8be0ed7d637f0f88f26d3742b20e',
        2018: 'a01f31547837ff1dfec1aba07b89bbec',
        2019: 'fa738201cddc1393dac4383b6ce2561a',
        2020: 'aa8f51690c7b01f3b3b413be9a7c36d6',
        2021: '47fc1794a64704a918b6ad586df4267c',
        2022: '11359748229e138cde971947864104a4',
        2023: '498ff8a512d32fe905720796fdb7fd52',
    }

    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {
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
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: tuple[float, float] | None = None,
        years: list[int] = [2023],
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
        assert set(classes) <= self.cmap.keys(), (
            f'Only the following classes are valid: {list(self.cmap.keys())}.'
        )
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
        # Check if the TIFF files for the specified years have already been downloaded
        exists = []
        for year in self.years:
            filename_year = self.filename_glob.replace('*', str(year), 1)
            assert isinstance(self.paths, str | os.PathLike)
            pathname = os.path.join(self.paths, filename_year)
            if os.path.exists(pathname):
                exists.append(True)
            else:
                exists.append(False)

        if all(exists):
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
