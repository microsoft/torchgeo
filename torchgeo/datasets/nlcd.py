# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""NLCD dataset."""

import os
from collections.abc import Callable, Iterable
from typing import ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from pyproj import CRS

from .errors import DatasetNotFoundError
from .geo import RasterDataset
from .utils import GeoSlice, Path, Sample, download_url, extract_archive


class NLCD(RasterDataset):
    """Annual National Land Cover Database (NLCD) dataset.

    The `Annual NLCD products
    <https://www.usgs.gov/centers/eros/science/annual-national-land-cover-database>`_
    is an annual land cover product for the conterminous U.S. covering the period
    from 1985 to 2024. The product is a joint effort between the United States Geological Survey
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

    filename_glob = 'Annual_NLCD_LndCov_*_CU_C1V1.tif'
    filename_regex = r'Annual_NLCD_LndCov_(?P<date>\d{4})_CU_C1V1\.tif'
    zipfile_glob = 'Annual_NLCD_LndCov_*_CU_C1V1.zip'
    date_format = '%Y'
    is_image = False

    url = 'https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_LndCov_{}_CU_C1V1.zip'

    md5s: ClassVar[dict[int, str]] = {
        1985: 'f7f743f805ab7ae7936b0800ecbef7db',
        1986: 'd93907e9a5492eeffe7cbf6723ddcc44',
        1987: 'b70033acf5db6e31b2a55d345e6aec5a',
        1988: '9259716eb6271c23318175e0f9b2ff87',
        1989: '38d18b76359a920a5380bc95ae25fd5b',
        1990: '73395f3d7eb407aa7174b2d8b0db3f15',
        1991: 'b5f0b2b7e9a4df2aaf52613111a441f6',
        1992: '8d7fe907c70ff272fcb5fc0ac8c73316',
        1993: '4418f8e4b01461491918575b1a460464',
        1994: '2291e36a8cbe8d7196796e51ceb6cba4',
        1995: '482e2a5c5f8c2fd1640aff7b4e72b3f2',
        1996: 'e811f432d5313951b2d05cf1501e306b',
        1997: '2cab84eaafaea4772bbf826a530dd54e',
        1998: '81ecc39e1bd449cd4d64e3c530f7786c',
        1999: 'a08ef94f6a4e40c48ccec8f582200370',
        2000: '57567a28b5f630aae56975f77f7d6dc1',
        2001: 'ee2b75def58a680fc61a754833ef37c5',
        2002: 'a48508b6bc8baead07c6d7d63168c16a',
        2003: '445eb671d1073c44fe036931b7ab5243',
        2004: '2cb5e27862f1ff338dd5fba0b04e8aa0',
        2005: '2e0649de53505720a28a662d5a936998',
        2006: '59b7a8351d9653e81d14a7b74e8bc0c9',
        2007: '36ba9129c569efa83f32bedba7c4b99a',
        2008: 'c574c8ac75a8474d29313d421e4d0b5b',
        2009: '2e6573190078bd47153bf33014d0cbc1',
        2010: '651107f0236d733e0ebd501862793370',
        2011: '55e95a40118c7c4509e4ab833aa35d80',
        2012: '95e2e3276f5cddf30790a18547c981aa',
        2013: 'e7ffdafbca94de9e2c3d6b160cdac2f4',
        2014: '984e2fb078ae67bdfe4cf81700d36754',
        2015: '2d1b623da4b1a76c5163948b001d3c98',
        2016: '6af6c05ddd87bd1cee7d49c4d08a6f61',
        2017: '46e12333ed2ab0170666344e9e6da406',
        2018: 'fbde066fdbc325de4157c53c2d294117',
        2019: 'cb707a4cfec22c743338282b60d08f79',
        2020: '986cb81ca0483d3ee52c4904682b2c4d',
        2021: '63b859744b5b12ffbd13b9896a587428',
        2022: '68514fedcf928b44fc562d166d938f02',
        2023: '2ac10a23e6a1ccef47b2a8e15ec3ba3c',
        2024: '3e0ded4eb7bb5d355743abe9552b3588',
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
        res: float | tuple[float, float] | None = None,
        years: list[int] = [2024],
        classes: list[int] = list(cmap.keys()),
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS in (xres, yres) format. If a
                single float is provided, it is used for both the x and y resolution.
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

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        sample = super().__getitem__(index)
        sample['mask'] = self.ordinal_map[sample['mask']]
        return sample

    def _verify(self) -> None:
        # Check if the extracted files already exist
        if self.files:
            return

        # Check if the zip files have already been downloaded
        exists = []
        assert isinstance(self.paths, str | os.PathLike)
        for year in self.years:
            pathname = os.path.join(
                self.paths, self.zipfile_glob.replace('*', str(year))
            )
            if os.path.exists(pathname):
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
        assert isinstance(self.paths, str | os.PathLike)
        for year in self.years:
            download_url(
                self.url.format(year),
                self.paths,
                md5=self.md5s[year] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str | os.PathLike)
        for year in self.years:
            zipfile_name = self.zipfile_glob.replace('*', str(year))
            pathname = os.path.join(self.paths, zipfile_name)
            extract_archive(pathname, self.paths)

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
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
