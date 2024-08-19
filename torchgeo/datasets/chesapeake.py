# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Cheasapeake Bay Program Land Use/Land Cover Data Project datasets."""

import glob
import os
import pathlib
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, cast

import fiona
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import rasterio.mask
import shapely.geometry
import shapely.ops
import torch
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from rasterio.crs import CRS
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import GeoDataset, RasterDataset
from .nlcd import NLCD
from .utils import BoundingBox, Path, download_url, extract_archive


class Chesapeake(RasterDataset, ABC):
    """Abstract base class for all Chesapeake datasets.

    `Chesapeake Bay Land Use and Land Cover (LULC) Database 2022 Edition
    <https://www.chesapeakeconservancy.org/conservation-innovation-center/high-resolution-data/lulc-data-project-2022/>`_

    The Chesapeake Bay Land Use and Land Cover Database (LULC) facilitates
    characterization of the landscape and land change for and between discrete time
    periods. The database was developed by the University of Vermont's Spatial Analysis
    Laboratory in cooperation with Chesapeake Conservancy (CC) and U.S. Geological
    Survey (USGS) as part of a 6-year Cooperative Agreement between Chesapeake
    Conservancy and the U.S. Environmental Protection Agency (EPA) and a separate
    Interagency Agreement between the USGS and EPA to provide geospatial support to the
    Chesapeake Bay Program Office.

    The database contains one-meter 13-class Land Cover (LC) and 54-class Land Use/Land
    Cover (LULC) for all counties within or adjacent to the Chesapeake Bay watershed for
    2013/14 and 2017/18, depending on availability of National Agricultural Imagery
    Program (NAIP) imagery for each state. Additionally, 54 LULC classes are generalized
    into 18 LULC classes for ease of visualization and communication of LULC trends. LC
    change between discrete time periods, detected by spectral changes in NAIP imagery
    and LiDAR, represents changes between the 12 land cover classes. LULC change uses LC
    change to identify where changes are happening and then LC is translated to LULC to
    represent transitions between the 54 LULC classes. The LULCC data is represented as
    a LULC class change transition matrix which provides users acres of change between
    multiple classes. It is organized by 18x18 and 54x54 LULC classes. The Chesapeake
    Bay Water (CBW) indicates raster tabulations were performed for only areas that fall
    inside the CBW boundary e.g., if user is interested in CBW portion of a county then
    they will use LULC Matrix CBW. Conversely, if they are interested change transitions
    across the entire county, they will use LULC Matrix.

    If you use this dataset in your research, please cite the following:

    * https://doi.org/10.5066/P981GV1L
    """

    url = 'https://hf.co/datasets/torchgeo/chesapeake/resolve/1e0370eda6a24d93af4153745e54fd383d015bf5/{state}_lulc_{year}_2022-Edition.zip'
    filename_glob = '{state}_lulc_*_2022-Edition.tif'
    filename_regex = r'^{state}_lulc_(?P<date>\d{{4}})_2022-Edition\.tif$'
    date_format = '%Y'
    is_image = False

    @property
    @abstractmethod
    def md5s(self) -> dict[int, str]:
        """Mapping between data year and zip file MD5."""

    @property
    def state(self) -> str:
        """State abbreviation."""
        return self.__class__.__name__[-2:].lower()

    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {
        11: (0, 92, 230, 255),
        12: (0, 92, 230, 255),
        13: (0, 92, 230, 255),
        14: (0, 92, 230, 255),
        15: (0, 92, 230, 255),
        21: (0, 0, 0, 255),
        22: (235, 6, 2, 255),
        23: (89, 89, 89, 255),
        24: (138, 138, 136, 255),
        25: (138, 138, 136, 255),
        26: (138, 138, 136, 255),
        27: (115, 115, 0, 255),
        28: (233, 255, 190, 255),
        29: (255, 255, 115, 255),
        41: (38, 115, 0, 255),
        42: (56, 168, 0, 255),
        51: (255, 255, 115, 255),
        52: (255, 255, 115, 255),
        53: (255, 255, 115, 255),
        54: (170, 255, 0, 255),
        55: (170, 255, 0, 255),
        56: (170, 255, 0, 255),
        62: (77, 209, 148, 255),
        63: (77, 209, 148, 255),
        64: (56, 168, 0, 255),
        65: (38, 115, 0, 255),
        72: (186, 245, 217, 255),
        73: (186, 245, 217, 255),
        74: (56, 168, 0, 255),
        75: (38, 115, 0, 255),
        83: (255, 211, 127, 255),
        84: (255, 211, 127, 255),
        85: (255, 211, 127, 255),
        91: (0, 168, 132, 255),
        92: (0, 168, 132, 255),
        93: (0, 168, 132, 255),
        94: (56, 168, 0, 255),
        95: (38, 115, 0, 255),
        127: (255, 255, 255, 255),
    }

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Chesapeake instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        self.filename_glob = self.filename_glob.format(state=self.state)
        self.filename_regex = self.filename_regex.format(state=self.state)

        self.paths = paths
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted file already exists
        if self.files:
            return

        # Check if the zip file has already been downloaded
        assert isinstance(self.paths, str | pathlib.Path)
        if glob.glob(os.path.join(self.paths, '**', '*.zip'), recursive=True):
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
        for year, md5 in self.md5s.items():
            url = self.url.format(state=self.state, year=year)
            print(url)
            download_url(url, self.paths, md5=md5 if self.checksum else None)

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str | pathlib.Path)
        for file in glob.iglob(os.path.join(self.paths, '**', '*.zip'), recursive=True):
            extract_archive(file)

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
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        cmap = torch.zeros(max(self.cmap) + 1, 4, dtype=torch.uint8)
        for key, value in self.cmap.items():
            cmap[key] = torch.tensor(value)

        mask = sample['mask'].squeeze(0)
        mask = cmap[mask]
        ncols = 1

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            pred = sample['prediction'].squeeze(0)
            pred = cmap[pred]
            ncols = 2

        fig, axs = plt.subplots(ncols=ncols, squeeze=False, figsize=(4 * ncols, 4))

        axs[0, 0].imshow(mask)
        axs[0, 0].axis('off')
        if show_titles:
            axs[0, 0].set_title('Mask')

        if showing_predictions:
            axs[0, 1].imshow(pred)
            axs[0, 1].axis('off')
            if show_titles:
                axs[0, 1].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class ChesapeakeDC(Chesapeake):
    """This subset of the dataset contains data only for Washington, D.C."""

    md5s: ClassVar[dict[int, str]] = {
        2013: '9f1df21afbb9d5c0fcf33af7f6750a7f',
        2017: 'c45e4af2950e1c93ecd47b61af296d9b',
    }


class ChesapeakeDE(Chesapeake):
    """This subset of the dataset contains data only for Delaware."""

    md5s: ClassVar[dict[int, str]] = {
        2013: '5850d96d897babba85610658aeb5951a',
        2018: 'ee94c8efeae423d898677104117bdebc',
    }


class ChesapeakeMD(Chesapeake):
    """This subset of the dataset contains data only for Maryland."""

    md5s: ClassVar[dict[int, str]] = {
        2013: '9c3ca5040668d15284c1bd64b7d6c7a0',
        2018: '0647530edf8bec6e60f82760dcc7db9c',
    }


class ChesapeakeNY(Chesapeake):
    """This subset of the dataset contains data only for New York."""

    md5s: ClassVar[dict[int, str]] = {
        2013: '38a29b721610ba661a7f8b6ec71a48b7',
        2017: '4c1b1a50fd9368cd7b8b12c4d80c63f3',
    }


class ChesapeakePA(Chesapeake):
    """This subset of the dataset contains data only for Pennsylvania."""

    md5s: ClassVar[dict[int, str]] = {
        2013: '86febd603a120a49ef7d23ef486152a3',
        2017: 'b11d92e4471e8cb887c790d488a338c1',
    }


class ChesapeakeVA(Chesapeake):
    """This subset of the dataset contains data only for Virginia."""

    md5s: ClassVar[dict[int, str]] = {
        2014: '49c9700c71854eebd00de24d8488eb7c',
        2018: '51731c8b5632978bfd1df869ea10db5b',
    }


class ChesapeakeWV(Chesapeake):
    """This subset of the dataset contains data only for West Virginia."""

    md5s: ClassVar[dict[int, str]] = {
        2014: '32fea42fae147bd58a83e3ea6cccfb94',
        2018: '80f25dcba72e39685ab33215c5d97292',
    }


class ChesapeakeCVPR(GeoDataset):
    """CVPR 2019 Chesapeake Land Cover dataset.

    The `CVPR 2019 Chesapeake Land Cover
    <https://lila.science/datasets/chesapeakelandcover>`_ dataset contains two layers of
    NAIP aerial imagery, Landsat 8 leaf-on and leaf-off imagery, Chesapeake Bay land
    cover labels, NLCD land cover labels, and Microsoft building footprint labels.

    This dataset was organized to accompany the 2019 CVPR paper, "Large Scale
    High-Resolution Land Cover Mapping with Multi-Resolution Data".

    The paper "Resolving label uncertainty with implicit generative models" added an
    additional layer of data to this dataset containing a prior over the Chesapeake Bay
    land cover classes generated from the NLCD land cover labels. For more information
    about this layer see `the dataset documentation
    <https://zenodo.org/record/5866525>`_.

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/cvpr.2019.01301
    """

    subdatasets = ('base', 'prior_extension')
    urls: ClassVar[dict[str, str]] = {
        'base': 'https://lilablobssc.blob.core.windows.net/lcmcvpr2019/cvpr_chesapeake_landcover.zip',
        'prior_extension': 'https://zenodo.org/record/5866525/files/cvpr_chesapeake_landcover_prior_extension.zip?download=1',
    }
    filenames: ClassVar[dict[str, str]] = {
        'base': 'cvpr_chesapeake_landcover.zip',
        'prior_extension': 'cvpr_chesapeake_landcover_prior_extension.zip',
    }
    md5s: ClassVar[dict[str, str]] = {
        'base': '1225ccbb9590e9396875f221e5031514',
        'prior_extension': '402f41d07823c8faf7ea6960d7c4e17a',
    }

    crs = CRS.from_epsg(3857)
    res = 1

    lc_cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (38, 115, 0, 255),
        3: (163, 255, 115, 255),
        4: (255, 170, 0, 255),
        5: (156, 156, 156, 255),
        6: (0, 0, 0, 255),
        15: (0, 0, 0, 0),
    }

    prior_color_matrix = np.array(
        [
            [0.0, 0.77254902, 1.0, 1.0],
            [0.14901961, 0.45098039, 0.0, 1.0],
            [0.63921569, 1.0, 0.45098039, 1.0],
            [0.61176471, 0.61176471, 0.61176471, 1.0],
        ]
    )

    valid_layers = (
        'naip-new',
        'naip-old',
        'landsat-leaf-on',
        'landsat-leaf-off',
        'nlcd',
        'lc',
        'buildings',
        'prior_from_cooccurrences_101_31_no_osm_no_buildings',
    )
    states = ('de', 'md', 'va', 'wv', 'pa', 'ny')
    splits = (
        [f'{state}-train' for state in states]
        + [f'{state}-val' for state in states]
        + [f'{state}-test' for state in states]
    )

    # these are used to check the integrity of the dataset
    _files = (
        'de_1m_2013_extended-debuffered-test_tiles',
        'de_1m_2013_extended-debuffered-train_tiles',
        'de_1m_2013_extended-debuffered-val_tiles',
        'md_1m_2013_extended-debuffered-test_tiles',
        'md_1m_2013_extended-debuffered-train_tiles',
        'md_1m_2013_extended-debuffered-val_tiles',
        'ny_1m_2013_extended-debuffered-test_tiles',
        'ny_1m_2013_extended-debuffered-train_tiles',
        'ny_1m_2013_extended-debuffered-val_tiles',
        'pa_1m_2013_extended-debuffered-test_tiles',
        'pa_1m_2013_extended-debuffered-train_tiles',
        'pa_1m_2013_extended-debuffered-val_tiles',
        'va_1m_2014_extended-debuffered-test_tiles',
        'va_1m_2014_extended-debuffered-train_tiles',
        'va_1m_2014_extended-debuffered-val_tiles',
        'wv_1m_2014_extended-debuffered-test_tiles',
        'wv_1m_2014_extended-debuffered-train_tiles',
        'wv_1m_2014_extended-debuffered-val_tiles',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_buildings.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_landsat-leaf-off.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_landsat-leaf-on.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_lc.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_naip-new.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_naip-old.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_nlcd.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif',
        'spatial_index.geojson',
    )

    p_src_crs = pyproj.CRS('epsg:3857')
    p_transformers: ClassVar[dict[str, CRS]] = {
        'epsg:26917': pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS('epsg:26917'), always_xy=True
        ).transform,
        'epsg:26918': pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS('epsg:26918'), always_xy=True
        ).transform,
    }

    def __init__(
        self,
        root: Path = 'data',
        splits: Sequence[str] = ['de-train'],
        layers: Sequence[str] = ['naip-new', 'lc'],
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            splits: a list of strings in the format "{state}-{train,val,test}"
                indicating the subset of data to use, for example "ny-train"
            layers: a list containing a subset of "naip-new", "naip-old", "lc", "nlcd",
                "landsat-leaf-on", "landsat-leaf-off", "buildings", or
                "prior_from_cooccurrences_101_31_no_osm_no_buildings" indicating which
                layers to load
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``splits`` or ``layers`` are not valid
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        for split in splits:
            assert split in self.splits
        assert all([layer in self.valid_layers for layer in layers])
        self.root = root
        self.layers = layers
        self.cache = cache
        self.download = download
        self.checksum = checksum

        self._verify()

        super().__init__(transforms)

        lc_colors = np.zeros((max(self.lc_cmap.keys()) + 1, 4))
        lc_colors[list(self.lc_cmap.keys())] = list(self.lc_cmap.values())
        lc_colors = lc_colors[:, :3] / 255
        self._lc_cmap = ListedColormap(lc_colors)

        nlcd_colors = np.zeros((max(NLCD.cmap.keys()) + 1, 4))
        nlcd_colors[list(NLCD.cmap.keys())] = list(NLCD.cmap.values())
        nlcd_colors = nlcd_colors[:, :3] / 255
        self._nlcd_cmap = ListedColormap(nlcd_colors)

        # Add all tiles into the index in epsg:3857 based on the included geojson
        mint: float = 0
        maxt: float = sys.maxsize
        with fiona.open(os.path.join(root, 'spatial_index.geojson'), 'r') as f:
            for i, row in enumerate(f):
                if row['properties']['split'] in splits:
                    box = shapely.geometry.shape(row['geometry'])
                    minx, miny, maxx, maxy = box.bounds
                    coords = (minx, maxx, miny, maxy, mint, maxt)

                    prior_fn = row['properties']['lc'].replace(
                        'lc.tif',
                        'prior_from_cooccurrences_101_31_no_osm_no_buildings.tif',
                    )

                    self.index.insert(
                        i,
                        coords,
                        {
                            'naip-new': row['properties']['naip-new'],
                            'naip-old': row['properties']['naip-old'],
                            'landsat-leaf-on': row['properties']['landsat-leaf-on'],
                            'landsat-leaf-off': row['properties']['landsat-leaf-off'],
                            'lc': row['properties']['lc'],
                            'nlcd': row['properties']['nlcd'],
                            'buildings': row['properties']['buildings'],
                            'prior_from_cooccurrences_101_31_no_osm_no_buildings': prior_fn,
                        },
                    )

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[dict[str, str]], [hit.object for hit in hits])

        sample = {'image': [], 'mask': [], 'crs': self.crs, 'bounds': query}

        if len(filepaths) == 0:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )
        elif len(filepaths) == 1:
            filenames = filepaths[0]
            query_geom_transformed = None  # is set by the first layer

            minx, maxx, miny, maxy, mint, maxt = query
            query_box = shapely.geometry.box(minx, miny, maxx, maxy)

            for layer in self.layers:
                fn = filenames[layer]

                with rasterio.open(os.path.join(self.root, fn)) as f:
                    dst_crs = f.crs.to_string().lower()

                    if query_geom_transformed is None:
                        query_box_transformed = shapely.ops.transform(
                            self.p_transformers[dst_crs], query_box
                        ).envelope
                        query_geom_transformed = shapely.geometry.mapping(
                            query_box_transformed
                        )

                    data, _ = rasterio.mask.mask(
                        f, [query_geom_transformed], crop=True, all_touched=True
                    )

                if layer in [
                    'naip-new',
                    'naip-old',
                    'landsat-leaf-on',
                    'landsat-leaf-off',
                ]:
                    sample['image'].append(data)
                elif layer in [
                    'lc',
                    'nlcd',
                    'buildings',
                    'prior_from_cooccurrences_101_31_no_osm_no_buildings',
                ]:
                    sample['mask'].append(data)
        else:
            raise IndexError(f'query: {query} spans multiple tiles which is not valid')

        sample['image'] = np.concatenate(sample['image'], axis=0)
        sample['mask'] = np.concatenate(sample['mask'], axis=0)

        sample['image'] = torch.from_numpy(sample['image']).float()
        sample['mask'] = torch.from_numpy(sample['mask']).long()

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""

        def exists(filename: Path) -> bool:
            return os.path.exists(os.path.join(self.root, filename))

        # Check if the extracted files already exist
        if all(map(exists, self._files)):
            return

        # Check if the zip files have already been downloaded
        if all(
            [
                os.path.exists(os.path.join(self.root, self.filenames[subdataset]))
                for subdataset in self.subdatasets
            ]
        ):
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
        for subdataset in self.subdatasets:
            download_url(
                self.urls[subdataset],
                self.root,
                filename=self.filenames[subdataset],
                md5=self.md5s[subdataset],
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        for subdataset in self.subdatasets:
            extract_archive(os.path.join(self.root, self.filenames[subdataset]))

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.4
        """
        image = np.rollaxis(sample['image'].numpy(), 0, 3)
        mask = sample['mask'].numpy()
        if mask.ndim == 3:
            mask = np.rollaxis(mask, 0, 3)
        else:
            mask = np.expand_dims(mask, 2)

        num_panels = len(self.layers)
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            predictions = sample['prediction'].numpy()
            num_panels += 1

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))

        i = 0
        for layer in self.layers:
            if layer == 'naip-new' or layer == 'naip-old':
                img = image[:, :, :3] / 255
                image = image[:, :, 4:]
                axs[i].axis('off')
                axs[i].imshow(img)
            elif layer == 'landsat-leaf-on' or layer == 'landsat-leaf-off':
                img = image[:, :, [3, 2, 1]] / 3000
                image = image[:, :, 9:]
                axs[i].axis('off')
                axs[i].imshow(img)
            elif layer == 'nlcd':
                img = mask[:, :, 0]
                mask = mask[:, :, 1:]
                axs[i].imshow(
                    img, vmin=0, vmax=95, cmap=self._nlcd_cmap, interpolation='none'
                )
                axs[i].axis('off')
            elif layer == 'lc':
                img = mask[:, :, 0]
                mask = mask[:, :, 1:]
                axs[i].imshow(
                    img, vmin=0, vmax=15, cmap=self._lc_cmap, interpolation='none'
                )
                axs[i].axis('off')
            elif layer == 'buildings':
                img = mask[:, :, 0]
                mask = mask[:, :, 1:]
                axs[i].imshow(img, vmin=0, vmax=1, cmap='gray', interpolation='none')
                axs[i].axis('off')
            elif layer == 'prior_from_cooccurrences_101_31_no_osm_no_buildings':
                img = (mask[:, :, :4] @ self.prior_color_matrix) / 255
                mask = mask[:, :, 4:]
                axs[i].imshow(img)
                axs[i].axis('off')

            if show_titles:
                if layer == 'prior_from_cooccurrences_101_31_no_osm_no_buildings':
                    axs[i].set_title('prior')
                else:
                    axs[i].set_title(layer)
            i += 1

        if showing_predictions:
            axs[i].imshow(
                predictions, vmin=0, vmax=15, cmap=self._lc_cmap, interpolation='none'
            )
            axs[i].axis('off')
            if show_titles:
                axs[i].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
