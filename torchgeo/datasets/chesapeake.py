# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Chesapeake Bay High-Resolution Land Cover Project datasets."""

import abc
import os
import pathlib
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

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


class Chesapeake(RasterDataset, abc.ABC):
    """Abstract base class for all Chesapeake datasets.

    `Chesapeake Bay High-Resolution Land Cover Project
    <https://www.chesapeakeconservancy.org/conservation-innovation-center/high-resolution-data/land-cover-data-project/>`_
    dataset.

    This dataset was collected by the Chesapeake Conservancy's Conservation Innovation
    Center (CIC) in partnership with the University of Vermont and WorldView Solutions,
    Inc. It consists of one-meter resolution land cover information for the Chesapeake
    Bay watershed (~100,000 square miles of land).
    """

    is_image = False

    # subclasses use the 13 class cmap by default
    cmap = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (0, 168, 132, 255),
        3: (38, 115, 0, 255),
        4: (76, 230, 0, 255),
        5: (163, 255, 115, 255),
        6: (255, 170, 0, 255),
        7: (255, 0, 0, 255),
        8: (156, 156, 156, 255),
        9: (0, 0, 0, 255),
        10: (115, 115, 0, 255),
        11: (230, 230, 0, 255),
        12: (255, 255, 115, 255),
        13: (197, 0, 255, 255),
    }

    @property
    @abc.abstractmethod
    def base_folder(self) -> str:
        """Parent directory of dataset in URL."""

    @property
    @abc.abstractmethod
    def filename(self) -> str:
        """Filename to find/store dataset in."""

    @property
    @abc.abstractmethod
    def zipfile(self) -> str:
        """Name of zipfile in download URL."""

    @property
    @abc.abstractmethod
    def md5(self) -> str:
        """MD5 checksum to verify integrity of dataset."""

    @property
    def url(self) -> str:
        """URL to download dataset from."""
        url = 'https://cicwebresources.blob.core.windows.net/chesapeakebaylandcover'
        url += f'/{self.base_folder}/{self.zipfile}'
        return url

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
        """Initialize a new Dataset instance.

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
        self.paths = paths
        self.download = download
        self.checksum = checksum

        self._verify()

        colors = []
        for i in range(len(self.cmap)):
            colors.append(
                (
                    self.cmap[i][0] / 255.0,
                    self.cmap[i][1] / 255.0,
                    self.cmap[i][2] / 255.0,
                )
            )
        self._cmap = ListedColormap(colors)

        super().__init__(paths, crs, res, transforms=transforms, cache=cache)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted file already exists
        if self.files:
            return

        # Check if the zip file has already been downloaded
        assert isinstance(self.paths, str | pathlib.Path)
        if os.path.exists(os.path.join(self.paths, self.zipfile)):
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
        download_url(self.url, self.paths, filename=self.zipfile, md5=self.md5)

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str | pathlib.Path)
        extract_archive(os.path.join(self.paths, self.zipfile))

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
        mask = sample['mask'].squeeze(0)
        ncols = 1

        showing_predictions = 'prediction' in sample
        if showing_predictions:
            pred = sample['prediction'].squeeze(0)
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

        if showing_predictions:
            axs[0].imshow(
                mask,
                vmin=0,
                vmax=self._cmap.N - 1,
                cmap=self._cmap,
                interpolation='none',
            )
            axs[0].axis('off')
            axs[1].imshow(
                pred,
                vmin=0,
                vmax=self._cmap.N - 1,
                cmap=self._cmap,
                interpolation='none',
            )
            axs[1].axis('off')
            if show_titles:
                axs[0].set_title('Mask')
                axs[1].set_title('Prediction')

        else:
            axs.imshow(
                mask,
                vmin=0,
                vmax=self._cmap.N - 1,
                cmap=self._cmap,
                interpolation='none',
            )
            axs.axis('off')
            if show_titles:
                axs.set_title('Mask')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class Chesapeake7(Chesapeake):
    """Complete 7-class dataset.

    This version of the dataset is composed of 7 classes:

    0. No Data: Background values
    1. Water: All areas of open water including ponds, rivers, and lakes
    2. Tree Canopy and Shrubs: All woody vegetation including trees and shrubs
    3. Low Vegetation: Plant material less than 2 meters in height including lawns
    4. Barren: Areas devoid of vegetation consisting of natural earthen material
    5. Impervious Surfaces: Human-constructed surfaces less than 2 meters in height
    6. Impervious Roads: Impervious surfaces that are used for transportation
    7. Aberdeen Proving Ground: U.S. Army facility with no labels
    """

    base_folder = 'BAYWIDE'
    filename = 'Baywide_7class_20132014.tif'
    filename_glob = filename
    zipfile = 'Baywide_7Class_20132014.zip'
    md5 = '61a4e948fb2551840b6557ef195c2084'

    cmap = {
        0: (0, 0, 0, 0),
        1: (0, 197, 255, 255),
        2: (38, 115, 0, 255),
        3: (163, 255, 115, 255),
        4: (255, 170, 0, 255),
        5: (156, 156, 156, 255),
        6: (0, 0, 0, 255),
        7: (197, 0, 255, 255),
    }


class Chesapeake13(Chesapeake):
    """Complete 13-class dataset.

    This version of the dataset is composed of 13 classes:

    0. No Data: Background values
    1. Water: All areas of open water including ponds, rivers, and lakes
    2. Wetlands: Low vegetation areas located along marine or estuarine regions
    3. Tree Canopy: Deciduous and evergreen woody vegetation over 3-5 meters in height
    4. Shrubland: Heterogeneous woody vegetation including shrubs and young trees
    5. Low Vegetation: Plant material less than 2 meters in height including lawns
    6. Barren: Areas devoid of vegetation consisting of natural earthen material
    7. Structures: Human-constructed objects made of impervious materials
    8. Impervious Surfaces: Human-constructed surfaces less than 2 meters in height
    9. Impervious Roads: Impervious surfaces that are used for transportation
    10. Tree Canopy over Structures: Tree cover overlapping impervious structures
    11. Tree Canopy over Impervious Surfaces: Tree cover overlapping impervious surfaces
    12. Tree Canopy over Impervious Roads: Tree cover overlapping impervious roads
    13. Aberdeen Proving Ground: U.S. Army facility with no labels
    """

    base_folder = 'BAYWIDE'
    filename = 'Baywide_13Class_20132014.tif'
    filename_glob = filename
    zipfile = 'Baywide_13Class_20132014.zip'
    md5 = '7e51118923c91e80e6e268156d25a4b9'


class ChesapeakeDC(Chesapeake):
    """This subset of the dataset contains data only for Washington, D.C."""

    base_folder = 'DC'
    filename = os.path.join('DC_11001', 'DC_11001.img')
    filename_glob = filename
    zipfile = 'DC_11001.zip'
    md5 = 'ed06ba7570d2955e8857d7d846c53b06'


class ChesapeakeDE(Chesapeake):
    """This subset of the dataset contains data only for Delaware."""

    base_folder = 'DE'
    filename = 'DE_STATEWIDE.tif'
    filename_glob = filename
    zipfile = '_DE_STATEWIDE.zip'
    md5 = '5e12eff3b6950c01092c7e480b38e544'


class ChesapeakeMD(Chesapeake):
    """This subset of the dataset contains data only for Maryland.

    .. note::

       This dataset requires the following additional library to be installed:

       * `zipfile-deflate64 <https://pypi.org/project/zipfile-deflate64/>`_ to extract
         the proprietary deflate64 compressed zip file.
    """

    base_folder = 'MD'
    filename = 'MD_STATEWIDE.tif'
    filename_glob = filename
    zipfile = '_MD_STATEWIDE.zip'
    md5 = '40c7cd697a887f2ffdb601b5c114e567'


class ChesapeakeNY(Chesapeake):
    """This subset of the dataset contains data only for New York.

    .. note::

       This dataset requires the following additional library to be installed:

       * `zipfile-deflate64 <https://pypi.org/project/zipfile-deflate64/>`_ to extract
         the proprietary deflate64 compressed zip file.
    """

    base_folder = 'NY'
    filename = 'NY_STATEWIDE.tif'
    filename_glob = filename
    zipfile = '_NY_STATEWIDE.zip'
    md5 = '1100078c526616454ef2e508affda915'


class ChesapeakePA(Chesapeake):
    """This subset of the dataset contains data only for Pennsylvania."""

    base_folder = 'PA'
    filename = 'PA_STATEWIDE.tif'
    filename_glob = filename
    zipfile = '_PA_STATEWIDE.zip'
    md5 = '20a2a857c527a4dbadd6beed8b47e5ab'


class ChesapeakeVA(Chesapeake):
    """This subset of the dataset contains data only for Virginia.

    .. note::

       This dataset requires the following additional library to be installed:

       * `zipfile-deflate64 <https://pypi.org/project/zipfile-deflate64/>`_ to extract
         the proprietary deflate64 compressed zip file.
    """

    base_folder = 'VA'
    filename = 'CIC2014_VA_STATEWIDE.tif'
    filename_glob = filename
    zipfile = '_VA_STATEWIDE.zip'
    md5 = '6f2c97deaf73bb3e1ea9b21bd7a3fc8e'


class ChesapeakeWV(Chesapeake):
    """This subset of the dataset contains data only for West Virginia."""

    base_folder = 'WV'
    filename = 'WV_STATEWIDE.tif'
    filename_glob = filename
    zipfile = '_WV_STATEWIDE.zip'
    md5 = '350621ea293651fbc557a1c3e3c64cc3'


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

    subdatasets = ['base', 'prior_extension']
    urls = {
        'base': 'https://lilablobssc.blob.core.windows.net/lcmcvpr2019/cvpr_chesapeake_landcover.zip',  # noqa: E501
        'prior_extension': 'https://zenodo.org/record/5866525/files/cvpr_chesapeake_landcover_prior_extension.zip?download=1',  # noqa: E501
    }
    filenames = {
        'base': 'cvpr_chesapeake_landcover.zip',
        'prior_extension': 'cvpr_chesapeake_landcover_prior_extension.zip',
    }
    md5s = {
        'base': '1225ccbb9590e9396875f221e5031514',
        'prior_extension': '402f41d07823c8faf7ea6960d7c4e17a',
    }

    crs = CRS.from_epsg(3857)
    res = 1

    lc_cmap = {
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

    valid_layers = [
        'naip-new',
        'naip-old',
        'landsat-leaf-on',
        'landsat-leaf-off',
        'nlcd',
        'lc',
        'buildings',
        'prior_from_cooccurrences_101_31_no_osm_no_buildings',
    ]
    states = ['de', 'md', 'va', 'wv', 'pa', 'ny']
    splits = (
        [f'{state}-train' for state in states]
        + [f'{state}-val' for state in states]
        + [f'{state}-test' for state in states]
    )

    # these are used to check the integrity of the dataset
    _files = [
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
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_landsat-leaf-off.tif',  # noqa: E501
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_landsat-leaf-on.tif',  # noqa: E501
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_lc.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_naip-new.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_naip-old.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_nlcd.tif',
        'wv_1m_2014_extended-debuffered-val_tiles/m_3708035_ne_17_1_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif',  # noqa: E501
        'spatial_index.geojson',
    ]

    p_src_crs = pyproj.CRS('epsg:3857')
    p_transformers = {
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
                            'prior_from_cooccurrences_101_31_no_osm_no_buildings': prior_fn,  # noqa: E501
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

        sample = {'image': [], 'mask': [], 'crs': self.crs, 'geo_bbox': query}

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
