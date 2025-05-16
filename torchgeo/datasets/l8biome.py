# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""L8 Biome dataset."""

import glob
import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from pyproj import CRS
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import IntersectionDataset, RasterDataset
from .utils import BoundingBox, Path, download_url, extract_archive


class L8BiomeImage(RasterDataset):
    """Images from the L8 Biome dataset."""

    # https://gisgeography.com/landsat-file-naming-convention/
    filename_glob = 'LC8*.TIF'
    filename_regex = r"""
        ^LC8
        (?P<wrs_path>\d{3})
        (?P<wrs_row>\d{3})
        (?P<date>\d{7})
        (?P<gsi>[A-Z]{3})
        (?P<version>\d{2})
        \.TIF$
    """
    date_format = '%Y%j'
    is_image = True
    rgb_bands = ('B4', 'B3', 'B2')
    all_bands = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11')


class L8BiomeMask(RasterDataset):
    """Masks from the L8 Biome dataset."""

    # https://gisgeography.com/landsat-file-naming-convention/
    filename_glob = 'LC8*_fixedmask.TIF'
    filename_regex = r"""
        ^LC8
        (?P<wrs_path>\d{3})
        (?P<wrs_row>\d{3})
        (?P<date>\d{7})
        (?P<gsi>[A-Z]{3})
        (?P<version>\d{2})
        _fixedmask
        \.TIF$
    """
    date_format = '%Y%j'
    is_image = False
    classes = ('Fill', 'Cloud Shadow', 'Clear', 'Thin Cloud', 'Cloud')
    ordinal_map = torch.zeros(256, dtype=torch.long)
    ordinal_map[64] = 1
    ordinal_map[128] = 2
    ordinal_map[192] = 3
    ordinal_map[255] = 4

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
        Returns:
            sample of image, mask and metadata at that index
        Raises:
            IndexError: if query is not found in the index
        """
        sample = super().__getitem__(query)
        sample['mask'] = self.ordinal_map[sample['mask']]
        return sample


class L8Biome(IntersectionDataset):
    """L8 Biome dataset.

    The `L8 Biome <https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data>`__
    dataset is a validation dataset for cloud cover assessment algorithms, consisting
    of Pre-Collection Landsat 8 Operational Land Imager (OLI) Thermal Infrared Sensor
    (TIRS) terrain-corrected (Level-1T) scenes.

    Dataset features:

    * Images evenly divided between 8 unique biomes
    * 96 scenes from Landsat 8 OLI/TIRS sensors
    * Imagery from global tiles between April 2013--October 2014
    * 11 Level-1 spectral bands with 30 m per pixel resolution

    Dataset format:

    * Images are composed of single multiband geotiffs
    * Labels are multiclass, stored in single geotiffs
    * Quality assurance bands, stored in single geotiffs
    * Level-1 metadata (MTL.txt file)
    * Landsat 8 OLI/TIRS bands: (B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11)

    Dataset classes:

    0. Fill
    1. Cloud Shadow
    2. Clear
    3. Thin Cloud
    4. Cloud

    If you use this dataset in your research, please cite the following:

    * https://doi.org/10.5066/F7251GDH
    * https://doi.org/10.1016/j.rse.2017.03.026

    .. versionadded:: 0.5
    """

    url = 'https://hf.co/datasets/torchgeo/l8biome/resolve/f76df19accce34d2acc1878d88b9491bc81f94c8/{}.tar.gz'

    md5s: ClassVar[dict[str, str]] = {
        'barren': '0eb691822d03dabd4f5ea8aadd0b41c3',
        'forest': '4a5645596f6bb8cea44677f746ec676e',
        'grass_crops': 'a69ed5d6cb227c5783f026b9303cdd3c',
        'shrubland': '19df1d0a604faf6aab46d6a7a5e6da6a',
        'snow_ice': 'af8b189996cf3f578e40ee12e1f8d0c9',
        'urban': '5450195ed95ee225934b9827bea1e8b0',
        'water': 'a81153415eb662c9e6812c2a8e38c743',
        'wetlands': '1f86cc354631ca9a50ce54b7cab3f557',
    }

    def __init__(
        self,
        paths: Path | Iterable[Path],
        crs: CRS | None = CRS.from_epsg(3857),
        res: float | tuple[float, float] | None = None,
        bands: Sequence[str] = L8BiomeImage.all_bands,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new L8Biome instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to EPSG:3857)
            res: resolution of the dataset in units of CRS in (xres, yres) format. If a
                single float is provided, it is used for both the x and y resolution.
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.paths = paths
        self.download = download
        self.checksum = checksum

        self._verify()

        self.image = L8BiomeImage(paths, crs, res, bands, transforms, cache)
        self.mask = L8BiomeMask(paths, crs, res, None, transforms, cache)

        super().__init__(self.image, self.mask)

    def _verify(self) -> None:
        """Verify the integrity of the dataset."""
        # Check if the extracted files already exist
        if not isinstance(self.paths, str | os.PathLike):
            return

        for classname in [L8BiomeImage, L8BiomeMask]:
            pathname = os.path.join(self.paths, '**', classname.filename_glob)
            if not glob.glob(pathname, recursive=True):
                break
        else:
            return

        # Check if the tar.gz files have already been downloaded
        pathname = os.path.join(self.paths, '*.tar.gz')
        if glob.glob(pathname):
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
        for biome, md5 in self.md5s.items():
            download_url(
                self.url.format(biome), self.paths, md5=md5 if self.checksum else None
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        assert isinstance(self.paths, str | os.PathLike)
        pathname = os.path.join(self.paths, '*.tar.gz')
        for tarfile in glob.iglob(pathname):
            extract_archive(tarfile)

    def plot(
        self,
        sample: dict[str, Tensor],
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

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.
        """
        rgb_indices = []
        for band in self.image.rgb_bands:
            if band in self.image.bands:
                rgb_indices.append(self.image.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = sample['image'][rgb_indices].permute(1, 2, 0)

        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample['mask'].numpy().astype('uint8').squeeze()

        num_panels = 2
        showing_predictions = 'prediction' in sample
        if showing_predictions:
            predictions = sample['prediction'].numpy().astype('uint8').squeeze()
            num_panels += 1

        kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 4, 'interpolation': 'none'}
        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 4, 5))
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[1].imshow(mask, **kwargs)
        axs[1].axis('off')
        if show_titles:
            axs[0].set_title('Image')
            axs[1].set_title('Mask')

        if showing_predictions:
            axs[2].imshow(predictions, **kwargs)
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Predictions')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
