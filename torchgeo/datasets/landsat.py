# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Landsat datasets."""

import abc
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .errors import RGBBandsMissingError
from .geo import RasterDataset
from .utils import Path


class Landsat(RasterDataset, abc.ABC):
    """Abstract base class for all Landsat datasets.

    `Landsat <https://landsat.gsfc.nasa.gov/>`__ is a joint NASA/USGS program,
    providing the longest continuous space-based record of Earth's land in existence.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.usgs.gov/centers/eros/data-citation

    If you use any of the following Level-2 products, there may be additional citation
    requirements, including papers you can cite. See the "Citation Information" section
    of the following pages:

    * `Surface Temperature <https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature>`_
    * `Surface Reflectance <https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance>`_
    * `U.S. Analysis Ready Data <https://www.usgs.gov/landsat-missions/landsat-collection-2-us-analysis-ready-data>`_
    """

    # https://www.usgs.gov/landsat-missions/landsat-collection-2
    filename_regex = r"""
        ^L
        (?P<sensor>[COTEM])
        (?P<satellite>\d{2})
        _(?P<processing_correction_level>[A-Z0-9]{4})
        _(?P<wrs_path>\d{3})
        (?P<wrs_row>\d{3})
        _(?P<date>\d{8})
        _(?P<processing_date>\d{8})
        _(?P<collection_number>\d{2})
        _(?P<collection_category>[A-Z0-9]{2})
        _(?P<band>[A-Z0-9_]+)
        \.
    """

    separate_files = True

    @property
    @abc.abstractmethod
    def default_bands(self) -> tuple[str, ...]:
        """Bands to load by default."""

    # https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
    @property
    @abc.abstractmethod
    def wavelengths(self) -> dict[str, float]:
        """Central wavelength (μm)."""

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        bands = bands or self.default_bands
        self.filename_glob = self.filename_glob.format(bands[0])

        super().__init__(paths, crs, res, bands, transforms, cache)

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

        Raises:
            RGBBandsMissingError: If *bands* does not include all RGB bands.

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise RGBBandsMissingError()

        image = sample['image'][rgb_indices].permute(1, 2, 0).float()

        # Stretch to the full range
        image = (image - image.min()) / (image.max() - image.min())

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis('off')

        if show_titles:
            ax.set_title('Image')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class Landsat1(Landsat):
    """Landsat 1 Multispectral Scanner (MSS)."""

    filename_glob = 'LM01_*_{}.*'

    default_bands = ('B4', 'B5', 'B6', 'B7')
    rgb_bands = ('B6', 'B5', 'B4')

    wavelengths: ClassVar[dict[str, float]] = {
        'B4': (0.5 + 0.6) / 2,
        'B5': (0.6 + 0.7) / 2,
        'B6': (0.7 + 0.8) / 2,
        'B7': (0.8 + 1.1) / 2,
    }


class Landsat2(Landsat1):
    """Landsat 2 Multispectral Scanner (MSS)."""

    filename_glob = 'LM02_*_{}.*'


class Landsat3(Landsat1):
    """Landsat 3 Multispectral Scanner (MSS)."""

    filename_glob = 'LM03_*_{}.*'


class Landsat4MSS(Landsat):
    """Landsat 4 Multispectral Scanner (MSS)."""

    filename_glob = 'LM04_*_{}.*'

    default_bands = ('B1', 'B2', 'B3', 'B4')
    rgb_bands = ('B3', 'B2', 'B1')

    wavelengths: ClassVar[dict[str, float]] = {
        'B1': (0.5 + 0.6) / 2,
        'B2': (0.6 + 0.7) / 2,
        'B3': (0.7 + 0.8) / 2,
        'B4': (0.8 + 1.1) / 2,
    }


class Landsat4TM(Landsat):
    """Landsat 4 Thematic Mapper (TM)."""

    filename_glob = 'LT04_*_{}.*'

    default_bands = ('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7')
    rgb_bands = ('SR_B3', 'SR_B2', 'SR_B1')

    wavelengths: ClassVar[dict[str, float]] = {
        'B1': (0.45 + 0.52) / 2,
        'B2': (0.52 + 0.60) / 2,
        'B3': (0.63 + 0.69) / 2,
        'B4': (0.76 + 0.90) / 2,
        'B5': (1.55 + 1.75) / 2,
        'B6': (10.40 + 12.50) / 2,
        'B7': (2.08 + 2.35) / 2,
    }


class Landsat5MSS(Landsat4MSS):
    """Landsat 4 Multispectral Scanner (MSS)."""

    filename_glob = 'LM04_*_{}.*'


class Landsat5TM(Landsat4TM):
    """Landsat 5 Thematic Mapper (TM)."""

    filename_glob = 'LT05_*_{}.*'


class Landsat7(Landsat):
    """Landsat 7 Enhanced Thematic Mapper Plus (ETM+)."""

    filename_glob = 'LE07_*_{}.*'

    default_bands = ('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7')
    rgb_bands = ('SR_B3', 'SR_B2', 'SR_B1')

    wavelengths: ClassVar[dict[str, float]] = {
        'B1': (0.45 + 0.52) / 2,
        'B2': (0.52 + 0.60) / 2,
        'B3': (0.63 + 0.69) / 2,
        'B4': (0.77 + 0.90) / 2,
        'B5': (1.55 + 1.75) / 2,
        'B6': (10.40 + 12.50) / 2,
        'B7': (2.09 + 2.35) / 2,
        'B8': (0.52 + 0.90) / 2,
    }


class Landsat8(Landsat):
    """Landsat 8 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS)."""

    filename_glob = 'LC08_*_{}.*'

    default_bands = ('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7')
    rgb_bands = ('SR_B4', 'SR_B3', 'SR_B2')

    wavelengths: ClassVar[dict[str, float]] = {
        'B1': (0.43 + 0.45) / 2,
        'B2': (0.45 + 0.51) / 2,
        'B3': (0.53 + 0.59) / 2,
        'B4': (0.64 + 0.67) / 2,
        'B5': (0.85 + 0.88) / 2,
        'B6': (1.57 + 1.65) / 2,
        'B7': (2.11 + 2.29) / 2,
        'B8': (0.50 + 0.68) / 2,
        'B9': (1.36 + 1.38) / 2,
        'B10': (10.6 + 11.19) / 2,
        'B11': (11.50 + 12.51) / 2,
    }


class Landsat9(Landsat8):
    """Landsat 9 Operational Land Imager (OLI-2) and Thermal Infrared Sensor (TIRS-2)."""

    filename_glob = 'LC09_*_{}.*'
