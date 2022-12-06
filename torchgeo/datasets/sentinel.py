# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Sentinel datasets."""

from typing import Any, Callable, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import torch
from rasterio.crs import CRS

from .geo import RasterDataset


class Sentinel(RasterDataset):
    """Abstract base class for all Sentinel datasets.

    `Sentinel <https://sentinel.esa.int/web/sentinel/home>`__ is a family of
    satellites launched by the `European Space Agency (ESA) <https://www.esa.int/>`_
    under the `Copernicus Programme <https://www.copernicus.eu/en>`_.

    If you use this dataset in your research, please cite it using the following format:

    * https://asf.alaska.edu/data-sets/sar-data-sets/sentinel-1/sentinel-1-how-to-cite/
    """


class Sentinel1(Sentinel):
    """Sentinel-1 dataset.

    The `Sentinel-1 mission
    <https://sentinel.esa.int/web/sentinel/missions/sentinel-1>`_ comprises a
    constellation of two polar-orbiting satellites, operating day and night
    performing C-band synthetic aperture radar imaging, enabling them to
    acquire imagery regardless of the weather.

    Data can be downloaded from:

    * `Copernicus Open Access Hub
      <https://scihub.copernicus.eu/>`_
    * `Alaska Satellite Facility (ASF) Distributed Active Archive Center (DAAC)
      <https://asf.alaska.edu/>`_
    * `Microsoft's Planetary Computer
      <https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc>`_

    Product Types:

    * `Level-0
      <https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/product-types-processing-levels/level-0>`_:
      Raw (RAW)
    * `Level-1
      <https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/product-types-processing-levels/level-1>`_:
      Single Look Complex (SLC)
    * `Level-1
      <https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/product-types-processing-levels/level-1>`_:
      Ground Range Detected (GRD)
    * `Level-2
      <https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/product-types-processing-levels/level-2>`_:
      Ocean (OCN)

    Polarizations:

    * HH: horizontal transmit, horizontal receive
    * HV: horizontal transmit, vertical receive
    * VV: vertical transmit, vertical receive
    * VH: vertical transmit, horizontal receive

    Acquisition Modes:

    * `Stripmap (SM)
      <https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/stripmap>`_
    * `Interferometric Wide (IW) swath
      <https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/interferometric-wide-swath>`_
    * `Extra Wide (EW) swatch
      <https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/extra-wide-swath>`_
    * `Wave (WV)
      <https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/wave>`_

    .. note::
       At the moment, this dataset only supports the GRD product type. Data must be
       radiometrically terrain corrected (RTC). This can be done manually using a DEM,
       or you can download an On Demand RTC product from ASF DAAC.

    .. versionadded:: 0.4
    """

    # SAFE format
    # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
    # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/document-library/-/asset_publisher/1dO7RF5fJMbd/content/sentinel-1-product-specification

    # ASF DAAC GRD RTC uses a different naming scheme (README.md.txt in download):
    #
    # S1x_yy_aaaaaaaaTbbbbbb_ppo_RTCzz_u_defklm_ssss
    #
    # x:          Sentinel-1 Mission (A or B)
    # yy:         Beam Mode
    # aaaaaaaa:   Start Date of Acquisition (YYYYMMDD)
    # bbbbbb:     Start Time of Acquisition (HHMMSS)
    # pp:         Polarization: Dual-pol (D) vs. Single-pol (S),
    #                 primary polarization (H vs. V)
    # o:          Orbit Type: Precise (P), Restituted (R), or Original Predicted (O)
    # zz:         Terrain Correction Pixel Spacing
    # u:          Software Package Used: GAMMA (G)
    # d:          Gamma-0 (g) or Sigma-0 (s) Output
    # e:          Power (p) or Decibel (d) or Amplitude (a) Output
    # f:          Unmasked (u) or Water Masked (w)
    # k:          Not Filtered (n) or Filtered (f)
    # l:          Entire Area (e) or Clipped Area (c)
    # m:          Dead Reckoning (d) or DEM Matching (m)
    # ssss:       Product ID
    filename_glob = "S1*{}.*"
    filename_regex = r"""
        ^S1(?P<mission>[AB])
        _(?P<mode>SM|IW|EW|WV)
        _(?P<date>\d{8}T\d{6})
        _(?P<polarization>[DS][HV])
        (?P<orbit>[PRO])
        _RTC(?P<spacing>\d{2})
        _(?P<package>G)
        _(?P<output1>[gs])
        (?P<output2>[pda])
        (?P<mask>[uw])
        (?P<filter>[nf])
        (?P<area>[ec])
        (?P<matching>[dm])
        _(?P<product>[0-9A-Z]{4})
        _(?P<band>[VH]{2})
        \.
    """
    date_format = "%Y%m%dT%H%M%S"
    all_bands = ["HH", "HV", "VV", "VH"]
    separate_files = True

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: float = 10,
        bands: Sequence[str] = ["VV", "VH"],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            AssertionError: if ``bands`` is invalid
            FileNotFoundError: if no files are found in ``root``
        """
        # Sentinel-1 uses dual polarization, it can only transmit either
        # horizontal or vertical at a single time
        assert list(bands) == ["HH", "HV"] or list(bands) == ["VV", "VH"]

        self.filename_glob = self.filename_glob.format(bands[0])

        super().__init__(root, crs, res, bands, transforms, cache)

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        # Either HH/HV or VV/VH, doesn't really matter which
        hh = sample["image"][0]
        hv = sample["image"][1]

        # https://gis.stackexchange.com/a/400780/123758
        hh = torch.clamp(hh / 0.3, min=0, max=1)
        hv = torch.clamp(hv / 0.05, min=0, max=1)
        hh_hv = torch.clamp(hh / hv / 25, min=0, max=1)

        image = torch.stack((hh, hv, hh_hv), dim=-1)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            title = "({0}, {1}, {0}/{1})".format(*self.bands)
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


class Sentinel2(Sentinel):
    """Sentinel-2 dataset.

    The `Copernicus Sentinel-2 mission
    <https://sentinel.esa.int/web/sentinel/missions/sentinel-2>`_ comprises a
    constellation of two polar-orbiting satellites placed in the same sun-synchronous
    orbit, phased at 180° to each other. It aims at monitoring variability in land
    surface conditions, and its wide swath width (290 km) and high revisit time (10 days
    at the equator with one satellite, and 5 days with 2 satellites under cloud-free
    conditions which results in 2-3 days at mid-latitudes) will support monitoring of
    Earth's surface changes.
    """

    # https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/naming-convention
    # https://sentinel.esa.int/documents/247904/685211/Sentinel-2-MSI-L2A-Product-Format-Specifications.pdf
    filename_glob = "T*_*_{}*.*"
    filename_regex = r"""
        ^T(?P<tile>\d{{2}}[A-Z]{{3}})
        _(?P<date>\d{{8}}T\d{{6}})
        _(?P<band>B[018][\dA])
        (?:_(?P<resolution>{}m))?
        \..*$
    """
    date_format = "%Y%m%dT%H%M%S"

    # https://gisgeography.com/sentinel-2-bands-combinations/
    all_bands = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]
    RGB_BANDS = ["B04", "B03", "B02"]

    separate_files = True

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: float = 10,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        bands = bands or self.all_bands
        self.filename_glob = self.filename_glob.format(bands[0])
        self.filename_regex = self.filename_regex.format(res)

        super().__init__(root, crs, res, bands, transforms, cache)

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            ValueError: if the RGB bands are not included in ``self.bands``

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        rgb_indices = []
        for band in self.RGB_BANDS:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image / 2000, min=0, max=1)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
