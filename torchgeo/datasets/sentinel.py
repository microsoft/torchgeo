# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Sentinel datasets."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .geo import RasterDataset
from .utils import RGBBandsMissingError


class Sentinel(RasterDataset):
    """Abstract base class for all Sentinel datasets.

    `Sentinel <https://sentinel.esa.int/web/sentinel/home>`__ is a family of
    satellites launched by the `European Space Agency (ESA) <https://www.esa.int/>`_
    under the `Copernicus Programme <https://www.copernicus.eu/en>`_.

    If you use this dataset in your research, please cite it using the following format:

    * https://asf.alaska.edu/data-sets/sar-data-sets/sentinel-1/sentinel-1-how-to-cite/
    """


class Sentinel1(Sentinel):
    r"""Sentinel-1 dataset.

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

    .. note::
       Mixing :math:`\gamma_0` and :math:`\sigma_0` backscatter coefficient data is not
       recommended. Similarly, power, decibel, and amplitude scale data should not be
       mixed, and TorchGeo does not attempt to convert all data to a common scale.

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
        _(?P<backscatter>[gs])
        (?P<scale>[pda])
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
        paths: str | list[str] = "data",
        crs: CRS | None = None,
        res: float = 10,
        bands: Sequence[str] = ["VV", "VH"],
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
            bands: bands to return (defaults to ["VV", "VH"])
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            AssertionError: if ``bands`` is invalid
            DatasetNotFoundError: If dataset is not found.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        assert len(bands) > 0, "'bands' cannot be an empty list"
        assert len(bands) == len(set(bands)), "'bands' contains duplicate bands"

        for band in bands:
            assert band in self.all_bands, f"invalid band '{band}'"

        # Sentinel-1 uses dual polarization, it can only transmit either
        # horizontal or vertical at a single time
        msg = """
'bands' cannot contain both horizontal and vertical transmit at the same time.
To create a dataset containing both, use:

    horizontal = Sentinel1(root, bands=["HH", "HV"])
    vertical = Sentinel1(root, bands=["VV", "VH"])
    dataset = horizontal | vertical
"""
        assert len({band[0] for band in bands}) == 1, msg

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
        """
        bands = self.bands

        if len(bands) == 1:
            # Only horizontal or vertical receive, plot as grayscale

            image = sample["image"][0]
            image = torch.clamp(image, min=0, max=1)

            title = f"({bands[0]})"
        else:
            # Both horizontal and vertical receive, plot as RGB

            # Deal with reverse order
            if bands in [["HV", "HH"], ["VH", "VV"]]:
                bands = bands[::-1]
                sample["image"] = torch.flip(sample["image"], dims=[0])

            co_polarization = sample["image"][0]  # transmit == receive
            cross_polarization = sample["image"][1]  # transmit != receive
            ratio = co_polarization / cross_polarization

            # https://gis.stackexchange.com/a/400780/123758
            co_polarization = torch.clamp(co_polarization / 0.3, min=0, max=1)
            cross_polarization = torch.clamp(cross_polarization / 0.05, min=0, max=1)
            ratio = torch.clamp(ratio / 25, min=0, max=1)

            image = torch.stack((co_polarization, cross_polarization, ratio), dim=-1)

            title = "({0}, {1}, {0}/{1})".format(*bands)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")

        if show_titles:
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
    rgb_bands = ["B04", "B03", "B02"]

    separate_files = True

    def __init__(
        self,
        paths: str | Iterable[str] = "data",
        crs: CRS | None = None,
        res: float = 10,
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
            DatasetNotFoundError: If dataset is not found.

        .. versionchanged:: 0.5
            *root* was renamed to *paths*
        """
        bands = bands or self.all_bands
        self.filename_glob = self.filename_glob.format(bands[0])
        self.filename_regex = self.filename_regex.format(res)

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

        image = sample["image"][rgb_indices].permute(1, 2, 0)
        # DN = 10000 * REFLECTANCE
        # https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/
        image = torch.clamp(image / 10000, min=0, max=1)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
