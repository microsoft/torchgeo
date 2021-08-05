"""Landsat datasets."""

import abc
from typing import Any, Callable, Dict, List, Optional, Sequence

from rasterio.crs import CRS

from .geo import RasterDataset


class Landsat(RasterDataset, abc.ABC):
    """Abstract base class for all Landsat datasets.

    `Landsat <https://landsat.gsfc.nasa.gov/>`_ is a joint NASA/USGS program,
    providing the longest continuous space-based record of Earth's land in existence.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.usgs.gov/centers/eros/data-citation
    """

    # https://www.usgs.gov/faqs/what-naming-convention-landsat-collections-level-1-scenes
    # https://www.usgs.gov/faqs/what-naming-convention-landsat-collection-2-level-1-and-level-2-scenes
    filename_glob = ""
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
        _SR
        _(?P<band>[A-Z0-9]{2})
        \..*$
    """

    # https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
    all_bands: List[str] = []
    rgb_bands: List[str] = []

    separate_files = True
    stretch = True

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        bands: Sequence[str] = [],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to project to. Uses the CRS
                of the files by default
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        self.bands = bands if bands else self.all_bands

        super().__init__(root, crs, transforms)


class Landsat1(Landsat):
    """Landsat 1 Multispectral Scanner (MSS)."""

    filename_glob = "LM01_*.*"

    all_bands = [
        "B4",
        "B5",
        "B6",
        "B7",
    ]
    rgb_bands = ["B6", "B5", "B4"]


class Landsat2(Landsat1):
    """Landsat 2 Multispectral Scanner (MSS)."""

    filename_glob = "LM02_*.*"


class Landsat3(Landsat1):
    """Landsat 3 Multispectral Scanner (MSS)."""

    filename_glob = "LM03_*.*"


class Landsat4MSS(Landsat):
    """Landsat 4 Multispectral Scanner (MSS)."""

    filename_glob = "LM04_*.*"

    all_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
    ]
    rgb_bands = ["B3", "B2", "B1"]


class Landsat4TM(Landsat):
    """Landsat 4 Thematic Mapper (TM)."""

    filename_glob = "LT04_*.*"

    all_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
    ]
    rgb_bands = ["B3", "B2", "B1"]


class Landsat5MSS(Landsat4MSS):
    """Landsat 4 Multispectral Scanner (MSS)."""

    filename_glob = "LM04_*.*"


class Landsat5TM(Landsat4TM):
    """Landsat 5 Thematic Mapper (TM)."""

    filename_glob = "LT05_*.*"


class Landsat7(Landsat):
    """Landsat 7 Enhanced Thematic Mapper Plus (ETM+)."""

    filename_glob = "LE07_*.*"

    all_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
    ]
    rgb_bands = ["B3", "B2", "B1"]


class Landsat8(Landsat):
    """Landsat 8 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS)."""

    filename_glob = "LC08_*.*"

    all_bands = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B9",
        "B10",
        "B11",
    ]
    rgb_bands = ["B4", "B3", "B2"]


class Landsat9(Landsat8):
    """Landsat 9 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS)."""

    filename_glob = "LC09_*.*"
