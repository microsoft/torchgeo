import abc
import glob
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import rasterio
import torch
from rasterio.merge import merge

from .geo import GeoDataset
from .utils import BoundingBox


class Landsat(GeoDataset, abc.ABC):
    """`Landsat <https://landsat.gsfc.nasa.gov/>`_ is a joint NASA/USGS program,
    providing the longest continuous space-based record of Earth's land in existence.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.usgs.gov/centers/eros/data-citation
    """

    @property
    @abc.abstractmethod
    def base_folder(self) -> str:
        """Subdirectory to find/store dataset in."""

    @property
    @abc.abstractmethod
    def band_names(self) -> Sequence[str]:
        """Spectral bands provided by a satellite.

        See https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites
        for more details.
        """

    def __init__(
        self,
        root: str = "data",
        bands: Sequence[str] = [],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Landsat Dataset.

        Parameters:
            root: root directory where dataset can be found
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        self.root = root
        self.bands = bands if bands else self.band_names
        self.transforms = transforms

        fileglob = os.path.join(root, self.base_folder, f"**_{self.bands[0]}.TIF")
        for filename in glob.iglob(fileglob):
            # https://www.usgs.gov/faqs/what-naming-convention-landsat-collections-level-1-scenes
            # https://www.usgs.gov/faqs/what-naming-convention-landsat-collection-2-level-1-and-level-2-scenes
            time = datetime.strptime(os.path.basename(filename).split("_")[3], "%Y%m%d")
            timestamp = time.timestamp()
            with rasterio.open(filename) as f:
                minx, miny, maxx, maxy = f.bounds
                coords = (minx, maxx, miny, maxy, timestamp, timestamp)
                self.index.insert(0, coords, filename)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Parameters:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data/labels and metadata at that index
        """
        bounds = rasterio.coords.BoundingBox(
            query.minx, query.miny, query.maxx, query.maxy
        )
        hits = self.index.intersection(query, objects=True)
        datasets = [hit.object for hit in hits]
        dest, out_transform = merge(datasets, bounds)
        dest = dest.astype(np.int32)
        return {
            "image": torch.tensor(dest),  # type: ignore[attr-defined]
            "transform": out_transform,
        }


class Landsat8_9(Landsat):
    """Landsat 8-9 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS)."""

    base_folder = "landsat_8_9"
    band_names = [
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


class Landsat7(Landsat):
    """Landsat 7 Enhanced Thematic Mapper Plus (ETM+)."""

    base_folder = "landsat_7"
    band_names = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
    ]


class Landsat4_5TM(Landsat):
    """Landsat 4-5 Thematic Mapper (TM)."""

    base_folder = "landsat_4_5_tm"
    band_names = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
    ]


class Landsat4_5MSS(Landsat):
    """Landsat 4-5 Multispectral Scanner (MSS)."""

    base_folder = "landsat_4_5_mss"
    band_names = [
        "B1",
        "B2",
        "B3",
        "B4",
    ]


class Landsat1_3(Landsat):
    """Landsat 1-3 Multispectral Scanner (MSS)."""

    base_folder = "landsat_1_3"
    band_names = [
        "B4",
        "B5",
        "B6",
        "B7",
    ]
