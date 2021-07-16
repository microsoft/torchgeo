"""Landsat datasets."""

import abc
import glob
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from rtree.index import Index, Property

from .geo import GeoDataset
from .utils import BoundingBox


class Landsat(GeoDataset, abc.ABC):
    """Abstract base class for all Landsat datasets.

    `Landsat <https://landsat.gsfc.nasa.gov/>`_ is a joint NASA/USGS program,
    providing the longest continuous space-based record of Earth's land in existence.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.usgs.gov/centers/eros/data-citation
    """

    @property
    def base_folder(self) -> str:
        """Subdirectory to find/store dataset in."""
        return self.__class__.__name__.lower()

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
        crs: CRS = CRS.from_epsg(32616),
        bands: Sequence[str] = [],
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Landsat Dataset.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to project to
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        self.root = root
        self.crs = crs
        self.bands = bands if bands else self.band_names
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        fileglob = os.path.join(root, self.base_folder, f"**_{self.bands[0]}.TIF")
        for i, filename in enumerate(glob.iglob(fileglob)):
            # https://www.usgs.gov/faqs/what-naming-convention-landsat-collections-level-1-scenes
            # https://www.usgs.gov/faqs/what-naming-convention-landsat-collection-2-level-1-and-level-2-scenes
            time = datetime.strptime(os.path.basename(filename).split("_")[3], "%Y%m%d")
            timestamp = time.timestamp()
            with rasterio.open(filename) as src:
                with WarpedVRT(src, crs=self.crs) as vrt:
                    minx, miny, maxx, maxy = vrt.bounds
            coords = (minx, maxx, miny, maxy, timestamp, timestamp)
            self.index.insert(i, coords, filename)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data/labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} is not within bounds of the index: {self.bounds}"
            )

        hits = self.index.intersection(query, objects=True)
        filename = next(hits).object  # TODO: this assumes there is only a single hit
        data_list = []
        for band in self.bands:
            tokens = filename.split("_")
            tokens[-1] = band + ".TIF"
            filename = "_".join(tokens)
            with rasterio.open(filename) as src:
                with WarpedVRT(src, crs=self.crs) as vrt:
                    col_off = (query.minx - vrt.bounds.left) // vrt.res[0]
                    row_off = (query.miny - vrt.bounds.bottom) // vrt.res[1]
                    width = query.maxx - query.minx
                    height = query.maxy - query.miny
                    window = Window(col_off, row_off, width, height)
                    image = vrt.read(window=window)
            data_list.append(image)
        image = np.concatenate(data_list)  # type: ignore[no-untyped-call]
        image = image.astype(np.int32)
        return {
            "image": torch.tensor(image),  # type: ignore[attr-defined]
            "crs": self.crs,
        }


class Landsat8(Landsat):
    """Landsat 8-9 Operational Land Imager (OLI) and Thermal Infrared Sensor (TIRS)."""

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


Landsat9 = Landsat8


class Landsat7(Landsat):
    """Landsat 7 Enhanced Thematic Mapper Plus (ETM+)."""

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


class Landsat4TM(Landsat):
    """Landsat 4-5 Thematic Mapper (TM)."""

    band_names = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
    ]


Landsat5TM = Landsat4TM


class Landsat4MSS(Landsat):
    """Landsat 4-5 Multispectral Scanner (MSS)."""

    band_names = [
        "B1",
        "B2",
        "B3",
        "B4",
    ]


Landsat5MSS = Landsat4MSS


class Landsat1(Landsat):
    """Landsat 1-3 Multispectral Scanner (MSS)."""

    band_names = [
        "B4",
        "B5",
        "B6",
        "B7",
    ]


Landsat2 = Landsat1
Landsat3 = Landsat1
