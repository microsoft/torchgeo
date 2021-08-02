"""Landsat datasets."""

import abc
import glob
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rtree.index import Index, Property
from torch import Tensor

from .geo import GeoDataset
from .utils import BoundingBox

_crs = CRS.from_epsg(32616)


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
        crs: CRS = _crs,
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

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        self.root = root
        self.crs = crs
        self.bands = bands if bands else self.band_names
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        path = os.path.join(root, self.base_folder)
        fileglob = os.path.join(path, f"**_{self.bands[0]}.TIF")
        for i, filename in enumerate(glob.iglob(fileglob, recursive=True)):
            # https://www.usgs.gov/faqs/what-naming-convention-landsat-collections-level-1-scenes
            # https://www.usgs.gov/faqs/what-naming-convention-landsat-collection-2-level-1-and-level-2-scenes
            time = datetime.strptime(os.path.basename(filename).split("_")[3], "%Y%m%d")
            timestamp = time.timestamp()
            with rasterio.open(filename) as src:
                with WarpedVRT(src, crs=self.crs) as vrt:
                    minx, miny, maxx, maxy = vrt.bounds
            coords = (minx, maxx, miny, maxy, timestamp, timestamp)
            self.index.insert(i, coords, filename)

        if "filename" not in locals():
            raise FileNotFoundError(f"No Landsat data was found in '{path}'")

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data and metadata at that index

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
                    window = rasterio.windows.from_bounds(
                        query.minx,
                        query.miny,
                        query.maxx,
                        query.maxy,
                        transform=vrt.transform,
                    )
                    image = vrt.read(window=window)
            data_list.append(image)
        image = np.concatenate(data_list)  # type: ignore[no-untyped-call]
        image = image.astype(np.int32)
        sample = {
            "image": torch.tensor(image),  # type: ignore[attr-defined]
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(self, image: Tensor) -> None:
        """Plot an image on a map.

        Args:
            image: the image to plot
        """
        # Convert from CxHxW to HxWxC
        image = image.permute((1, 2, 0))
        array = image.numpy()

        # Stretch to the range of 2nd to 98th percentile
        per98 = np.percentile(array, 98)  # type: ignore[no-untyped-call]
        per02 = np.percentile(array, 2)  # type: ignore[no-untyped-call]
        array = (array - per02) / (per98 - per02)
        array = np.clip(array, 0, 1)

        # Plot the image
        ax = plt.axes()
        ax.imshow(array)
        ax.axis("off")
        plt.show()


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
