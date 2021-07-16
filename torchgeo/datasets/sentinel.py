"""Sentinel datasets."""

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


class Sentinel(GeoDataset, abc.ABC):
    """Abstract base class for all Sentinel datasets.

    `Sentinel <https://sentinel.esa.int/web/sentinel/home>`_ is a family of
    satellites launched by the `European Space Agency (ESA) <https://www.esa.int/>`_
    under the `Copernicus Programme <https://www.copernicus.eu/en>`_.

    If you use this dataset in your research, please cite it using the following format:

    * https://asf.alaska.edu/data-sets/sar-data-sets/sentinel-1/sentinel-1-how-to-cite/
    """

    # TODO: is this ABC actually needed?
    # Do these datasets actually share anything in common?
    # Could still keep it just to document what Sentinel is and how to cite it...


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

    base_folder = "sentinel"
    band_names = [
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

    def __init__(
        self,
        root: str = "data",
        crs: CRS = CRS.from_epsg(32641),
        bands: Sequence[str] = band_names,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Sentinel-2 Dataset.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to project to
            bands: bands to return
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        self.root = root
        self.crs = crs
        self.bands = bands
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        fileglob = os.path.join(root, self.base_folder, f"**_{bands[0]}_*.tif")
        for i, filename in enumerate(glob.iglob(fileglob)):
            # https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention
            time = datetime.strptime(
                os.path.basename(filename).split("_")[1], "%Y%m%dT%H%M%S"
            )
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
            tokens[2] = band
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
