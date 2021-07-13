import abc
import glob
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence

import rasterio
import torch
from rasterio.merge import merge

from .geo import GeoDataset
from .utils import BoundingBox


class Sentinel(GeoDataset, abc.ABC):
    """`Sentinel <https://sentinel.esa.int/web/sentinel/home>`_ is a family of
    satellites launched by the `European Space Agency (ESA) <https://www.esa.int/>`_
    under the `Copernicus Programme <https://www.copernicus.eu/en>`_.

    If you use this dataset in your research, please cite it using the following format:

    * https://asf.alaska.edu/data-sets/sar-data-sets/sentinel-1/sentinel-1-how-to-cite/
    """

    # TODO: is this ABC actually needed?
    # Do these datasets actually share anything in common?
    # Could still keep it just to document what Sentinel is and how to cite it...


class Sentinel2(Sentinel):
    """The `Copernicus Sentinel-2 mission
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
        bands: Sequence[str] = band_names,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Sentinel-2 Dataset.

        Parameters:
            root: root directory where dataset can be found
            bands: bands to return
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
        """
        self.root = root
        self.bands = bands
        self.transforms = transforms

        fileglob = os.path.join(root, self.base_folder, f"**_{bands[0]}_*.tif")
        for filename in glob.iglob(fileglob):
            # https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention
            time = datetime.strptime(
                os.path.basename(filename).split("_")[1], "%Y%m%dT%H%M%S"
            )
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
        return {
            "image": torch.tensor(dest),  # type: ignore[attr-defined]
            "transform": out_transform,
        }
