"""National Agriculture Imagery Program (NAIP) dataset."""

import glob
import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional

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

_crs = CRS.from_epsg(26918)


class NAIP(GeoDataset):
    """National Agriculture Imagery Program (NAIP) dataset.

    The `National Agriculture Imagery Program (NAIP)
    <https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/>`_
    acquires aerial imagery during the agricultural growing seasons in the continental
    U.S. A primary goal of the NAIP program is to make digital ortho photography
    available to governmental agencies and the public within a year of acquisition.

    NAIP is administered by the USDA's Farm Service Agency (FSA) through the Aerial
    Photography Field Office in Salt Lake City. This "leaf-on" imagery is used as a base
    layer for GIS programs in FSA's County Service Centers, and is used to maintain the
    Common Land Unit (CLU) boundaries.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.fisheries.noaa.gov/inport/item/49508/citation
    """

    def __init__(
        self,
        root: str,
        crs: CRS = _crs,
        transforms: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Initialize a new NAIP dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to project to
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        self.root = root
        self.crs = crs
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        fileglob = os.path.join(root, "**.tif")
        for i, filename in enumerate(glob.iglob(fileglob, recursive=True)):
            with rasterio.open(filename) as src:
                with WarpedVRT(src, crs=self.crs) as vrt:
                    minx, miny, maxx, maxy = vrt.bounds
            # https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcs141p2_015644.pdf
            date = filename.split("_")[-1].replace(".tif", "")
            time = datetime.strptime(date, "%Y%m%d")
            timestamp = time.timestamp()
            coords = (minx, maxx, miny, maxy, timestamp, timestamp)
            self.index.insert(i, coords, filename)

        if "filename" not in locals():
            raise FileNotFoundError(f"No NAIP data was found in '{root}'")

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} is not within bounds of the index: {self.bounds}"
            )

        hits = self.index.intersection(query, objects=True)
        filename = next(hits).object
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
        ax.imshow(array, origin="lower")
        ax.axis("off")
        plt.show()
