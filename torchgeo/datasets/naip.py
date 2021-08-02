"""National Agriculture Imagery Program (NAIP) dataset."""

import glob
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
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

    # https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcs141p2_015644.pdf
    # https://planetarycomputer.microsoft.com/dataset/naip#Storage-Documentation
    filename_glob = "m_*.tif"
    filename_regex = re.compile(
        r"""
        ^m
        _(?P<quadrangle>\d+)
        _(?P<quarter_quad>[a-z]+)
        _(?P<utm_zone>\d+)
        _(?P<resolution>\d+)
        _(?P<acquisition_date>\d+)
        (?:_(?P<processing_date>\d+))?
        .tif$
    """,
        re.VERBOSE,
    )
    date_format = "%Y%m%d"

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
        fileglob = os.path.join(root, "**", self.filename_glob)
        for i, filename in enumerate(glob.iglob(fileglob, recursive=True)):
            match = re.match(self.filename_regex, os.path.basename(filename))
            if match is not None:
                with rasterio.open(filename) as src:
                    with WarpedVRT(src, crs=self.crs) as vrt:
                        minx, miny, maxx, maxy = vrt.bounds
                date = match.group("acquisition_date")
                time = datetime.strptime(date, self.date_format)
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
            with WarpedVRT(src, crs=self.crs, nodata=0) as vrt:
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
        # Drop NIR channel
        image = image[:3]

        # Convert from CxHxW to HxWxC
        image = image.permute((1, 2, 0))
        array = image.numpy()

        # Plot the image
        ax = plt.axes()
        ax.imshow(array, origin="lower")
        ax.axis("off")
        plt.show()
