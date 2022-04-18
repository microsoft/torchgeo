"""In-memory geographical xarray.DataArray."""

import sys
from typing import Any, Callable, Dict, Optional, cast

import torch
import xarray as xr
from rasterio.crs import CRS
from rtree.index import Index, Property

from .geo import GeoDataset
from .utils import BoundingBox


class RioXarrayDataset(GeoDataset):
    """Wrapper for geographical datasets stored as an xarray.DataArray.

    Relies on rioxarray.
    """

    def __init__(
        self,
        xr_dataarray: xr.DataArray,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            xr_dataarray: n-dimensional xarray.DataArray
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of dataarray)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the dataarray)
            transforms: a function/transform that takes an input sample
                and returns a transformed version

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(transforms)

        self.xr_dataarray = xr_dataarray
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        # Populate the dataset index
        if crs is None:
            crs = xr_dataarray.rio.crs
        if res is None:
            res = xr_dataarray.rio.resolution()[0]

        (minx, miny, maxx, maxy) = xr_dataarray.rio.bounds()
        if hasattr(xr_dataarray, "time"):
            mint = int(xr_dataarray.time.min().data)
            maxt = int(xr_dataarray.time.max().data)
        else:
            mint = 0
            maxt = sys.maxsize
        coords = (minx, maxx, miny, maxy, mint, maxt)
        self.index.insert(0, coords, xr_dataarray.name)

        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        items = [hit.object for hit in hits]

        if not items:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        image = self.xr_dataarray.rio.clip_box(
            minx=query.minx, miny=query.miny, maxx=query.maxx, maxy=query.maxy
        )
        sample = {"image": torch.tensor(image.data), "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
