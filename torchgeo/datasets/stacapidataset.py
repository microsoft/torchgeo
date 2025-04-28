# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""STACAPIDataset."""

import sys
from typing import Any, Callable, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import planetary_computer as pc
import stackstac
import torch
from pyproj import Transformer
from pystac_client import Client
from rasterio.crs import CRS
from torch import Tensor

from torchgeo.datasets.geo import GeoDataset
from torchgeo.datasets.utils import BoundingBox


class STACAPIDataset(GeoDataset):
    """STACApiDataset.

    SpatioTemporal Asset Catalogs (`STACs <https://stacspec.org/>`_) are a way
    to organize geospatial datasets. STAC APIs let you query huge STAC Catalogs by
    date, time, and other metadata.


    .. versionadded:: 0.3
    """

    sentinel_bands = [
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
        "B11",
        "B12",
    ]

    def __init__(  # type: ignore[no-untyped-def]
        self,
        root: str,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Sequence[str] = sentinel_bands,
        is_image: bool = True,
        api_endpoint: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        **query_parameters,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: sequence of of stac asset band names
            is_image: if true, :meth:`__getitem__` uses `image` as sample key, `mask`
                otherwise
            api_endpoint: api for pystac Client to access
            transforms: a function/transform that takes an input sample
                and returns a transformed versio
            query_parameters: parameters for the catalog to search, for an idea see
            <https://pystac-client.readthedocs.io/en/latest/api.html#pystac_client.ItemSearch>
        """
        self.root = root
        self.api_endpoint = api_endpoint
        self.bands = bands
        self.is_image = is_image

        super().__init__(transforms)

        catalog = Client.open(api_endpoint)

        search = catalog.search(**query_parameters)

        items = list(search.get_items())

        if not items:
            raise RuntimeError(
                f"No items returned from search criteria: {query_parameters}"
            )

        epsg = items[0].properties["proj:epsg"]
        src_crs = CRS.from_epsg(epsg)
        if crs is None:
            crs = src_crs

        for i, item in enumerate(items):
            minx, miny, maxx, maxy = item.bbox

            transformer = Transformer.from_crs(4326, crs.to_epsg(), always_xy=True)
            (minx, maxx), (miny, maxy) = transformer.transform(
                [minx, maxx], [miny, maxy]
            )
            mint = 0
            maxt = sys.maxsize
            coords = (minx, maxx, miny, maxy, mint, maxt)
            self.index.insert(i, coords, item)

        self._crs = crs
        self.res = res
        self.items = items

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

        # suggested #
        signed_items = [pc.sign(item).to_dict() for item in items]

        stack = stackstac.stack(
            signed_items,
            assets=self.bands,
            resolution=self.res,
            epsg=self._crs.to_epsg(),
        )

        aoi = stack.loc[
            ..., query.maxy : query.miny, query.minx : query.maxx  # type: ignore[misc]
        ]

        data = aoi.compute(scheduler="single-threaded").data

        # handle time dimension here
        image: Tensor = torch.Tensor(data)

        key = "image" if self.is_image else "mask"
        sample = {key: image, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self,
        sample: Dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        image = sample["image"].permute(1, 2, 0)
        image = torch.clip(image / 10000, 0, 1)  # type: ignore[attr-defined]

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")

        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig


if __name__ == "__main__":

    area_of_interest = {
        "type": "Polygon",
        "coordinates": [
            [
                [-148.56536865234375, 60.80072385643073],
                [-147.44338989257812, 60.80072385643073],
                [-147.44338989257812, 61.18363894915102],
                [-148.56536865234375, 61.18363894915102],
                [-148.56536865234375, 60.80072385643073],
            ]
        ],
    }

    time_of_interest = "2019-06-01/2019-08-01"

    collections = (["sentinel-2-l2a"],)
    intersects = (area_of_interest,)
    datetime = (time_of_interest,)
    query = ({"eo:cloud_cover": {"lt": 10}},)

    rgb_bands = ["B04", "B03", "B02"]
    ds = STACAPIDataset(
        "./data",
        bands=rgb_bands,
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 10}},
    )

    minx = 420688.14962388354
    maxx = 429392.15007465985
    miny = 6769145.954634559
    maxy = 6777492.989499866
    mint = 0
    maxt = 100000

    bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
    sample = ds[bbox]
