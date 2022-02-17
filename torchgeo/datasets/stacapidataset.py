# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""STACAPIDataset."""

import abc
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import fiona
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer as pc
import stackstac
import torch
from pystac.item import Item
from pystac_client import Client
from rasterio.crs import CRS
from rioxarray.merge import merge_arrays
from rtree.index import Index, Property
from torch import Tensor

# from torch.utils.data import DataLoader
from xarray.core.dataarray import DataArray

from torchgeo.datasets.geo import RasterDataset
from torchgeo.datasets.utils import BoundingBox

# from torchgeo.samplers import RandomGeoSampler


class STACAPIDataset(RasterDataset, abc.ABC):
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
            api_endpoint: api for pystac Client to access
            transforms: a function/transform that takes an input sample
                and returns a transformed versio
            query_parameters: parameters for the catalog to search, for an idea see
            <https://pystac-client.readthedocs.io/en/latest/api.html#pystac_client.ItemSearch>
        """
        self.root = root
        self.api_endpoint = api_endpoint
        self.bands = bands

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        catalog = Client.open(api_endpoint)

        start = time.time()
        search = catalog.search(**query_parameters)

        items = list(search.get_items())

        end = time.time()
        print(
            "Elapsed time Catalog search get items: {:0.4f} seconds".format(end - start)
        )

        if not items:
            raise RuntimeError(
                f"Your search criteria off {query_parameters} did not return any items"
            )

        epsg = items[0].properties["proj:epsg"]
        crs_dict = {"init": "epsg:{}".format(epsg)}
        src_crs = CRS.from_dict(crs_dict)
        if crs is None:
            crs = CRS.from_dict(crs_dict)

        start = time.time()

        for i, item in enumerate(items):
            minx, miny, maxx, maxy = item.bbox

            (minx, maxx), (miny, maxy) = fiona.transform.transform(
                src_crs.to_dict(), crs.to_dict(), [minx, maxx], [miny, maxy]
            )
            mint = 0
            maxt = sys.maxsize
            coords = (minx, maxx, miny, maxy, mint, maxt)
            self.index.insert(i, coords, item)

        end = time.time()
        print("Elapsed time populate index search: {:0.4f} seconds".format(end - start))
        self._crs = crs
        self.res = cast(float, res)
        self.transforms = transforms
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
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        bounds = (query.minx, query.miny, query.maxx, query.maxy)

        raster_list = []
        for item in filepaths:
            raster_list.append(self._snap_to_single_raster(item, bounds))

        # merge single rasters
        data = self._merge_rasters(raster_list)

        # if only single time step then squeeze out time dimenstion
        data = data.squeeze(0)

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _snap_to_single_raster(
        self, item: Item, bounds: Tuple[float, ...]
    ) -> DataArray:
        """Load and merge one or multiple individual bands to one raster.

        Args:
            item: one search item from cataloge
            bounds: (minx, maxx, miny, maxy) coordinates to index

        Returns:
            computed data array from stac
        """
        start = time.time()

        signed_items = pc.sign(item).to_dict()

        aoi = stackstac.stack(
            signed_items, assets=self.bands, bounds_latlon=bounds, resolution=10
        )

        dest = aoi.compute(scheduler="single-threaded")
        end = time.time()
        print("Elapsed time compute AoI: {:0.4f} seconds".format(end - start))
        return dest

    def _merge_rasters(self, raster_list: List[DataArray]) -> Tensor:
        """Merge a list of rasters.

        Args:
            raster_list: list of xarrays

        Returns:
            Tensor of merged xarray data.
        """
        start = time.time()

        time_dim = raster_list[0].shape[0]

        # rioxarray only supports merges for 2D and 3D arrays so merge per time_step
        # and later stack to one tensor
        merged_rasters = []
        for t in range(time_dim):
            rasters_at_t = [r[t, ...] for r in raster_list]
            merged_rasters.append(merge_arrays(rasters_at_t))

        data: "np.typing.NDArray[np.float_]" = np.stack(merged_rasters)
        tensor: Tensor = torch.tensor(data)  # type: ignore[attr-defined]
        end = time.time()
        print("Elapsed time merge rasters: {:0.4f} seconds".format(end - start))
        return tensor

    def plot(  # type: ignore[override]
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

    minx = -148.46876
    maxx = -148.31072
    miny = 61.0491
    maxy = 61.12567489536982
    mint = 0
    maxt = 100000

    bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
    sample = ds[bbox]

    ds.plot(sample)
    import pdb

    pdb.set_trace()

    # tile_size_pix = 40
    # sampler_size = tile_size_pix * ds.res
    # sampler = RandomGeoSampler(ds, size=sampler_size, length=2)
    # dl = DataLoader(ds, sampler=sampler, collate_fn=stack_samples, batch_size=1)

    # for sample in dl:
    #     k = sample["image"]
    #     print(k.shape)
