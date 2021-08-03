"""Canadian Building Footprints dataset."""

import glob
import os
import sys
from typing import Any, Callable, Dict, Optional

import fiona
import fiona.transform
import matplotlib.pyplot as plt
import rasterio
import torch
from rasterio.crs import CRS
from rtree.index import Index, Property
from torch import Tensor

from .geo import GeoDataset
from .utils import BoundingBox, check_integrity, download_and_extract_archive

_crs = CRS.from_epsg(4326)


class CanadianBuildingFootprints(GeoDataset):
    """Canadian Building Footprints dataset.

    The `Canadian Building Footprints
    <https://github.com/Microsoft/CanadianBuildingFootprints>`_ dataset contains
    11,842,186 computer generated building footprints in all Canadian provinces and
    territories in GeoJSON format. This data is freely available for download and use.
    """

    # TODO: how does one cite this dataset?
    # https://github.com/microsoft/CanadianBuildingFootprints/issues/11

    url = "https://usbuildingdata.blob.core.windows.net/canadian-buildings-v2/"
    provinces_territories = [
        "Alberta",
        "BritishColumbia",
        "Manitoba",
        "NewBrunswick",
        "NewfoundlandAndLabrador",
        "NorthwestTerritories",
        "NovaScotia",
        "Nunavut",
        "Ontario",
        "PrinceEdwardIsland",
        "Quebec",
        "Saskatchewan",
        "YukonTerritory",
    ]
    md5s = [
        "8b4190424e57bb0902bd8ecb95a9235b",
        "fea05d6eb0006710729c675de63db839",
        "adf11187362624d68f9c69aaa693c46f",
        "44269d4ec89521735389ef9752ee8642",
        "65dd92b1f3f5f7222ae5edfad616d266",
        "346d70a682b95b451b81b47f660fd0e2",
        "bd57cb1a7822d72610215fca20a12602",
        "c1f29b73cdff9a6a9dd7d086b31ef2cf",
        "76ba4b7059c5717989ce34977cad42b2",
        "2e4a3fa47b3558503e61572c59ac5963",
        "9ff4417ae00354d39a0cf193c8df592c",
        "a51078d8e60082c7d3a3818240da6dd5",
        "c11f3bd914ecabd7cac2cb2871ec0261",
    ]

    def __init__(
        self,
        root: str = "data",
        crs: CRS = _crs,
        res: float = 1,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Canadian Building Footprints dataset.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to project to
            res: resolution to use when rasterizing features
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            RuntimeError: if ``download=False`` and data is not found, or
                ``checksum=True`` and checksums don't match
        """
        self.root = root
        self.crs = crs
        self.res = res
        self.transforms = transforms
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        fileglob = os.path.join(root, "**.geojson")
        for i, filename in enumerate(glob.iglob(fileglob, recursive=True)):
            with fiona.open(filename) as src:
                minx, miny, maxx, maxy = src.bounds
                (minx, maxx), (miny, maxy) = fiona.transform.transform(
                    src.crs, crs.to_dict(), [minx, maxx], [miny, maxy]
                )
            mint = 0
            maxt = sys.maxsize
            coords = (minx, maxx, miny, maxy, mint, maxt)
            self.index.insert(i, coords, filename)

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} is not within bounds of the index: {self.bounds}"
            )

        hits = self.index.intersection(query, objects=True)
        filename = next(hits).object  # TODO: this assumes there is only a single hit
        shapes = []
        with fiona.open(filename) as src:
            # We need to know the bounding box of the query in the source CRS
            (minx, maxx), (miny, maxy) = fiona.transform.transform(
                self.crs.to_dict(),
                src.crs,
                [query.minx, query.maxx],
                [query.miny, query.maxy],
            )

            # Filter geometries to those that intersect with the bounding box
            for feature in src.filter((minx, miny, maxx, maxy)):
                # Warp geometries to requested CRS
                shape = fiona.transform.transform_geom(
                    src.crs, self.crs.to_dict(), feature["geometry"]
                )
                shapes.append(shape)

        # Rasterize geometries
        width = (query.maxx - query.minx) / self.res
        height = (query.maxy - query.miny) / self.res
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )
        masks = rasterio.features.rasterize(shapes, transform=transform)

        # Clip to bounding box
        rows, cols = rasterio.transform.rowcol(
            transform, [query.minx, query.maxx], [query.miny, query.maxy]
        )
        masks = masks[rows[0] : rows[1], cols[0] : cols[1]]

        sample = {
            "masks": torch.tensor(masks),  # type: ignore[attr-defined]
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        for prov_terr, md5 in zip(self.provinces_territories, self.md5s):
            filepath = os.path.join(self.root, prov_terr + ".zip")
            if not check_integrity(filepath, md5 if self.checksum else None):
                return False
        return True

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        for prov_terr, md5 in zip(self.provinces_territories, self.md5s):
            download_and_extract_archive(
                self.url + prov_terr + ".zip",
                self.root,
                md5=md5 if self.checksum else None,
            )

    def plot(self, image: Tensor) -> None:
        """Plot an image on a map.

        Args:
            image: the image to plot
        """
        array = image.squeeze().numpy()

        # Plot the image
        ax = plt.axes()
        ax.imshow(array)
        ax.axis("off")
        plt.show()
        plt.close()
