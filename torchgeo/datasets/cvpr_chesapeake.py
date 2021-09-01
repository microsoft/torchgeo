# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CVPR 2019 Chesapeake Land Cover dataset."""

import os
import sys
from typing import Any, Callable, Dict, Optional

import fiona
import pyproj
import rasterio
import rasterio.mask
import shapely.geometry
import shapely.ops
from rasterio.crs import CRS

from .geo import GeoDataset
from .utils import BoundingBox, check_integrity, download_and_extract_archive


class CVPRChesapeake(GeoDataset):
    """CVPR 2019 Chesapeake Land Cover dataset.

    The `CVPR 2019 Chesapeake Land Cover
    <https://lila.science/datasets/chesapeakelandcover>`_ dataset contains two layers of
    NAIP aerial imagery, Landsat 8 leaf-on and leaf-off imagery, Chesapeake Bay land
    cover labels, NLCD land cover labels, and Microsoft building footprint labels.

    This dataset was organized to accompany the 2019 CVPR paper, "Large Scale
    High-Resolution Land Cover Mapping with Multi-Resolution Data".

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/cvpr.2019.01301
    """

    url = "TODO"
    filename = "TODO"
    md5 = "TODO"

    crs = CRS.from_epsg(3857)
    res = 1

    states = ["de", "md", "va", "wv", "pa", "ny"]
    splits = (
        [f"{state}-train" for state in states]
        + [f"{state}-val" for state in states]
        + [f"{state}-test" for state in states]
    )

    p_src_crs = pyproj.CRS("epsg:3857")
    p_transformers = {
        "epsg:26917": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26917"), always_xy=True
        ).transform,
        "epsg:26918": pyproj.Transformer.from_crs(
            p_src_crs, pyproj.CRS("epsg:26918"), always_xy=True
        ).transform,
    }

    def __init__(
        self,
        root: str = "data",
        split: str = "de-train",
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: a string in the format "{state}-{train,val,test}" indicating the
                subset of data to use
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        assert split in self.splits
        super().__init__(transforms)  # creates self.index and self.transform
        self.root = root
        self.cache = cache
        self.checksum = checksum

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                + "You can use download=True to download it"
            )

        # Add all tiles into the index in epsg:3857 based on the included geojson
        mint: float = 0
        maxt: float = sys.maxsize
        with fiona.open(os.path.join(root, "spatial_index.geojson"), "r") as f:
            for i, row in enumerate(f):
                if row["properties"]["split"] == split:
                    box = shapely.geometry.shape(row["geometry"])
                    minx, miny, maxx, maxy = box.bounds
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(
                        i,
                        coords,
                        {
                            "naip-new": row["properties"]["naip-new"],
                            "naip-old": row["properties"]["naip-old"],
                            "landsat-leaf-on": row["properties"]["landsat-leaf-on"],
                            "landsat-leaf-off": row["properties"]["landsat-leaf-off"],
                            "lc": row["properties"]["lc"],
                            "nlcd": row["properties"]["nlcd"],
                            "buildings": row["properties"]["buildings"],
                        },
                    )

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(query, objects=True)
        filepaths = [hit.object for hit in hits]

        if len(filepaths) == 0:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        elif len(filepaths) == 1:
            filenames = filepaths[0]
            naip_fn = filenames["naip-new"]
            lc_fn = filenames["lc"]

            minx, maxx, miny, maxy, mint, maxt = query
            query_box = shapely.geometry.box(minx, miny, maxx, maxy)

            with rasterio.open(os.path.join(self.root, naip_fn)) as f:
                dst_crs = f.crs.to_string().lower()
                query_box_transformed = shapely.ops.transform(
                    self.p_transformers[dst_crs], query_box
                ).envelope
                query_geom_transformed = shapely.geometry.mapping(query_box_transformed)
                naip_data, _ = rasterio.mask.mask(
                    f, [query_geom_transformed], crop=True, all_touched=True
                )

            with rasterio.open(os.path.join(self.root, lc_fn)) as f:
                lc_data, _ = rasterio.mask.mask(
                    f, [query_geom_transformed], crop=True, all_touched=True
                )

        else:
            raise IndexError(f"query: {query} spans multiple tiles which is not valid")

        sample = {
            "image": naip_data,
            "mask": lc_data,
            "crs": self.crs,
            "bbox": query,
        }

        print(naip_data.shape, query)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset MD5s match, else False
        """
        if self.checksum:
            integrity: bool = check_integrity(
                os.path.join(self.root, self.filename),
                self.md5,
            )
            return integrity
        else:
            return True

    def _download(self) -> None:
        """Download the dataset and extract it."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5,
        )
