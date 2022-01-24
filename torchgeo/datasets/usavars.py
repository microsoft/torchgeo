# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pickle
import rtree
import shapely

from .utils import download_url

class USAVars:
    url_prefix = "https://files.codeocean.com/files/verified/fa908bbc-11f9-4421-8bd3-72a4bf00427f_v2.0/data/int/applications/"

    label_urls = {
                "housing": url_prefix + "housing/outcomes_sampled_housing_CONTUS_16_640_POP_100000_0.csv?download",
                "income": url_prefix + "income/outcomes_sampled_income_CONTUS_16_640_POP_100000_0.csv?download",
                "roads": url_prefix + "roads/outcomes_sampled_roads_CONTUS_16_640_POP_100000_0.csv?download",
                "nightligths": url_prefix + "nightlights/outcomes_sampled_nightlights_CONTUS_16_640_POP_100000_0.csv?download",
                "population": url_prefix + "population/outcomes_sampled_population_CONTUS_16_640_UAR_100000_0.csv?download",
                "elevation": url_prefix + "elevation/outcomes_sampled_elevation_CONTUS_16_640_UAR_100000_0.csv?download",
                "treecover": url_prefix + "treecover/outcomes_sampled_treecover_CONTUS_16_640_UAR_100000_0.csv?download",
            }

    NAIP_BLOB_ROOT = 'https://naipblobs.blob.core.windows.net/naip/'
    NAIP_INDEX_BLOB_ROOT = "https://naipblobs.blob.core.windows.net/naip-index/rtree/"
    INDEX_FNS = ["tile_index.dat", "tile_index.idx", "tiles.p"]


    def __init__(
        self,
        root: str = "data",
    ) -> None:
        """Initialize a new USAVars dataset instance.
        """

        self.root = root

        self._verify()

        self.tile_rtree = rtree.index.Index(self.root + "/tile_index")
        self.tile_index = pickle.load(open(self.root + "/tiles.p", "rb"))

    def _verify(self) -> None:
        self._download()

    def _download(self) -> None:
        for f_name in self.label_urls:
            download_url(
                self.label_urls[f_name],
                self.root,
                filename=f_name + ".csv",
            )

        for fn in self.INDEX_FNS:
            download_url(
                self.NAIP_INDEX_BLOB_ROOT + fn,
                self.root,
            )

    def lookup_point(self, lat, lon):
        '''Given a lat/lon coordinate pair, return the list of NAIP tiles that *contain* that point.

        Args:
            lat (float): Latitude in EPSG:4326
            lon (float): Longitude in EPSG:4326
        Returns:
            intersected_files (list): A list of URLs of NAIP tiles that *contain* the given (`lat`, `lon`) point
        
        Raises:
            IndexError: Raised if no tile within the index contains the given (`lat`, `lon`) point
        '''

        point = shapely.geometry.Point(float(lon), float(lat))
        geom = shapely.geometry.mapping(point)

        return self.lookup_geom(geom)

    def lookup_geom(self, geom):
        '''Given a GeoJSON geometry, return the list of NAIP tiles that *contain* that feature.
        
        Args:
            geom (dict): A GeoJSON geometry in EPSG:4326
        Returns:
            intersected_files (list): A list of URLs of NAIP tiles that *contain* the given `geom`
        
        Raises:
            IndexError: Raised if no tile within the index fully contains the given `geom`
        '''
        shape = shapely.geometry.shape(geom)
        intersected_indices = list(self.tile_rtree.intersection(shape.bounds))

        intersected_files = []
        tile_intersection = False

        for idx in intersected_indices:
            intersected_file = self.tile_index[idx][0]
            intersected_geom = self.tile_index[idx][1]
            if intersected_geom.contains(shape):
                tile_intersection = True
                intersected_files.append(self.NAIP_BLOB_ROOT + intersected_file)

        if not tile_intersection and len(intersected_indices) > 0:
            raise IndexError("There are overlaps with tile index, but no tile contains the shape")
        elif len(intersected_files) <= 0:
            raise IndexError("No tile intersections")
        else:
            return intersected_files
