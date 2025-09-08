# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""OpenStreetMap dataset."""

import contextlib
import hashlib
import json
import pathlib
import time
from collections.abc import Callable, Iterable
from typing import Any, ClassVar, cast
from urllib.request import Request, urlopen

import geopandas as gpd
import matplotlib.pyplot as plt
import shapely
from geopandas import GeoDataFrame
from matplotlib.figure import Figure
from pyproj import CRS

from .errors import DatasetNotFoundError
from .geo import VectorDataset
from .utils import Path


class OpenStreetMap(VectorDataset):
    """OpenStreetMap dataset.

    The `OpenStreetMap <https://www.openstreetmap.org/>`__ dataset provides
    access to crowd-sourced geographic data. This implementation uses the
    `Overpass API <https://wiki.openstreetmap.org/wiki/Overpass_API>`__
    to query and download OSM data for a specified geographic bounding box
    at initialization, then allows spatial querying of the cached data.

    Dataset features:

    * Vector data (points, lines, polygons) for various geographic features
    * Flexible querying by feature type (buildings, roads, amenities, etc.)
    * Data fetched once at initialization and cached locally
    * Standard GeoDataset spatial indexing and CRS support
    * Support for custom Overpass QL queries

    If you use this dataset in your research, please cite the following:

    * https://www.openstreetmap.org/copyright
    """

    # Overpass API endpoints (will try in order if one fails)
    _overpass_endpoints: ClassVar[list[str]] = [
        'https://overpass-api.de/api/interpreter',
        'https://overpass.kumi.systems/api/interpreter',
    ]

    # Rate limiting (minimum seconds between requests)
    _min_request_interval = 1.0
    _last_request_time = 0.0

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] = (0.0001, 0.0001),
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        feature_type: str = 'building',
        custom_query: str | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new OpenStreetMap dataset instance.

        Args:
            bbox: bounding box for initial data fetch as (minx, miny, maxx, maxy) in EPSG:4326
            paths: root directory where dataset will be stored
            crs: coordinate reference system (CRS) to warp to (defaults to EPSG:4326)
            res: resolution of the dataset in units of CRS
            transforms: a function/transform that takes input sample and returns
                a transformed version
            feature_type: type of OSM features to query for. Common types include:
                - 'building': buildings, houses, commercial structures
                - 'highway': roads, paths, railways (any transportation route)
                - 'amenity': restaurants, shops, hospitals, schools, etc.
                - 'leisure': parks, sports facilities, playgrounds
                - 'natural': water bodies, forests, beaches, cliffs
                - 'landuse': residential, commercial, industrial, farmland areas
                - 'shop': specific retail establishments
                - 'tourism': hotels, attractions, information centers
                - 'man_made': towers, bridges, pipelines, walls
                For a complete list of feature types and their subtypes, see:
                https://wiki.openstreetmap.org/wiki/Map_features
            custom_query: custom Overpass QL query string. If provided, feature_type
                is ignored. For query syntax, see:
                https://wiki.openstreetmap.org/wiki/Overpass_API
            download: if True, download dataset and store it in the root directory

        Raises:
            DatasetNotFoundError: if dataset is not found and download is False
        """
        self.bbox = bbox
        self.feature_type = feature_type
        self.custom_query = custom_query

        # Handle paths parameter
        if isinstance(paths, str | pathlib.Path):
            self.root = pathlib.Path(paths)
        else:
            # If it's an iterable, take the first one as root
            paths_iterable = cast(Iterable[Path], paths)
            self.root = pathlib.Path(next(iter(paths_iterable)))

        # Create data directory
        self.root.mkdir(parents=True, exist_ok=True)

        # Download data if requested
        if download:
            self._download_data()

        # Check that we have the data file
        if not self._check_integrity():
            raise DatasetNotFoundError(self)

        # Initialize parent VectorDataset with the downloaded file
        data_file = self._get_data_filename()
        super().__init__(paths=data_file, crs=crs, res=res, transforms=transforms)

    def _get_data_filename(self) -> pathlib.Path:
        """Get the filename for the cached data file."""
        # Create a hash of the query parameters for filename
        cache_key = {
            'bbox': self.bbox,
            'feature_type': self.feature_type,
            'custom_query': self.custom_query,
        }
        cache_str = json.dumps(cache_key, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:16]

        return self.root / f'osm_{self.feature_type}_{cache_hash}.geojson'

    def _check_integrity(self) -> bool:
        """Check if the dataset file exists."""
        return self._get_data_filename().exists()

    def _rate_limit(self) -> None:
        """Implement rate limiting for API requests."""
        current_time = time.time()
        elapsed = current_time - OpenStreetMap._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        OpenStreetMap._last_request_time = time.time()

    def _build_overpass_query(self) -> str:
        """Build an Overpass QL query for the specified bounding box.

        Returns:
            Overpass QL query string
        """
        if self.custom_query:
            # Replace bbox placeholder in custom query if present
            minx, miny, maxx, maxy = self.bbox
            return self.custom_query.replace('{{bbox}}', f'{miny},{minx},{maxy},{maxx}')

        minx, miny, maxx, maxy = self.bbox
        # Convert to Overpass API bbox format: south, west, north, east
        overpass_bbox = f'{miny},{minx},{maxy},{maxx}'

        # Build query based on feature type
        if self.feature_type == 'building':
            query = f"""
            [out:json][timeout:25];
            (
              way["building"]({overpass_bbox});
              relation["building"]({overpass_bbox});
            );
            out geom;
            """
        elif self.feature_type == 'highway':
            query = f"""
            [out:json][timeout:25];
            (
              way["highway"]({overpass_bbox});
            );
            out geom;
            """
        elif self.feature_type == 'amenity':
            query = f"""
            [out:json][timeout:25];
            (
              node["amenity"]({overpass_bbox});
              way["amenity"]({overpass_bbox});
              relation["amenity"]({overpass_bbox});
            );
            out geom;
            """
        else:
            # Generic query for any feature type
            query = f"""
            [out:json][timeout:25];
            (
              node["{self.feature_type}"]({overpass_bbox});
              way["{self.feature_type}"]({overpass_bbox});
              relation["{self.feature_type}"]({overpass_bbox});
            );
            out geom;
            """

        return query.strip()

    def _download_data(self) -> None:
        """Download OSM data from Overpass API."""
        data_file = self._get_data_filename()

        # Skip if already exists
        if data_file.exists():
            return

        # Build query
        query = self._build_overpass_query()

        # Try each endpoint until one works
        last_exception = None
        for endpoint in self._overpass_endpoints:
            try:
                self._rate_limit()

                # Make request
                req = Request(endpoint, data=query.encode('utf-8'))
                req.add_header('Content-Type', 'application/x-www-form-urlencoded')

                with urlopen(req, timeout=30) as response:
                    data = response.read()

                # Parse JSON response
                osm_data = json.loads(data.decode('utf-8'))

                # Convert to GeoDataFrame
                gdf = self._parse_overpass_response(osm_data)

                # Save to file
                if len(gdf) > 0:
                    gdf.to_file(data_file, driver='GeoJSON')
                else:
                    # Create empty file to indicate we tried
                    gpd.GeoDataFrame(
                        columns=['geometry'], geometry='geometry', crs='EPSG:4326'
                    ).to_file(data_file, driver='GeoJSON')

                return

            except Exception as e:
                last_exception = e
                continue

        # If we get here, all endpoints failed
        raise RuntimeError(
            f'All Overpass API endpoints failed. Last error: {last_exception}'
        )

    def _parse_overpass_response(self, osm_data: dict[str, Any]) -> GeoDataFrame:
        """Parse Overpass API response into a GeoDataFrame.

        Args:
            osm_data: JSON response from Overpass API

        Returns:
            GeoDataFrame containing parsed geometries and properties
        """
        geometries = []
        properties = []

        for element in osm_data.get('elements', []):
            geom = self._element_to_geometry(element)
            if geom is not None:
                geometries.append(geom)
                # Extract properties, excluding geometry-related fields
                props = element.get('tags', {}).copy()
                props['osm_id'] = element.get('id')
                props['osm_type'] = element.get('type')
                properties.append(props)

        if not geometries:
            return gpd.GeoDataFrame(
                columns=['geometry'], geometry='geometry', crs='EPSG:4326'
            )

        return gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')

    def _element_to_geometry(self, element: dict[str, Any]) -> shapely.Geometry | None:
        """Convert OSM element to Shapely geometry.

        Args:
            element: OSM element from Overpass API response

        Returns:
            Shapely geometry or None if conversion fails
        """
        element_type = element.get('type')

        with contextlib.suppress(Exception):
            if element_type == 'node':
                lat = element.get('lat')
                lon = element.get('lon')
                if lat is not None and lon is not None:
                    return shapely.Point(lon, lat)

            elif element_type == 'way':
                if 'geometry' in element:
                    coords = [
                        (node['lon'], node['lat']) for node in element['geometry']
                    ]
                    if len(coords) >= 2:
                        return (
                            shapely.Polygon(coords)
                            if len(coords) >= 4 and coords[0] == coords[-1]
                            else shapely.LineString(coords)
                        )

        return None

    def __getitem__(self, query: Any) -> dict[str, Any]:
        """Retrieve vector data indexed by spatiotemporal slice.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index

        Returns:
            Sample containing raw vector data and metadata
        """
        # Get the basic sample from VectorDataset (contains CRS, bounds, etc.)
        sample = super().__getitem__(query)

        # Read the raw vector data from the GeoJSON file
        gdf = gpd.read_file(self._get_data_filename())

        # If we have vector data, filter it to the query bounds
        if len(gdf) > 0:
            x, y, t = self._disambiguate_slice(query)

            # Create a bounding box for filtering
            from shapely.geometry import box

            query_bounds = box(x.start, y.start, x.stop, y.stop)

            # Filter geometries that intersect with the query bounds
            gdf_filtered = gdf[gdf.geometry.intersects(query_bounds)]

            # Add the filtered vector data to the sample
            sample['vector'] = gdf_filtered
        else:
            # Return empty GeoDataFrame if no data
            sample['vector'] = gpd.GeoDataFrame(
                columns=['geometry'], geometry='geometry', crs='EPSG:4326'
            )

        return sample

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`VectorDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # VectorDataset can return either 'vector' key or rasterized mask
        if 'vector' in sample:
            gdf = sample['vector']
        else:
            # If VectorDataset rasterized the data, read the original vector data
            gdf = gpd.read_file(self._get_data_filename())

        if len(gdf) == 0:
            ax.text(
                0.5,
                0.5,
                'No data found for this area',
                ha='center',
                va='center',
                transform=ax.transAxes,
                fontsize=14,
            )
        else:
            gdf.plot(ax=ax, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Fix axis formatting to avoid scientific notation
        ax.ticklabel_format(useOffset=False, style='plain')

        if show_titles:
            feature_name = 'Custom Query' if self.custom_query else self.feature_type
            ax.set_title(f'OpenStreetMap - {feature_name}', fontsize=14, pad=20)

        if suptitle is not None:
            fig.suptitle(suptitle)

        plt.tight_layout()
        return fig
