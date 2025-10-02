# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""OpenStreetMap dataset."""

import contextlib
import hashlib
import json
import pathlib
import re
import time
import warnings
from collections.abc import Callable, Iterable
from typing import Any, ClassVar, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    Dataset features
    ----------------

    * Vector data (points, lines, polygons) for various geographic features
    * Flexible querying by channel configuration (buildings, highways, amenities, etc.)
    * Data fetched once at initialization and cached locally
    * Channel-based labeling with priority-based assignment

    Channel priority and label assignment
    -------------------------------------
    `channels` is a list of dicts defining feature classes. Each has `name` (str) and `selector` (list of OSM tag filters).
    Features are assigned labels based on the order of channels in this list:

    - First channel gets label=1, second gets label=2, etc.
    - If a feature matches multiple channels, it receives the label of the first matching channel
    - Features that don't match any channel get label=0 (background)

    Example::

        channels = [
            {'name': 'buildings', 'selector': [{'building': '*'}]},        # label=1
            {'name': 'roads', 'selector': [{'highway': '*'}]},             # label=2
            {'name': 'commercial', 'selector': [{'landuse': 'commercial'}]} # label=3
        ]

        # A feature with tags {'building': 'yes', 'landuse': 'commercial'}
        # would get label=1 (buildings) because buildings comes first

    If you use this dataset in your research, please cite the following source:

    * https://www.openstreetmap.org/copyright

    .. versionadded:: 0.8
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
        channels: list[dict[str, Any]],
        paths: Path | Iterable[Path] = 'data',
        res: float | tuple[float, float] = (0.0001, 0.0001),
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new OpenStreetMap dataset instance.

        Args:
            bbox: bounding box for initial data fetch as (minx, miny, maxx, maxy) in EPSG:4326
            channels: list of dicts defining feature classes. Each has `name` (str) and `selector` (list of OSM tag filters).
                Features get labels 1-N based on channel order, with first match taking priority.
            paths: root directory where dataset will be stored
            res: resolution of the dataset in units of EPSG:4326 (degrees)
            transforms: a function/transform that takes input sample and returns
                a transformed version
            download: if True, download dataset and store it in the root directory

        Raises:
            DatasetNotFoundError: if dataset is not found and download is False
            ValueError: if invalid channel configuration
        """
        # Validate channels parameter
        self._validate_channels(channels)

        self.bbox = bbox
        self.channels = channels

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

        super().__init__(
            paths=data_file,
            crs=CRS.from_epsg(4326),  # Always use WGS84 for OSM data
            res=res,
            transforms=transforms,
            label_name='label',
        )

        # Load GeoDataFrame once and store as attribute for efficiency
        self.gdf = gpd.read_file(data_file)

        # Check for empty channels and warn user
        if self.channels:
            self._check_empty_channels()

    def __len__(self) -> int:
        """Return the number of features in the dataset.

        Returns:
            number of OSM features in the dataset
        """
        return len(self.gdf)

    def _validate_channels(self, channels: list[dict[str, Any]]) -> None:
        """Validate channels configuration."""
        if not isinstance(channels, list) or not channels:
            raise ValueError('channels must be a non-empty list')

        for i, channel in enumerate(channels):
            if not isinstance(channel, dict):
                raise ValueError(f'Channel {i} must be a dictionary')
            if 'name' not in channel or 'selector' not in channel:
                raise ValueError(f'Channel {i} must have "name" and "selector" keys')
            if not isinstance(channel['selector'], list):
                raise ValueError(f'Channel {i} selector must be a list')
            for j, selector in enumerate(channel['selector']):
                if not isinstance(selector, dict):
                    raise ValueError(f'Channel {i} selector {j} must be a dictionary')

    def _get_data_filename(self) -> pathlib.Path:
        """Get the filename for the cached data file."""
        # Create a hash of the query parameters for filename
        cache_key = {'bbox': self.bbox, 'channels': self.channels}
        cache_str = json.dumps(cache_key, sort_keys=True)
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:16]
        return self.root / f'osm_features_{cache_hash}.geojson'

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
        """Build Overpass query from channels configuration.

        Returns:
            Overpass QL query string
        """
        minx, miny, maxx, maxy = self.bbox
        overpass_bbox = f'{miny},{minx},{maxy},{maxx}'

        queries = []
        for channel in self.channels:
            for selector in channel['selector']:
                for tag, values in selector.items():
                    if values == '*':
                        # Tag exists, any value
                        queries.append(f'wr["{tag}"]({overpass_bbox});')
                    elif isinstance(values, list):
                        # Multiple specific values
                        regex = f'^({"|".join(re.escape(v) for v in values)})$'
                        queries.append(f'wr["{tag}"~"{regex}"]({overpass_bbox});')
                    else:
                        # Single specific value
                        queries.append(f'wr["{tag}"="{values}"]({overpass_bbox});')

        query = f"""
        [out:json][timeout:25];
        (
          {chr(10).join('  ' + q for q in queries)}
        );
        out tags geom;
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
                payload = urlencode({'data': query}).encode('utf-8')
                req = Request(endpoint, data=payload)
                req.add_header(
                    'Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8'
                )
                req.add_header('Accept', 'application/json')

                with urlopen(req, timeout=30) as response:
                    data = response.read()

                # Parse JSON response
                osm_data = json.loads(data.decode('utf-8'))

                # Convert to GeoDataFrame
                gdf = self._parse_overpass_response(osm_data)

                # Save to file or raise error if no features found
                if len(gdf) > 0:
                    gdf.to_file(data_file, driver='GeoJSON')
                    return
                else:
                    # No features found - provide clear error message
                    minx, miny, maxx, maxy = self.bbox
                    bbox_str = f'{minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f}'

                    msg = (
                        f'No features found in the specified area (bbox: {bbox_str}). '
                    )

                    msg += 'Try a different feature type, larger bounding box, or different geographic area.'
                    raise ValueError(msg)

            except ValueError:
                # Re-raise ValueError for empty results with clear message
                raise
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
            GeoDataFrame containing parsed geometries, properties, and pre-computed labels
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
                columns=['geometry', 'label'], geometry='geometry', crs='EPSG:4326'
            )

        # Add pre-computed labels directly from properties (avoid slow .iterrows())
        labels = [
            self._get_channel_label({'properties': props}) for props in properties
        ]

        # Add labels to properties before creating GeoDataFrame
        for props, label in zip(properties, labels):
            props['label'] = label

        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')
        return gdf

    def _element_to_geometry(self, element: dict[str, Any]) -> shapely.Geometry | None:
        """Convert OSM element to Shapely geometry.

        Args:
            element: OSM element from Overpass API response

        Returns:
            Shapely geometry or None if conversion fails
        """
        element_type = element.get('type')

        with contextlib.suppress(KeyError, ValueError, TypeError):
            if element_type == 'node':
                lat = element.get('lat')
                lon = element.get('lon')
                if lat is not None and lon is not None:
                    return shapely.Point(lon, lat)

            elif element_type in ('way', 'relation'):
                if 'geometry' in element:
                    coords = [
                        (node['lon'], node['lat']) for node in element['geometry']
                    ]
                    if len(coords) >= 2:
                        tags = element.get('tags', {})
                        is_area = tags.get('area') == 'yes' or (
                            len(coords) >= 4 and coords[0] == coords[-1]
                        )
                        return (
                            shapely.Polygon(coords)
                            if is_area
                            else shapely.LineString(coords)
                        )

        return None

    def get_label(self, feature: dict[str, Any]) -> int:
        """Get label value to use for rendering a feature.

        Args:
            feature: the feature from which to extract the label.

        Returns:
            the integer label, or 0 if the feature should not be rendered.
        """
        # Try to use pre-computed label first
        if 'properties' in feature and 'label' in feature['properties']:
            return int(feature['properties']['label'])

        # Fallback to channels computation (shouldn't normally be needed)
        return self._get_channel_label(feature)

    def _get_channel_label(self, feature: dict[str, Any]) -> int:
        """Get label based on channels priority (first match wins).

        Channels are checked in the order they appear in self.channels list.
        The first channel whose selector matches the feature determines the label.
        This means if a feature has tags matching multiple channels, only the
        first matching channel's label is assigned.

        Args:
            feature: the feature from which to extract the label.

        Returns:
            the integer label (1-based), or 0 if no match.
        """
        props = feature.get('properties', {})

        # Check each channel in order (priority-based)
        for channel_idx, channel in enumerate(self.channels):
            for selector in channel['selector']:
                if self._feature_matches_selector(props, selector):
                    return channel_idx + 1  # 1-based labeling

        return 0  # No match

    def _feature_matches_selector(
        self, props: dict[str, Any], selector: dict[str, Any]
    ) -> bool:
        """Check if feature properties match a selector.

        Args:
            props: feature properties (may contain JSON string from GeoDataFrame)
            selector: selector dictionary

        Returns:
            True if feature matches selector
        """
        # Handle case where properties might be stored as JSON string
        if 'properties' in props and isinstance(props['properties'], str):
            try:
                actual_props = json.loads(props['properties'])
            except (json.JSONDecodeError, TypeError):
                actual_props = props
        else:
            actual_props = props

        for tag, expected_values in selector.items():
            if tag not in actual_props:
                return False

            actual_value = actual_props[tag]
            if actual_value is None:
                return False

            # Also check for pandas NaN values
            if pd.isna(actual_value):
                return False

            if expected_values == '*':
                # Any value is acceptable
                continue
            elif isinstance(expected_values, list):
                # Must match one of the specified values
                if actual_value not in expected_values:
                    return False
            elif actual_value != expected_values:
                return False

        return True

    def _check_empty_channels(self) -> None:
        """Check for channels with no geometries and warn the user."""
        if not self.channels or len(self.gdf) == 0:
            return

        # Use pre-computed labels to count channel usage
        label_counts = self.gdf['label'].value_counts()

        # Warn about empty channels
        for i, channel in enumerate(self.channels):
            channel_label = i + 1  # 1-based labeling
            if label_counts.get(channel_label, 0) == 0:
                warnings.warn(
                    f"Channel '{channel['name']}' (label={channel_label}) has no geometries in this AOI. "
                    f'This may be due to no features of this type in the area or all features '
                    f'being assigned to higher-priority channels.',
                    UserWarning,
                    stacklevel=2,
                )

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
        mask = sample['mask'].squeeze()
        ncols = 1

        showing_prediction = 'prediction' in sample
        if showing_prediction:
            pred = sample['prediction'].squeeze()
            ncols = 2

        # Color palette for channels
        colors = [
            '#FF6B6B',
            '#4ECDC4',
            '#45B7D1',
            '#96CEB4',
            '#FECA57',
            '#FF9FF3',
            '#54A0FF',
            '#5F27CD',
            '#00D2D3',
            '#FF3838',
            '#FF9500',
            '#7bed9f',
        ]

        def apply_cmap(
            arr: 'np.typing.NDArray[Any]',
        ) -> 'np.typing.NDArray[np.float64]':
            """Apply colormap to label array."""
            # Convert tensor to numpy if needed
            if hasattr(arr, 'numpy'):
                arr = arr.numpy()

            # Create RGB image
            h, w = arr.shape
            rgb = np.zeros((h, w, 3), dtype=np.float64)

            # Color 0 (background) as black
            # Color each label with its corresponding channel color
            for label in np.unique(arr):
                if label == 0:
                    continue
                channel_idx = int(label - 1)
                if channel_idx < len(colors):
                    # Convert hex to RGB
                    hex_color = colors[channel_idx % len(colors)]
                    r = int(hex_color[1:3], 16) / 255.0
                    g = int(hex_color[3:5], 16) / 255.0
                    b = int(hex_color[5:7], 16) / 255.0
                    rgb[arr == label] = [r, g, b]

            return rgb

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 5, 5))

        # Create legend handles
        legend_handles = []
        unique_labels = np.unique(mask.numpy() if hasattr(mask, 'numpy') else mask)
        for label in unique_labels:
            if label == 0:
                continue
            channel_idx = int(label - 1)
            if channel_idx < len(self.channels):
                channel_name = self.channels[channel_idx]['name']
                color = colors[channel_idx % len(colors)]
                legend_handles.append(
                    mpatches.Patch(color=color, label=channel_name.title())
                )

        if showing_prediction:
            axs[0].imshow(apply_cmap(mask))
            axs[0].axis('off')
            axs[1].imshow(apply_cmap(pred))
            axs[1].axis('off')
            if show_titles:
                axs[0].set_title('Mask')
                axs[1].set_title('Prediction')
            if legend_handles:
                axs[0].legend(handles=legend_handles, loc='upper right')
        else:
            axs.imshow(apply_cmap(mask))
            axs.axis('off')
            if show_titles:
                axs.set_title('Mask')
            if legend_handles:
                axs.legend(handles=legend_handles, loc='upper right')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
