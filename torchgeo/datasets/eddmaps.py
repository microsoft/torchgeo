# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for EDDMapS."""

import datetime
import os
import sys
from typing import Any

import contextily as ctx
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from rasterio.crs import CRS

from .errors import DatasetNotFoundError
from .geo import GeoDataset
from .utils import BoundingBox, Path, disambiguate_timestamp


class EDDMapS(GeoDataset):
    """Dataset for EDDMapS.

    `EDDMapS <https://www.eddmaps.org/>`__, Early Detection and Distribution Mapping
    System, is a web-based mapping system for documenting invasive species and pest
    distribution. Launched in 2005 by the Center for Invasive Species and Ecosystem
    Health at the University of Georgia, it was originally designed as a tool for
    state Exotic Pest Plant Councils to develop more complete distribution data of
    invasive species. Since then, the program has expanded to include the entire US
    and Canada as well as to document certain native pest species.

    EDDMapS query results can be downloaded in CSV, KML, or Shapefile format. This
    dataset currently only supports CSV files.

    If you use an EDDMapS dataset in your research, please cite it like so:

    * EDDMapS. *YEAR*. Early Detection & Distribution Mapping System. The University of
      Georgia - Center for Invasive Species and Ecosystem Health. Available online at
      https://www.eddmaps.org/; last accessed *DATE*.

    .. versionadded:: 0.3
    """

    res = (0, 0)
    _crs = CRS.from_epsg(4326)  # Lat/Lon

    def __init__(self, root: Path = 'data') -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found

        Raises:
            DatasetNotFoundError: If dataset is not found.
        """
        super().__init__()

        self.root = root

        filepath = os.path.join(root, 'mappings.csv')
        if not os.path.exists(filepath):
            raise DatasetNotFoundError(self)

        # Read CSV file
        data = pd.read_csv(
            filepath, engine='c', usecols=['ObsDate', 'Latitude', 'Longitude']
        )

        # Convert from pandas DataFrame to rtree Index
        i = 0
        for date, y, x in data.itertuples(index=False, name=None):
            # Skip rows without lat/lon
            if np.isnan(y) or np.isnan(x):
                continue

            if not pd.isna(date):
                mint, maxt = disambiguate_timestamp(date, '%m-%d-%y')
            else:
                mint, maxt = 0, sys.maxsize

            coords = (x, x, y, y, mint, maxt)
            self.index.insert(i, coords)
            i += 1

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        bboxes = [hit.bbox for hit in hits]

        if not bboxes:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        sample = {'crs': self.crs, 'bounds': bboxes}

        return sample

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.8
        """
        # Create figure and axis - using regular matplotlib axes
        fig, ax = plt.subplots(figsize=(10, 8))

        # Extract bounding boxes (coordinates) from the sample
        bboxes = sample['bounds']

        # Extract coordinates and timestamps
        longitudes = [bbox[0] for bbox in bboxes]  # minx
        latitudes = [bbox[1] for bbox in bboxes]  # miny
        timestamps = [bbox[2] for bbox in bboxes]  # mint (timestamp)

        # Convert timestamps to datetime objects
        dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

        # Create a normalize object for color mapping
        if dates:
            min_date, max_date = min(dates), max(dates)
            min_date_num: float = mdates.date2num(min_date)  # type: ignore
            max_date_num: float = mdates.date2num(max_date)  # type: ignore
            norm = Normalize(min_date_num, max_date_num)

            date_nums: list[float] = []
            for date in dates:
                date_nums.append(mdates.date2num(date))  # type: ignore
        else:
            norm = Normalize(0, 1)
            date_nums = []

        # Plot the points with colors based on date
        scatter = ax.scatter(
            longitudes,
            latitudes,
            c=date_nums,
            cmap='coolwarm',
            norm=norm,
            s=30,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
            zorder=3,  # Ensure points appear above basemap
        )

        # Add a more polished colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)

        # Improve colorbar tick formatting
        locator: mdates.AutoDateLocator = mdates.AutoDateLocator()  # type: ignore
        formatter: mdates.ConciseDateFormatter = mdates.ConciseDateFormatter(locator)  # type: ignore
        cbar.ax.yaxis.set_major_locator(locator)
        cbar.ax.yaxis.set_major_formatter(formatter)

        # Set better colorbar label with proper positioning
        cbar.set_label(
            'Observation Date',
            rotation=270,
            labelpad=20,
            fontsize=12,
            fontweight='bold',
        )

        # Calculate the extent of the data for better visualization
        if longitudes and latitudes:
            min_lon, max_lon = min(longitudes), max(longitudes)
            min_lat, max_lat = min(latitudes), max(latitudes)

            # Add some padding
            lon_padding = (max_lon - min_lon) * 0.2 if max_lon != min_lon else 0.5
            lat_padding = (max_lat - min_lat) * 0.2 if max_lat != min_lat else 0.5

            # Set axis limits with padding
            ax.set_xlim(min_lon - lon_padding, max_lon + lon_padding)
            ax.set_ylim(min_lat - lat_padding, max_lat + lat_padding)

        # Add the basemap tiles
        try:
            # Add basemap (OpenStreetMap by default)
            ctx.add_basemap(
                ax,
                source=ctx.providers.OpenStreetMap.Mapnik,  # You can change the provider
                zoom='auto',  # Automatically determine zoom level
                crs='EPSG:4326',  # WGS84 coordinate system (standard lat/lon)
            )
        except Exception as e:
            # Print warning if basemap fails (can happen due to network issues)
            print(f'Warning: Could not add basemap. Error: {e}')
            # Add a basic grid as fallback
            ax.grid(True, linestyle='--', alpha=0.7)

        # Set labels
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')

        # Add titles if requested
        if show_titles:
            ax.set_title(
                'EDDMapS Observation Locations by Date', fontsize=14, fontweight='bold'
            )

        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=16, fontweight='bold')

        fig.tight_layout()
        return fig
