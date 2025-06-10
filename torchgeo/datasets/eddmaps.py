# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for EDDMapS."""

import functools
import os
from datetime import datetime
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from .errors import DatasetNotFoundError
from .geo import GeoDataset
from .utils import GeoSlice, Path, disambiguate_timestamp


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
        df = pd.read_csv(filepath, usecols=['ObsDate', 'Latitude', 'Longitude'])
        df = df[df.Latitude.notna()]
        df = df[df.Longitude.notna()]

        # Convert from pandas DataFrame to geopandas GeoDataFrame
        func = functools.partial(disambiguate_timestamp, format='%m-%d-%y')
        index = pd.IntervalIndex.from_tuples(
            df['ObsDate'].apply(func), closed='both', name='datetime'
        )
        geometry = gpd.points_from_xy(df.Longitude, df.Latitude)
        self.index = GeoDataFrame(index=index, geometry=geometry, crs='EPSG:4326')

    def __getitem__(self, query: GeoSlice) -> dict[str, Any]:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *query* is not found in the index.
        """
        x, y, t = self._disambiguate_slice(query)
        interval = pd.Interval(t.start, t.stop)
        index = self.index.iloc[self.index.index.overlaps(interval)]
        index = index.iloc[:: t.step]
        index = index.cx[x.start : x.stop, y.start : y.stop]  # type: ignore[misc]

        if index.empty:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        sample = {'crs': self.crs, 'bounds': index}

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
            suptitle: optional suptitle to use for Figure
        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.8
        """
        # Create figure and axis - using regular matplotlib axes
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(ls='--')

        # Extract bounding boxes (coordinates) from the sample
        index = sample['bounds']

        # Extract coordinates and timestamps
        longitudes = [point.x for point in index.geometry]
        latitudes = [point.y for point in index.geometry]
        timestamps = [time.timestamp() for time in index.index.left]

        # Plot the points with colors based on date
        scatter = ax.scatter(longitudes, latitudes, c=timestamps, edgecolors='black')

        # Create a formatter function
        def format_date(x: float, pos: int | None = None) -> str:
            # Convert timestamp to datetime
            return datetime.fromtimestamp(x).strftime('%Y-%m-%d')

        # Add a colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.04)
        cbar.set_label('Observed Timestamp', rotation=90, labelpad=-100, va='center')

        # Apply the formatter to the colorbar
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_date))

        # Set labels
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Add titles if requested
        if show_titles:
            ax.set_title('EDDMapS Observation Locations by Date')

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig
