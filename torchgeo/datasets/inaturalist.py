# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for iNaturalist."""

import functools
import glob
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
from .utils import BoundingBox, Path, disambiguate_timestamp


class INaturalist(GeoDataset):
    """Dataset for iNaturalist.

    `iNaturalist <https://www.inaturalist.org/>`__ is a joint initiative of the
    California Academy of Sciences and the National Geographic Society. It allows
    citizen scientists to upload observations of organisms that can be downloaded by
    scientists and researchers.

    If you use an iNaturalist dataset in your research, please cite it according to:

    * https://help.inaturalist.org/en/support/solutions/articles/151000170344-how-should-i-cite-inaturalist-

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

        files = glob.glob(os.path.join(root, '**.csv'))
        if not files:
            raise DatasetNotFoundError(self)

        # Read CSV file
        usecols = ['observed_on', 'time_observed_at', 'latitude', 'longitude']
        df = pd.read_csv(files[0], header=0, usecols=usecols)
        df = df[df.latitude.notna()]
        df = df[df.longitude.notna()]

        # Convert from pandas DataFrame to geopandas GeoDataFrame
        func = functools.partial(disambiguate_timestamp, format='%Y-%m-%d %H:%M:%S %z')
        time = df.time_observed_at.apply(func)
        func = functools.partial(disambiguate_timestamp, format='%Y-%m-%d')
        date = df.observed_on.apply(func)
        time[time.isnull()] = date[time.isnull()]
        index = pd.IntervalIndex.from_tuples(time, closed='both', name='datetime')
        geometry = gpd.points_from_xy(df.longitude, df.latitude)
        self.index = GeoDataFrame(index=index, geometry=geometry, crs='EPSG:4326')

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        interval = pd.Interval(query.mint, query.maxt)
        index = self.index.iloc[self.index.index.overlaps(interval)]
        index = index.cx[query.minx : query.maxx, query.miny : query.maxy]  # type: ignore[misc]

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
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(ls='--')

        # Extract coordinates and timestamps
        index = sample['bounds']
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
            ax.set_title('iNaturalist Dataset Plot')

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig
