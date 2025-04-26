# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for the Global Biodiversity Information Facility."""

import functools
import glob
import os
from datetime import datetime
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from .errors import DatasetNotFoundError
from .geo import GeoDataset
from .utils import BoundingBox, Path, disambiguate_timestamp


class GBIF(GeoDataset):
    """Dataset for the Global Biodiversity Information Facility.

    `GBIF <https://www.gbif.org/>`__, the Global Biodiversity Information Facility,
    is an international network and data infrastructure funded by the world's
    governments and aimed at providing anyone, anywhere, open access to data about
    all types of life on Earth.

    This dataset is intended for use with GBIF's
    `occurrence records <https://www.gbif.org/occurrence/search>`_. It may or may not
    work for other GBIF `datasets <https://www.gbif.org/dataset/search>`_. Data for a
    particular species or region of interest can be downloaded from the above link.

    If you use a GBIF dataset in your research, please cite it according to:

    * https://www.gbif.org/citation-guidelines

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

        # Read tab-delimited CSV file
        usecols = ['decimalLatitude', 'decimalLongitude', 'day', 'month', 'year']
        dtype = {'day': str, 'month': str, 'year': str}
        df = pd.read_table(files[0], usecols=usecols, dtype=dtype)
        df = df[df.decimalLatitude.notna()]
        df = df[df.decimalLongitude.notna()]
        df.day = df.day.str.zfill(2)
        df.month = df.month.str.zfill(2)
        date = df.day + ' ' + df.month + ' ' + df.year

        # Convert from pandas DataFrame to geopandas GeoDataFrame
        func = functools.partial(disambiguate_timestamp, format='%d %m %Y')
        index = pd.IntervalIndex.from_tuples(date.apply(func))
        geometry = gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude)
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
        geometry = shapely.box(*query[:4])
        interval = pd.Interval(*query[4:])
        index = self.index.iloc[self.index.index.overlaps(interval)]
        index = index.iloc[index.sindex.query(geometry, predicate='intersects')]

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
            ax.set_title('GBIF Occurrence Locations by Date')

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig
