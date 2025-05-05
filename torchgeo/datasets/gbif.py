# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for the Global Biodiversity Information Facility."""

import glob
import os
import sys
from datetime import datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from rasterio.crs import CRS

from .errors import DatasetNotFoundError
from .geo import GeoDataset
from .utils import BoundingBox, Path


def _disambiguate_timestamps(
    year: float, month: float, day: float
) -> tuple[float, float]:
    """Disambiguate partial timestamps.

    Based on :func:`torchgeo.datasets.utils.disambiguate_timestamps`.

    Args:
        year: year, possibly nan
        month: month, possibly nan
        day: day, possibly nan

    Returns:
        minimum and maximum possible time range
    """
    if np.isnan(year):
        # No temporal info
        return 0, sys.maxsize
    elif np.isnan(month):
        # Year resolution
        mint = datetime(int(year), 1, 1)
        maxt = datetime(int(year) + 1, 1, 1)
    elif np.isnan(day):
        # Month resolution
        mint = datetime(int(year), int(month), 1)
        if month == 12:
            maxt = datetime(int(year) + 1, 1, 1)
        else:
            maxt = datetime(int(year), int(month) + 1, 1)
    else:
        # Day resolution
        mint = datetime(int(year), int(month), int(day))
        maxt = mint + timedelta(days=1)

    maxt -= timedelta(microseconds=1)

    return mint.timestamp(), maxt.timestamp()


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

        files = glob.glob(os.path.join(root, '**.csv'))
        if not files:
            raise DatasetNotFoundError(self)

        # Read tab-delimited CSV file
        data = pd.read_table(
            files[0],
            engine='c',
            usecols=['decimalLatitude', 'decimalLongitude', 'day', 'month', 'year'],
        )

        # Convert from pandas DataFrame to rtree Index
        i = 0
        for y, x, day, month, year in data.itertuples(index=False, name=None):
            # Skip rows without lat/lon
            if np.isnan(y) or np.isnan(x):
                continue

            mint, maxt = _disambiguate_timestamps(year, month, day)

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
            suptitle: optional suptitle to use for Figure
        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.8
        """
        # Create figure and axis - using regular matplotlib axes
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(ls='--')

        # Extract bounding boxes (coordinates) from the sample
        bboxes = sample['bounds']

        # Extract coordinates and timestamps
        longitudes = [bbox[0] for bbox in bboxes]  # minx
        latitudes = [bbox[1] for bbox in bboxes]  # miny
        timestamps = [bbox[2] for bbox in bboxes]  # mint

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
