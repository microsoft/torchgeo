# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for iNaturalist."""

import glob
import os
import sys
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from rasterio.crs import CRS

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

        # Read CSV file
        data = pd.read_csv(
            files[0],
            engine='c',
            usecols=['observed_on', 'time_observed_at', 'latitude', 'longitude'],
        )

        # Dataset contains many possible timestamps:
        #
        # * observed_on_string: no consistent format (can't use)
        # * observed_on: day precision (better)
        # * time_observed_at: second precision (best)
        # * created_at: when observation was submitted (shouldn't use)
        # * updated_at: when submission was updated (shouldn't use)
        #
        # The created_at/updated_at timestamps can be years after the actual submission,
        # so they shouldn't be used, even if observed_on/time_observed_at are missing.

        # Convert from pandas DataFrame to rtree Index
        i = 0
        for date, time, y, x in data.itertuples(index=False, name=None):
            # Skip rows without lat/lon
            if pd.isna(y) or pd.isna(x):
                continue

            if not pd.isna(time):
                mint, maxt = disambiguate_timestamp(time, '%Y-%m-%d %H:%M:%S %z')
            elif not pd.isna(date):
                mint, maxt = disambiguate_timestamp(date, '%Y-%m-%d')
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
            suptitle: optional suptitle to use for Figure
        Returns:
            a matplotlib Figure with the rendered sample

        .. versionadded:: 0.8
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(ls='--')

        # Extract coordinates and timestamps
        bboxes = sample['bounds']
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
            ax.set_title('GBIF Occurence Locations by Date')

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig
