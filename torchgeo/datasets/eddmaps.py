# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for EDDMapS."""

import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
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

    def plot(self, query: BoundingBox | None = None) -> tuple[Figure, Axes]:
        """Plot the data using the R-tree index for efficient spatial querying.

        Args:
            query: Optional BoundingBox to filter data If None, all data will be plotted

        Returns:
            fig, ax: The figure and axis objects
        """
        fig, ax = plt.subplots()

        # If no query_box is provided, use the full bounds of the dataset
        if query is None:
            # Create a BoundingBox that covers the entire dataset
            # Assuming self.bounds returns the full bounds of the index
            query = BoundingBox(
                minx=float('-inf'),
                maxx=float('inf'),
                miny=float('-inf'),
                maxy=float('inf'),
                mint=0,
                maxt=sys.maxsize,
            )

        # Query the R-tree to get matching coordinates
        try:
            hits = self.index.intersection(tuple(query), objects=True)

            # Extract coordinates from the R-tree query results
            x_coords = []
            y_coords = []

            for hit in hits:
                bbox = hit.bbox
                # The coordinates in the R-tree are (x, x, y, y, mint, maxt)
                x_coords.append(bbox[0])  # minx = maxx in our case (point data)
                y_coords.append(bbox[2])  # miny = maxy in our case (point data)

            # Plot the points
            ax.scatter(x_coords, y_coords, color='red')

            # Set labels and title
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('EDDMapS Observations')

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)

        except Exception as e:
            plt.close(fig)
            raise Exception(f'Error querying R-tree: {e!s}')

        plt.tight_layout()
        return fig, ax
