# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Dataset for EDDMapS."""

import functools
import os
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import torch
from geopandas import GeoDataFrame
from matplotlib.figure import Figure

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
        data = df['ObsDate'].apply(func).to_list()
        index = pd.IntervalIndex.from_tuples(data, closed='both', name='datetime')
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
        index = index.cx[x.start : x.stop, y.start : y.stop]

        if index.empty:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        keypoints = torch.tensor(index.get_coordinates().values, dtype=torch.float32)
        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample = {
            'bounds': self._slice_to_tensor(query),
            'keypoints': keypoints,
            'transform': torch.tensor(transform),
        }

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

        # Extract coordinates
        keypoints = sample['keypoints']
        x = keypoints[:, 0]
        y = keypoints[:, 1]

        # Plot the points
        ax.scatter(x, y)

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
