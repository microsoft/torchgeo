# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Dataset for iNaturalist."""

import functools
import glob
import os
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import torch
from geopandas import GeoDataFrame
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import GeoDataset
from .utils import GeoSlice, Path, disambiguate_timestamp


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
        sample = {'crs': self.crs, 'bounds': query, 'keypoints': keypoints}

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
            ax.set_title('iNaturalist Dataset Plot')

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig
