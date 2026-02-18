# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Dataset for the Global Biodiversity Information Facility."""

import functools
import glob
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import torch
from geopandas import GeoDataFrame
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import GeoDataset
from .utils import GeoSlice, Path, Sample, disambiguate_timestamp


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
        df = df[df['decimalLatitude'].notna()]
        df = df[df['decimalLongitude'].notna()]
        df['day'] = df['day'].str.zfill(2)
        df['month'] = df['month'].str.zfill(2)
        date = df['day'] + ' ' + df['month'] + ' ' + df['year']

        # Convert from pandas DataFrame to geopandas GeoDataFrame
        func = functools.partial(disambiguate_timestamp, format='%d %m %Y')
        index = pd.IntervalIndex.from_tuples(
            date.apply(func).to_list(), closed='both', name='datetime'
        )
        geometry = gpd.points_from_xy(df['decimalLongitude'], df['decimalLatitude'])
        self.index = GeoDataFrame(index=index, geometry=geometry, crs='EPSG:4326')

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f'index: {index} not found in dataset with bounds: {self.bounds}'
            )

        keypoints = torch.tensor(df.get_coordinates().values, dtype=torch.float32)
        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample = {
            'bounds': self._slice_to_tensor(index),
            'keypoints': keypoints,
            'transform': torch.tensor(transform),
        }

        return sample

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
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
            ax.set_title('GBIF Occurrence Locations by Date')

        if suptitle is not None:
            fig.suptitle(suptitle)

        fig.tight_layout()
        return fig
