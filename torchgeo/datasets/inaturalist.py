# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for iNaturalist."""

import glob
import os
import sys
from typing import Any

import pandas as pd
from rasterio.crs import CRS

import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib import cm
import numpy as np

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

    res = 0
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



    def plot(self, query: BoundingBox = None, time_range: tuple = None, cmap: str = 'viridis') -> None:
        """ Plot the observations in a map when given a geographical bounding box as well as time frame which is optional.

        Args:
            query: Optional case for bounding box which is meant to cut the observations (minx,maxx,miny,maxy,mint,maxt).
            time_range: Optional filtering in terms of exact date and location for (start_time, end_time).
            cmap: A color map for the time sequence.
        """
        # Step 1: If a query and/or time_range is provided, filter the dataset based on this criteria. 
        data = self._filter_data(query, time_range)

        # Step 2: Prepare a GeoDataFrame for geospatial visualization by geographic reference.
        gdf = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
            crs=self._crs
        )

        
        # Step 3: Illustrate the observations that were made.
        fig, ax = plt.subplots(figsize=(10, 8))
        gdf.plot(ax=ax, color=self._get_color_by_time(data, cmap=cmap), markersize=10, alpha=0.7)


        # Step 4: Incorporate cartographic refinements (for example, coastlines, gridlines) to the map.
        ax.set_title('iNaturalist Observations', fontsize=16)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)

        # Insert a colorbar to show the progression of time 
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data['observed_on'].min(), vmax=data['observed_on'].max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Time of Observation')
        plt.show()

    def _filter_data(self, query: BoundingBox, time_range: tuple) -> pd.DataFrame:
        """This is a Helper function that helps to filter the dataset based on the bounding box provided by the query and a time range."""      
        # First stage filter on bounding box
        if query:
            data = self._filter_by_bbox(query)
        else:
            data = self._load_data()

        # Now we filter on time range
        if time_range:
            data = data[(data['observed_on'] >= time_range[0]) & (data['observed_on'] <= time_range[1])]
        return data
    

    def _get_color_by_time(self, data: pd.DataFrame, cmap: str) -> np.ndarray:
        """Creates a mapping of time with color."""
        norm = plt.Normalize(vmin=data['observed_on'].min(), vmax=data['observed_on'].max())
        colormap = cm.get_cmap(cmap)
        return colormap(norm(data['observed_on']))
    

    def _filter_by_bbox(self, query: BoundingBox) -> pd.DataFrame:
        """Helper function that broadens filters with bounding box type parameters to filter the data frame."""
        minx, maxx, miny, maxy, _, _ = query
        data = self._load_data()
        return data[(data['longitude'] >= minx) & (data['longitude'] <= maxx) & 
                    (data['latitude'] >= miny) & (data['latitude'] <= maxy)]

    def _load_data(self) -> pd.DataFrame:
        """Tries to get dataset from the CSV file."""      
        files = glob.glob(os.path.join(self.root, '**.csv'))
        if not files:
            raise DatasetNotFoundError(self)            
        
        data = pd.read_csv(
            files[0],
            engine='c',
            usecols=['observed_on', 'latitude', 'longitude'],
        )
        return data



