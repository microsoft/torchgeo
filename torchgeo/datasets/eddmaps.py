# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for EDDMapS."""

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from rasterio.crs import CRS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional

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


def plot(
    self,
    query: Optional[BoundingBox] = None,
    title: str = "EDDMapS Dataset",
    point_size: int = 20,
    point_color: str = 'blue',
    query_color: str = 'red',
    annotate: bool = False,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    
    """ Plot the dataset points with optional query bounding box
    Args:
        
        query: The query to look for points, in the form of a bounding box: (minx,maxx,miny,maxy,mint,maxt)
        title: Title of the plot
        point_size: The size of the points plotted
        point_color: The default color of the points, in case no such map is provided
        query_color: color for the points which fall into the query
        annotate: Controls if the points with timestamps are annotated
        figsize: Size of drawn figure in the shape: (width, height)
        
        Raises:

    ValueError: When no points could be plotted because none were valid.

    """
    
    # Filtering valid lat and long rows
    valid_data = self.data.dropna(subset = [ 'Latitude' , 'Longitude'])
    if valid_data.empty:
        raise ValueError("No valid lat/long data to plot.")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot-at-all points

    ax.scatter(

    valid_data['Longitude'],

    valid_data['Latitude'],

    s = point_size,

    c = point_color,

    edgecolor = 'k',

    alpha = 0.6,

    label = 'All Observations'

    )

    #highlighting queried points (If) provided bounding box query

    if query:
        minx, maxx, miny, maxy, mint, maxt = query
        hits = self.index.intersection((minx,maxx,miny,maxy,mint, maxt))
                  
    # Get coordinates of hits to highlight
    query_points = valid_data.iloc[[list(hits)]]
    ax.scatter(
        query_points['Longitude'],
        query_points['Latitude'],
        s = point_size * 1.5,
        c = query_color,
        edgecolor = 'white',
        alpha = 0.8,
        label = 'Query Results'
              )
    
    # Draw a bounding box
    bbox_patch = patches.rectangle(
        (minx, miny), maxx - minx, maxy - miny,
        linewidth = 2, edgecolor = 'red', facecolor='none', linestyle = '--', label = "Query Bounding Box"
    )
    ax.add_patch(bbox_patch)
    
    # Optional annotations
    if annotate:
        for _, row in valid_data.iterrows():
            ax.annotate(
                row['ObsDate'], (row['Longitude'], row['Latitude']),
                fontsize=8, alpha=0.7, textcoords="offset points", xytext=(0, 5), ha='center'
            )

    # Plot styling
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    plt.show()


    
