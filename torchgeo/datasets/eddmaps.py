# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for EDDMapS."""

import os
import sys
from typing import Any, Dict

import numpy as np
from rasterio.crs import CRS

from .geo import GeoDataset
from .utils import BoundingBox, disambiguate_timestamp


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

    .. note::
       This dataset requires the following additional library to be installed:

       * `pandas <https://pypi.org/project/pandas/>`_ to load CSV files

    .. versionadded:: 0.3
    """

    res = 0
    _crs = CRS.from_epsg(4326)  # Lat/Lon

    def __init__(self, root: str = "data") -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found

        Raises:
            FileNotFoundError: if no files are found in ``root``
            ImportError: if pandas is not installed
        """
        super().__init__()

        self.root = root

        filepath = os.path.join(root, "mappings.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found in `root={self.root}`")

        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas is not installed and is required to use this dataset"
            )

        # Read CSV file
        data = pd.read_csv(
            filepath, engine="c", usecols=["ObsDate", "Latitude", "Longitude"]
        )

        # Convert from pandas DataFrame to rtree Index
        i = 0
        for date, y, x in data.itertuples(index=False, name=None):
            # Skip rows without lat/lon
            if np.isnan(y) or np.isnan(x):
                continue

            if not pd.isna(date):
                mint, maxt = disambiguate_timestamp(date, "%m-%d-%y")
            else:
                mint, maxt = 0, sys.maxsize

            coords = (x, x, y, y, mint, maxt)
            self.index.insert(i, coords)
            i += 1

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
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
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        sample = {"crs": self.crs, "bbox": bboxes}

        return sample
