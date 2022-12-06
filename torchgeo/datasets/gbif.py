# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for the Global Biodiversity Information Facility."""

import glob
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
from rasterio.crs import CRS

from .geo import GeoDataset
from .utils import BoundingBox


def _disambiguate_timestamps(
    year: float, month: float, day: float
) -> Tuple[float, float]:
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

        files = glob.glob(os.path.join(root, "**.csv"))
        if not files:
            raise FileNotFoundError(f"Dataset not found in `root={self.root}`")

        try:
            import pandas as pd  # noqa: F401
        except ImportError:
            raise ImportError(
                "pandas is not installed and is required to use this dataset"
            )

        # Read tab-delimited CSV file
        data = pd.read_table(
            files[0],
            engine="c",
            usecols=["decimalLatitude", "decimalLongitude", "day", "month", "year"],
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
