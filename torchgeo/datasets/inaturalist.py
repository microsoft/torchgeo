# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Dataset for iNaturalist."""

import glob
import os
import sys
from typing import Any, Dict

from rasterio.crs import CRS

from .geo import GeoDataset
from .utils import BoundingBox, disambiguate_timestamp


class INaturalist(GeoDataset):
    """Dataset for iNaturalist.

    `iNaturalist <https://www.inaturalist.org/>`_ is a joint initiative of the
    California Academy of Sciences and the National Geographic Society. It allows
    citizen scientists to upload observations of organisms that can be downloaded by
    scientists and researchers.

    If you use an iNaturalist dataset in your research, please cite it according to:

    * https://www.inaturalist.org/pages/help#cite

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

        # Read CSV file
        data = pd.read_csv(
            files[0],
            engine="c",
            usecols=["observed_on", "time_observed_at", "latitude", "longitude"],
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
                mint, maxt = disambiguate_timestamp(time, "%Y-%m-%d %H:%M:%S %z")
            elif not pd.isna(date):
                mint, maxt = disambiguate_timestamp(date, "%Y-%m-%d")
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
