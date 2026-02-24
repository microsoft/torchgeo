# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""SedonaDB-based vector dataset implementation."""

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely import Geometry

from .geo import VectorDataset
from .utils import GeoSlice, Sample, lazy_import


class SedonaDBDataset(VectorDataset):
    """Vector dataset using SedonaDB for geospatial operations.

    This class inherits from :class:`VectorDataset` but replaces all GeoPandas
    operations with SedonaDB equivalents for improved performance on large datasets.

    Args:
        *args: Arguments passed to :class:`VectorDataset`.
        **kwargs: Keyword arguments passed to :class:`VectorDataset`.

    Raises:
        DatasetNotFoundError: If dataset is not found.
        DependencyNotFoundError: If sedonadb is not installed.
        ValueError: If task is not one of allowed tasks

    .. note::
        This dataset requires the following additional library to be installed:

        * `sedonadb <https://github.com/apache/sedona-db>`_ to load the dataset
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a new SedonaDBDataset instance.

        All arguments are passed to :class:`VectorDataset`.
        """
        lazy_import('sedonadb')
        super().__init__(*args, **kwargs)

    def filter_index(self, index: GeoSlice) -> list[tuple[Geometry, np.int32]]:
        """Filter the index to the given query.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            list of tuples of (geometry, label)
        """
        query = index
        x, y, t = self._disambiguate_slice(query)
        interval = pd.Interval(t.start, t.stop)
        index_df = self.index.iloc[self.index.index.overlaps(interval)]
        index_df = index_df.iloc[:: t.step]
        index_df = index_df.cx[x.start : x.stop, y.start : y.stop]

        if index_df.empty:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        sedona_db = lazy_import('sedonadb')
        sd = sedona_db.connect()

        shapes = []
        for filepath in index_df.filepath:
            if str(filepath).endswith('.parquet'):
                source_df = sd.read_parquet(filepath)
                src_crs = gpd.read_parquet(filepath).crs
            else:
                options: dict[str, str] = {}
                if self.layer is not None:
                    options['layer'] = str(self.layer)
                source_df = sd.read_pyogrio(filepath, options=options)
                src_crs = gpd.read_file(filepath, layer=self.layer).crs

            src_crs = pyproj.CRS.from_user_input(src_crs or self.crs)
            transformer = pyproj.Transformer.from_crs(self.crs, src_crs, always_xy=True)
            (minx, miny) = transformer.transform(x.start, y.start)
            (maxx, maxy) = transformer.transform(x.stop, y.stop)
            query_wkt = shapely.to_wkt(shapely.box(minx, miny, maxx, maxy))
            src_crs_json = src_crs.to_json().replace("'", "''")
            geometry_col = (
                'wkb_geometry' if 'wkb_geometry' in source_df.columns else 'geometry'
            )

            source_df.to_view('temp_df', overwrite=True)
            label_select = f'temp_df.{self.label_name},' if self.label_name else ''
            filtered_df = sd.sql(
                f"""
                SELECT {label_select} {geometry_col} as geometry
                FROM temp_df
                WHERE ST_Intersects(
                    ST_SetCrs({geometry_col}, '{src_crs_json}'),
                    ST_SetCrs(ST_GeomFromWKT('{query_wkt}'), '{src_crs_json}')
                )
                """
            )

            filtered_gdf = filtered_df.to_pandas()
            if len(filtered_gdf) > 0:
                filtered_gdf = filtered_gdf.set_geometry('geometry', crs=src_crs)
                filtered_gdf = filtered_gdf.to_crs(self.crs)

                labels = np.array(
                    [self.get_label(row) for _, row in filtered_gdf.iterrows()]
                ).astype(np.int32)

                shapes.extend(list(zip(filtered_gdf.geometry, labels)))

        return shapes

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve a sample and include source CRS metadata."""
        sample = super().__getitem__(index)
        sample['crs'] = self.crs
        return sample
