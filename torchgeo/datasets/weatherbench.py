# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WeatherBench datasets."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .utils import Path, lazy_import

#: Public WeatherBench 2 ERA5 store on Google Cloud Storage.
DEFAULT_STORE = (
    'gs://weatherbench2/datasets/era5/'
    '1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
)


class WeatherBench2:
    """`WeatherBench 2 <https://sites.research.google/gr/weatherbench/>`__ ERA5.

    Thin wrapper around :func:`xarray.open_zarr` that exposes the underlying
    :class:`xarray.Dataset` via :attr:`data`. Use ``dataset.data.sel(...)`` for
    spatial/temporal slicing and :meth:`plot` for quick visualization.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2308.15560

    .. versionadded:: 0.8
    """

    def __init__(
        self, paths: Path = DEFAULT_STORE, storage_options: dict[str, Any] | None = None
    ) -> None:
        """Initialize a new WeatherBench2 instance.

        Args:
            paths: a local Zarr store or a remote URI (e.g. ``gs://...``).
                Defaults to the public WeatherBench 2 ERA5 store on GCS.
            storage_options: keyword arguments forwarded to
                :func:`xarray.open_zarr`. ``gs://`` paths default to
                ``{'token': 'anon'}`` for anonymous access to public buckets.

        Raises:
            DatasetNotFoundError: If the Zarr store cannot be opened.
            DependencyNotFoundError: If xarray or zarr is not installed.
        """
        xr = lazy_import('xarray')
        lazy_import('zarr')
        self.paths = paths
        if storage_options is None and str(paths).startswith('gs://'):
            storage_options = {'token': 'anon'}
        self.storage_options = storage_options
        try:
            self.data = xr.open_zarr(paths, storage_options=storage_options)
        except (OSError, ValueError, KeyError) as e:
            raise DatasetNotFoundError(self) from e  # ty: ignore[invalid-argument-type]

    def plot(
        self,
        variable: str,
        time: str | pd.Timestamp,
        level: float | None = None,
        region: tuple[float, float, float, float] | None = None,
        ax: Axes | None = None,
        suptitle: str | None = None,
        **imshow_kwargs: Any,
    ) -> Figure:
        """Plot a single variable at a single timestamp.

        Args:
            variable: name of a data variable in :attr:`data`.
            time: timestamp to select along the ``time`` axis.
            level: pressure level to select along the ``level`` axis (required
                for atmospheric variables).
            region: optional ``(xmin, ymin, xmax, ymax)`` longitude/latitude
                bounding box to restrict the plot to.
            ax: optional matplotlib :class:`~matplotlib.axes.Axes` to draw into.
            suptitle: optional suptitle to use for the figure.
            imshow_kwargs: extra keyword arguments forwarded to
                :meth:`xarray.plot.imshow` (e.g. ``cmap``, ``vmin``, ``vmax``).

        Returns:
            A matplotlib Figure.
        """
        da = self.data[variable].sel(time=time)
        if level is not None:
            da = da.sel(level=level)
        if region is not None:
            xmin, ymin, xmax, ymax = region
            # WeatherBench 2 latitude is descending; slice ymax -> ymin.
            da = da.sel(longitude=slice(xmin, xmax), latitude=slice(ymax, ymin))
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        da.plot.imshow(ax=ax, **imshow_kwargs)
        if suptitle is not None:
            fig.suptitle(suptitle)
        return fig  # ty: ignore[invalid-return-type]
