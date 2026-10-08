# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WeatherBench datasets."""

import math
from collections.abc import Callable, Sequence

import shapely
import torch
from geopandas import GeoDataFrame
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import IntervalIndex, Timestamp
from pyproj import CRS

from .geo import GeoDataset, GeoSlice
from .utils import Path, Sample, lazy_import

# https://microsoft.github.io/aurora/batch.html
_AURORA_VARS = (
    # Surface-level variables
    '2m_temperature',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'mean_sea_level_pressure',
    # Static variables
    'land_sea_mask',
    'soil_type',
    'geopotential_at_surface',
    # Atmospheric variables
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_humidity',
    'geopotential',
)


class WeatherBench2(GeoDataset):
    """WeatherBench 2 dataset.

    `WeatherBench <https://sites.research.google/gr/weatherbench/>`__ is an open
    framework for evaluating ML and physics-based weather forecasting models in a
    like-for-like fashion.

    This data loader supports several publicly available, cloud-optimized
    ground-truth and baseline datasets including a comprehensive copy of the
    `ERA5 <https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3803>`__
    dataset used for training most ML models.

    See the
    `data guide <https://weatherbench2.readthedocs.io/en/latest/data-guide.html>`__
    for more information on data availability.

    Requires the following additional dependencies:

    * `gcsfs <https://pypi.org/project/gcsfs/>`_: if loading data directly from GCS.
    * `rioxarray <https://pypi.org/project/rioxarray/>`_: to support geospatial operations.
    * `xarray` <https://pypi.org/project/xarray/>`_: to load an Xarray dataset.
    * `zarr <https://pypi.org/project/zarr/>`_: to load Zarr files.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2308.15560

    .. versionadded:: 0.11

    .. warning::
       This dataset does not yet support reprojection or resampling. If you use this
       dataset in combination with other datasets via intersection or union, please
       ensure that this dataset comes first or that other datasets are manually
       reprojected or resampled to match this dataset.
    """

    def __init__(
        self,
        store: Path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',
        *,
        data_vars: Sequence[str] | None = _AURORA_VARS,
        level: float | slice | Sequence[float] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
    ) -> None:
        """Initialize a new WeatherBench2 instance.

        Args:
            store: Zarr store to load.
            data_vars: List of data variables to load (defaults to all variables).
            level: Atmospheric level(s) to load (defaults to all levels).
            transforms: A function/transform that takes an input sample
                and returns a transformed version.

        Raises:
            DependencyNotFoundError: If rioxarray or xarray is not installed.
        """
        lazy_import('rioxarray')
        xr = lazy_import('xarray')

        self.data = xr.open_zarr(store)
        self.data_vars = data_vars or list(self.data.data_vars.keys())
        self.level = level or list(self.data.level.values)
        self.transforms = transforms

        # CRS is missing from file
        crs = CRS.from_epsg(4326)
        self.data = self.data.rio.write_crs(crs)

        # Transform is inverted in xarray
        res = self.data.rio.resolution()
        self._res = (res[0], -res[1])

        tmin = self.data.time.values.min()
        tmax = self.data.time.values.max()

        filepaths = [store]
        datetimes = [(Timestamp(tmin), Timestamp(tmax))]
        geometries = [shapely.box(*self.data.rio.bounds())]

        data = {'filepath': filepaths}
        index = IntervalIndex.from_tuples(datetimes, closed='both', name='datetime')
        self.index = GeoDataFrame(data, index=index, geometry=geometries, crs=crs)

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            index: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.
        """
        x, y, t = self._disambiguate_slice(index)

        # Step size must be integer multiple of pixels
        x = slice(x.start, x.stop, int(x.step // self.res[0]))
        y = slice(y.start, y.stop, int(y.step // self.res[1]))

        # Latitude dimension must be inverted
        y = slice(y.stop, y.start, y.step)

        data = self.data.sel(time=t, latitude=y, longitude=x, level=self.level)

        # https://microsoft.github.io/aurora/batch.html#batch-metadata
        sample = {
            'time': torch.tensor(self.data.time.values.astype(float)),
            'latitude': torch.tensor(self.data.latitude.values),
            'longitude': torch.tensor(self.data.longitude.values),
            'level': torch.tensor(self.data.level.values),
        }
        for var in self.data_vars:
            sample[var] = torch.tensor(data[var].values)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        nvars = len(self.data_vars)
        ncols = math.ceil(math.sqrt(nvars))
        nrows = math.ceil(nvars / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 3 * nrows), squeeze=False
        )
        axes = axes.ravel()

        for i, var in enumerate(self.data_vars):
            axes[i].axis('off')
            if show_titles:
                axes[i].set_title(self.data[var].attrs.get('long_name', var))

            # Image
            image = sample[var]
            match self.data[var].ndim:
                case 3:
                    image = torch.mean(image, dim=0)  # T Y X -> Y Z
                case 4:
                    image = torch.mean(image, dim=(0, 1))  # T Z Y X -> Y X

            im = axes[i].imshow(image)

            # Colorbar
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes('right', size='5%', pad=0.15)
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label(self.data[var].attrs.get('units', ''))

        # Hide unused axes
        for ax in axes[nvars:]:
            ax.set_visible(False)

        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout()
        return fig
