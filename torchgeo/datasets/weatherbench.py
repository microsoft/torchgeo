# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WeatherBench datasets."""

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import shapely
import torch
from geopandas import GeoDataFrame
from matplotlib.figure import Figure
from pandas import IntervalIndex, Timestamp
from pyproj import CRS

from .geo import GeoDataset, GeoSlice
from .utils import Path, Sample, lazy_import


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
    * `xarray` <https://pypi.org/project/xarray/>`_: to load an Xarray dataset.
    * `zarr <https://pypi.org/project/zarr/>`_: to load Zarr files.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2308.15560

    .. versionadded:: 0.11
    """

    _res = (0.25, 0.25)

    def __init__(
        self,
        store: Path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',
        *,
        data_vars: Sequence[str] | None = None,
    ) -> None:
        """Initialize a new WeatherBench2 instance.

        Args:
            store: Zarr store to load.
            data_vars: List of data variables to load (defaults to all variables).

        Raises:
            DependencyNotFoundError: If xarray is not installed.
        """
        xr = lazy_import('xarray')

        self.data = xr.open_zarr(store)
        self.data_vars = data_vars or list(self.data.data_vars.keys())

        xmin = self.data.longitude.values.min()
        xmax = self.data.longitude.values.max()
        ymin = self.data.latitude.values.min()
        ymax = self.data.latitude.values.max()
        tmin = self.data.time.values.min()
        tmax = self.data.time.values.max()

        filepaths = [store]
        datetimes = [(Timestamp(tmin), Timestamp(tmax))]
        geometries = [shapely.box(xmin, ymin, xmax, ymax)]

        data = {'filepath': filepaths}
        index = IntervalIndex.from_tuples(datetimes, closed='both', name='datetime')
        crs = CRS.from_epsg(4326)
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
        x = slice(x.start, x.stop, int(x.step // 0.25))
        y = slice(y.start, y.stop, int(y.step // 0.25))

        # Latitude dimension must be inverted
        y = slice(y.stop, y.start, y.step)

        data = self.data.sel(time=t, latitude=y, longitude=x)

        masks = []  # C Y X
        images = []  # C T Y X
        videos = []  # C T Z Y X
        for var in self.data_vars:
            match self.data[var].ndim:
                case 2:
                    masks.append(torch.tensor(data[var].values))
                case 3:
                    images.append(torch.tensor(data[var].values))
                case 4:
                    videos.append(torch.tensor(data[var].values))

        sample = {}
        if masks:
            sample['mask'] = torch.stack(masks, dim=0)  # C Y X
        if images:
            sample['image'] = torch.stack(images, dim=1)  # T C Y X
        if videos:
            sample['video'] = torch.stack(videos, dim=1)  # T C Z Y X

        return sample

    def plot(
        self, sample: Sample, show_titles: bool = True, suptitle: str | None = None
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by :meth:`XarrayDataset.__getitem__`.
            show_titles: Flag indicating whether to show titles above each panel.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        nvars = len(self.data_vars)
        ncols = math.ceil(math.sqrt(nvars))
        nrows = math.ceil(nvars / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False
        )
        axes = axes.ravel()

        mask_id = 0
        image_id = 0
        video_id = 0
        for i, var in enumerate(self.data_vars):
            if show_titles:
                axes[i].set_title(self.data[var].attrs.get('long_name', var))

            # Image/mask
            match self.data[var].ndim:
                case 2:
                    image = sample['mask'][mask_id]
                    mask_id += 1
                case 3:
                    image = sample['image'][:, image_id]
                    image = torch.mean(image, dim=0)  # T Y X -> Y Z
                    image_id += 1
                case 4:
                    image = sample['video'][:, video_id]
                    image = torch.mean(image, dim=(0, 1))  # T Z Y X -> Y X
                    video_id += 1

            im = axes[i].imshow(image)

            # Colorbar
            cbar = fig.colorbar(im, ax=axes[i])
            cbar.set_label(self.data[var].attrs.get('units', ''))

        # Hide unused axes
        for ax in axes[nvars:]:
            ax.set_visible(False)

        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout()
        return fig
