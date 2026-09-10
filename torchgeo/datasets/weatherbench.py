# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WeatherBench datasets."""

import math
from typing import ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from .geo import XarrayDataset
from .utils import Sample


class WeatherBench2(XarrayDataset):
    """WeatherBench 2 dataset.

    `WeatherBench <https://sites.research.google/gr/weatherbench/>__ is an open
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

    * `zarr <https://pypi.org/project/zarr/>`_: to load Zarr files.
    * `gcsfs <https://pypi.org/project/gcsfs/>`_: if loading data directly from GCS.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2308.15560

    .. versionadded:: 0.11
    """

    cmaps: ClassVar[dict[str, str]] = {
        'temperature': 'hot',
        'precipitation': 'gist_ncar',
        'flux': 'plasma',
        'orography': 'terrain',
        'wind': 'jet',
    }

    # https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation
    units: ClassVar[dict[str, str]] = {
        'geopotential': 'm$^2$/s$^2$',
        'temperature': 'K',
        'specific_humidity': 'kg/kg',
        'relative_humidity': '%',
        'wind': 'm/s',
        'vorticity': '1/s',
        'potential_vorticity': 'K m$^2$/kg s',
        'precipitation': 'm',
        'angle': 'rad',
        'height': 'm',
        'divergence': '1/s',
        'leaf': 'm$^2$/m$^2$',
        'pressure': 'Pa',
        'flux': 'W/m$^2$',
        'moisture_divergence': 'kg/m$^2$s',
        'snow_depth': 'm of water equivalent',
        'total_column': 'kg/m$^2$',
        'vertical_velocity': 'Pa/s',
        'volumetric': 'm$^3$/m$^3$',
    }

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
        # Average across time: T C H W -> C H W
        data = torch.mean(sample['image'], dim=0)

        nvars = len(self.data_vars)
        ncols = math.ceil(math.sqrt(nvars))
        nrows = math.ceil(nvars / ncols)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False
        )
        axes = axes.ravel()

        for i, var in enumerate(self.data_vars):
            if show_titles:
                axes[i].set_title(var.replace('_', ' '))

            # Image
            cmap = 'viridis'
            for key, value in self.cmaps:
                if key in var:
                    cmap = value

            im = axes[i].imshow(data[i], cmap=cmap)

            # Colorbar
            cbar = fig.colorbar(im, ax=axes[i])
            for key, value in self.units:
                if key in var:
                    cbar.set_label(value)

        # Hide unused axes
        for ax in axes[nvars:]:
            ax.set_visible(False)

        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout()
        return fig
