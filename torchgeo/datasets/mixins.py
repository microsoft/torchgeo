# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Mixins for dataset classes."""

from matplotlib.colors import Colormap


class PlottingMixin:
    """Mixin for dataset plotting.

    .. versionadded:: 0.10
    """

    #: Color map for the dataset, used for plotting
    cmap: str | Colormap | None = None

    #: Names of all available bands in the dataset
    all_bands: tuple[str, ...] = ()

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: tuple[str, ...] = ()
