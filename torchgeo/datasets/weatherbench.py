# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WeatherBench datasets."""

from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from pyproj import CRS

from .geo import XarrayDataset
from .utils import Path


class WeatherBench2(XarrayDataset):
    """WeatherBench 2 dataset.

    `WeatherBench <https://sites.research.google/gr/weatherbench/>__ is an open
    framework for evaluating ML and physics-based weather forecasting models in a
    like-for-like fashion.

    This data loader supports several publicly available, cloud-optimized ground-truth
    and baseline
    `datasets <https://weatherbench2.readthedocs.io/en/latest/data-guide.html>`__,
    including a comprehensive copy of the
    `ERA5 <https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3803>`__ dataset
    used for training most ML models.

    See the
    `documentation <https://weatherbench2.readthedocs.io/en/latest/data-guide.html>`__
    for more information on data availability.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2308.15560

    .. versionadded:: 0.8
    """
