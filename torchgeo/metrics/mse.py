# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Mean Squared Error (MSE)."""

from collections.abc import Sequence
from typing import Any

import einops
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Metric

from .utils import latitude_weights


class MeanSquaredError(Metric):
    r"""Compute mean squared error (MSE).

    .. math::

       \text{MSE}_l = \frac{1}{TIJ} \sum_t^T \sum_i^I \sum_j^J w(i)\left(f_{t,l,i,j} - o_{t,i,j}\right)^2

    Where:

    =========  ====================  =================
    Symbol     Range                 Description
    =========  ====================  =================
    :math:`f`                        Forecast
    :math:`o`                        Observation
    :math:`w`                        Latitude weight
    :math:`t`  :math:`1, \ldots, T`  Verification time
    :math:`l`  :math:`1, \ldots, L`  Lead time
    :math:`i`  :math:`1, \ldots, I`  Latitude index
    :math:`j`  :math:`1, \ldots, J`  Longitude index
    =========  ====================  =================

    For more details on this formulation, see equation 2 in:

    * https://arxiv.org/abs/2308.15560

    .. versionadded:: 0.11
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound = 0.0

    sse: Tensor
    total: Tensor

    def __init__(
        self,
        latitude: Tensor,
        num_outputs: tuple[int, ...],
        squared: bool = True,
        **kwargs: Any,
    ) -> None:
        r"""Initialize a new MeanSquaredError instance.

        Args:
            latitude: Latitude of :math:`f` and :math:`o` in degrees
                :math:`[-90, \ldots, 90]`.
            num_outputs: Dimensions of output metric after mean across
                :math:`t`, :math:`i`, and :math:`j` dimensions.
            squared: If True returns MSE value, if False returns RMSE value.
        """
        super().__init__(**kwargs)

        self.weights = einops.rearrange(latitude_weights(latitude), 'y -> y 1')
        self.squared = squared

        self.add_state('sse', default=torch.zeros(num_outputs), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Forecast of shape [:math:`B`, ..., :math:`H`, :math:`W`].
            target: Observations of shape [:math:`B`, ..., :math:`H`, :math:`W`].
        """
        self.sse += torch.sum(self.weights * (preds - target) ** 2, dim=[0, -2, -1])
        self.total += preds.shape[0] * preds.shape[-2] * preds.shape[-1]

    def compute(self) -> Tensor:
        """Compute mean squared error over state.

        Returns:
            MSE or RMSE.
        """
        mse = self.sse / self.total
        if self.squared:
            return mse
        else:
            return torch.sqrt(mse)

    def plot(
        self, val: Tensor | Sequence[Tensor] | None = None, ax: Axes | None = None
    ) -> tuple[Figure, Axes]:
        """Plot a single or multiple values from the metric.

        Args:
            val: The output of :meth:`MeanSquaredError.compute` or a list of these
                results. If *val* is not provided, will automatically call
                :meth:`MeanSquaredError.compute` and plot that result.
            ax: A matplotlib axis object. If provided will add plot to that axis.

        Returns:
            The resulting figure and axis.
        """
        return self._plot(val, ax)  # ty: ignore[invalid-return-type]
