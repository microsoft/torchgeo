# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch
from matplotlib import pyplot as plt

from torchgeo.metrics import MeanSquaredError

B = 2
T = 3
Z = 4
Y = 5
X = 6
latitude = torch.tensor([-60.0, -30.0, 0.0, 30.0, 60.0])


@pytest.mark.parametrize('dims', [(), (T,), (T, Z)])
def test_mse(dims: tuple[int, ...]) -> None:
    preds = torch.ones(B, *dims, Y, X)
    target = torch.zeros(B, *dims, Y, X)
    expected = torch.ones(dims)
    metric = MeanSquaredError(latitude, dims)
    metric.update(preds, target)
    mse = metric.compute()
    assert torch.allclose(mse, expected)


@pytest.mark.parametrize('dims', [(), (T,), (T, Z)])
def test_rmse(dims: tuple[int, ...]) -> None:
    preds = torch.rand(B, *dims, Y, X)
    target = torch.rand(B, *dims, Y, X)
    metric1 = MeanSquaredError(latitude, dims, squared=True)
    metric2 = MeanSquaredError(latitude, dims, squared=False)
    metric1.update(preds, target)
    metric2.update(preds, target)
    mse = metric1.compute()
    rmse = metric2.compute()
    assert torch.allclose(torch.sqrt(mse), rmse)


@pytest.mark.parametrize('dims', [(), (T,)])
def test_plot(dims: tuple[int, ...]) -> None:
    preds = torch.rand(B, *dims, Y, X)
    target = torch.rand(B, *dims, Y, X)
    metric = MeanSquaredError(latitude, dims)
    metric.update(preds, target)
    metric.plot()
    plt.close()
