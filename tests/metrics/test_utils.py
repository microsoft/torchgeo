# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.metrics.utils import latitude_weights


def test_latitude_weights() -> None:
    latitude = torch.tensor([-60.0, -30.0, 0.0, 30.0, 60.0])
    expected = torch.tensor([0.7322330, 1.1207193, 1.2940952, 1.1207193, 0.7322330])
    weights = latitude_weights(latitude)
    assert torch.allclose(weights, expected)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
def test_latitude_weights_dtype(dtype: torch.dtype) -> None:
    latitude = torch.tensor([-60.0, -30.0, 0.0, 30.0, 60.0], dtype=dtype)
    weights = latitude_weights(latitude)
    assert latitude.dtype == weights.dtype == dtype


def test_latitude_weights_descending() -> None:
    latitude = torch.tensor([60.0, 30.0, 0.0, -30.0, -60.0])
    with pytest.raises(AssertionError, match='latitude must be in ascending order'):
        latitude_weights(latitude)
