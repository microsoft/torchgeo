# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.datasets.utils import quantile_normalization


def test_quantile_normalization_integer() -> None:
    img = torch.arange(1, 101, dtype=torch.int16).reshape(10, 10)

    result = quantile_normalization(img)

    values = img.float().flatten()
    lower = torch.quantile(values, 0.02, interpolation='higher')
    upper = torch.quantile(values, 0.98, interpolation='lower')
    expected = torch.clamp(
        (values.reshape_as(img) - lower) / (upper - lower + 1e-5), 0, 1
    )

    assert result.dtype == torch.float32
    assert torch.allclose(result, expected)


def test_quantile_normalization_without_torch_quantile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def size_limit(*args: object, **kwargs: object) -> None:
        raise RuntimeError('quantile() input tensor is too large')

    monkeypatch.setattr(torch, 'quantile', size_limit)
    img = torch.arange(1, 101, dtype=torch.float32).reshape(10, 10)

    result = quantile_normalization(img)

    assert 0 <= result.min() <= result.max() <= 1
