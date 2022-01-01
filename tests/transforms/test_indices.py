# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Dict

import pytest
import torch
from torch import Tensor

from torchgeo.transforms import indices


@pytest.fixture
def sample() -> Dict[str, Tensor]:
    return {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
    }


@pytest.fixture
def batch() -> Dict[str, Tensor]:
    return {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[0, 0, 1], [0, 1, 1], [1, 1, 1]]]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
    }


def test_append_index_sample(sample: Dict[str, Tensor]) -> None:
    c, h, w = sample["image"].shape
    tr = indices.AppendNormalizedDifferenceIndex(index_a=0, index_b=0)
    output = tr(sample)
    assert output["image"].shape == (c + 1, h, w)


def test_append_index_batch(batch: Dict[str, Tensor]) -> None:
    b, c, h, w = batch["image"].shape
    tr = indices.AppendNormalizedDifferenceIndex(index_a=0, index_b=0)
    output = tr(batch)
    assert output["image"].shape == (b, c + 1, h, w)
