# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Iterator

import pytest
import torch
from _pytest.fixtures import SubRequest


@pytest.fixture(params=[True, False])
def features_only(request: SubRequest) -> bool:
    return bool(request.param)


@pytest.fixture
def use_bfloat16() -> Iterator[None]:
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(dtype)
