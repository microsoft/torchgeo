# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from collections.abc import Iterator

import pytest
import torch
from _pytest.fixtures import SubRequest


@pytest.fixture(params=[True, False])
def features_only(request: SubRequest) -> bool:
    return bool(request.param)


@pytest.fixture(autouse=True)
def set_precision(request: SubRequest) -> Iterator[None]:
    previous_dtype = torch.get_default_dtype()
    group = request.node.get_closest_marker('xdist_group')

    if (
        os.environ.get('TORCHGEO_USE_BFLOAT16') == 'true'
        and group is not None
        and group.args == ('memory_intensive',)
    ):
        torch.set_default_dtype(torch.bfloat16)

    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)
