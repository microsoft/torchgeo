# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
from _pytest.fixtures import SubRequest


@pytest.fixture(params=[True, False])
def features_only(request: SubRequest) -> bool:
    return bool(request.param)
