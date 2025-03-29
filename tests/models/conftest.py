# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from _pytest.fixtures import SubRequest


@pytest.fixture(params=[True, False])
def features_only(request: SubRequest) -> bool:
    # features_only arg in ViT supported only from timm 1.0.3
    pytest.importorskip('timm', minversion='1.0.3')
    return bool(request.param)
