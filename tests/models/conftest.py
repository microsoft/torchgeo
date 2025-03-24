import pytest
from _pytest.fixtures import SubRequest


@pytest.fixture(params=[True, False])
def features_only(request: SubRequest) -> bool:
    if request.param:
        pytest.importorskip('timm', minversion='1.0.3')
    return bool(request.param)
