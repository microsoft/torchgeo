from typing import Any, Dict

import pytest

from torchgeo.trainers import CycloneSimpleRegressionTask


def test_cyclone_args() -> None:

    invalid_model = "fakemodel"
    error_message = f"Model type '{invalid_model}' is not valid."
    kwargs: Dict[str, Any] = {"model": invalid_model}
    with pytest.raises(ValueError, match=error_message):
        CycloneSimpleRegressionTask(**kwargs)
