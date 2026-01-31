# Copyright (c) TorchGeo contributors.
# Licensed under the MIT License.

from collections import OrderedDict
from pathlib import Path
from typing import Any
import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import GalileoWeights, galileo

SPACE_TIME_BANDS = 14
SPACE_BANDS = 20
TIME_BANDS = 6
STATIC_BANDS = 40


SPACE_TIME_GROUPS = OrderedDict({
    "g0": list(range(SPACE_TIME_BANDS)),
})

SPACE_GROUPS = OrderedDict({
    "g0": list(range(SPACE_BANDS)),
})

TIME_GROUPS = OrderedDict({
    "g0": list(range(TIME_BANDS)),
})

STATIC_GROUPS = OrderedDict({
    "g0": list(range(STATIC_BANDS)),
})


def make_inputs(
    batch: int = 2,
    H: int = 64,
    W: int = 64,
    T: int = 2
) -> dict[str, torch.Tensor]:
    """Generate valid dummy inputs with exact band counts."""
    return {
        "s_t_x": torch.randn(batch, H, W, T, SPACE_TIME_BANDS),
        "sp_x": torch.randn(batch, H, W, SPACE_BANDS),
        "t_x": torch.randn(batch, T, TIME_BANDS),
        "st_x": torch.randn(batch, STATIC_BANDS),

        "s_t_m": torch.zeros(batch, H, W, T, SPACE_TIME_BANDS),
        "sp_m": torch.zeros(batch, H, W, SPACE_BANDS),
        "t_m": torch.zeros(batch, T, TIME_BANDS),
        "st_m": torch.zeros(batch, STATIC_BANDS),

        "months": torch.zeros(batch, T, dtype=torch.long),
    }


@pytest.fixture(params=[*GalileoWeights])
def weights(request: SubRequest) -> WeightsEnum:
    return request.param


@pytest.fixture
def mocked_weights(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    load_state_dict_from_url: Any
) -> GalileoWeights:
    """Create a tiny local checkpoint to test weight loading."""
    w: GalileoWeights.GALILEO_S2_NANO_V1
    path = tmp_path / "dummy_encoder.pth"

    model = galileo(
        variant=w.meta["variant"],
        space_time_groups=SPACE_TIME_GROUPS,
        space_groups=SPACE_GROUPS,
        time_groups=TIME_GROUPS,
        static_groups=STATIC_GROUPS,
    )
    torch.save(model.state_dict(), path)

    monkeypatch.setattr(w.value, "url", str(path))
    return w


@pytest.mark.parametrize("w", list(GalileoWeights))
def test_galileo_forward(w: WeightsEnum) -> None:
    """Ensure the forward pass runs with correct output shape."""
    model = galileo(
        variant=w.meta["variant"],
        space_time_groups=SPACE_TIME_GROUPS,
        space_groups=SPACE_GROUPS,
        time_groups=TIME_GROUPS,
        static_groups=STATIC_GROUPS,
    )
    inputs = make_inputs()
    out = model(**inputs, patch_size=16)

    s_t_x = out[0]
    assert s_t_x.shape[-1] == w.meta["embed_dim"]


def test_invalid_input_shape() -> None:
    """Incorrect band counts must raise an error."""
    model = galileo(
        variant="nano",
        space_time_groups=SPACE_TIME_GROUPS,
        space_groups=SPACE_GROUPS,
        time_groups=TIME_GROUPS,
        static_groups=STATIC_GROUPS,
    )

    bad = make_inputs()
    bad["s_t_x"] = torch.randn(2, 64, 64, 2, 3)

    with pytest.raises(Exception):
        model(**bad, patch_size=16)


def test_galileo_weights(mocked_weights: GalileoWeights) -> None:
    """Verify weight loading from a mocked local checkpoint."""
    w = mocked_weights
    model = galileo(
        weights=w,
        space_time_groups=SPACE_TIME_GROUPS,
        space_groups=SPACE_GROUPS,
        time_groups=TIME_GROUPS,
        static_groups=STATIC_GROUPS,
    )

    inputs = make_inputs(batch=1)
    out = model(**inputs, patch_size=16)

    assert isinstance(out[0], torch.Tensor)


@pytest.mark.slow
@pytest.mark.parametrize("w", list(GalileoWeights))
def test_real_weight_download(w: WeightsEnum) -> None:
    """Verify real HuggingFace weight download works."""
    model = galileo(
        weights=w,
        space_time_groups=SPACE_TIME_GROUPS,
        space_groups=SPACE_GROUPS,
        time_groups=TIME_GROUPS,
        static_groups=STATIC_GROUPS,
    )
    assert isinstance(model, torch.nn.Module)
