# Copyright (c) TorchGeo contributors.
# Licensed under the MIT License.

import pytest
import torch
from pathlib import Path
from pytest import MonkeyPatch
from _pytest.fixtures import SubRequest
from torchvision.models._api import WeightsEnum

from torchgeo.models import galileo, GalileoWeights


SPACE_TIME_BANDS = 14
SPACE_BANDS = 20
TIME_BANDS = 6
STATIC_BANDS = 40



def make_inputs(batch=2, H=64, W=64, T=2):
    """Generate valid dummy inputs with exact band counts."""
    return dict(
        s_t_x=torch.randn(batch, H, W, T, SPACE_TIME_BANDS),
        sp_x=torch.randn(batch, H, W, SPACE_BANDS),
        t_x=torch.randn(batch, T, TIME_BANDS),
        st_x=torch.randn(batch, STATIC_BANDS),

        s_t_m=torch.zeros(batch, H, W, T, SPACE_TIME_BANDS),
        sp_m=torch.zeros(batch, H, W, SPACE_BANDS),
        t_m=torch.zeros(batch, T, TIME_BANDS),
        st_m=torch.zeros(batch, STATIC_BANDS),

        months=torch.zeros(batch, T, dtype=torch.long),
    )


@pytest.fixture(params=[*GalileoWeights])
def weights(request: SubRequest) -> WeightsEnum:
    return request.param


@pytest.fixture
def mocked_weights(tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url):
    """Create a tiny local checkpoint to test weight loading."""
    w = GalileoWeights.GALILEO_S2_NANO_V1
    path = tmp_path / "dummy_encoder.pth"

  
    model = galileo(variant=w.meta["variant"])
    torch.save(model.state_dict(), path)


    monkeypatch.setattr(w.value, "url", str(path))

    return w


@pytest.mark.parametrize("w", list(GalileoWeights))
def test_galileo_forward(w):
    """Ensure the forward pass runs with correct output shape."""
    model = galileo(variant=w.meta["variant"])
    inputs = make_inputs()
    out = model(**inputs, patch_size=16)

    s_t_x = out[0]
    assert s_t_x.shape[-1] == w.meta["embed_dim"]


def test_invalid_input_shape():
    """Incorrect band counts must raise an error."""
    model = galileo(variant="nano")

    bad = make_inputs()
    bad["s_t_x"] = torch.randn(2, 64, 64, 2, 3)

    with pytest.raises(Exception):
        model(**bad, patch_size=16)


def test_galileo_weights(mocked_weights):
    """Verify weight loading from a mocked local checkpoint."""
    w = mocked_weights
    model = galileo(weights=w)

    inputs = make_inputs(batch=1)
    out = model(**inputs, patch_size=16)

    assert isinstance(out[0], torch.Tensor)


@pytest.mark.slow
@pytest.mark.parametrize("w", list(GalileoWeights))
def test_real_weight_download(w):
    """Verify real HuggingFace weight download works."""
    model = galileo(weights=w)
    assert isinstance(model, torch.nn.Module)


