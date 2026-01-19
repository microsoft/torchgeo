# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import torch
import pytest
from torchgeo.models import galileo, GalileoWeights, GalileoEncoder
from torchgeo.models.galileo import GalileoVariant
from typing import Optional


@pytest.mark.parametrize("variant,dim", [
    ("nano", 192),
    ("tiny", 384),
    ("base", 768),
])
def test_galileo_forward_shapes(variant: str, dim: int) -> None:
    model = galileo(variant=variant)
    x = torch.randn(2, 4, 224, 224)
    y = model(x)

    assert y.shape == (2, dim)

def test_galileo_invalid_input_size() -> None:
    model = galileo()
    x = torch.randn(1, 4, 256, 256)

    with pytest.raises(ValueError):
        model(x)


def test_galileo_variant_weight_mismatch() -> None:
    with pytest.raises(ValueError):
        galileo(
            variant="tiny",
            weights=GalileoWeights.GALILEO_S2_BASE_V1,
        )

def test_galileo_load_state_dict(monkeypatch):
    model = galileo(variant="nano")

    dummy_state = {
        k: torch.randn_like(v)
        for k, v in model.state_dict().items()
    }

    def fake_get_state_dict(*args, **kwargs):
        return dummy_state

    monkeypatch.setattr(
        GalileoWeights.GALILEO_S2_NANO_V1,
        "get_state_dict",
        fake_get_state_dict,
    )

def galileo(
    *,
    variant: GalileoVariant = "base",
    weights: Optional[GalileoWeights] = None,
) -> GalileoEncoder:

    if weights is not None:
        weights = GalileoWeights.verify(weights)
        expected = weights.meta["variant"]

        # ✅ FAIL FAST — no checkpoint touched
        if variant != expected:
            raise ValueError(
                f"Variant '{variant}' does not match weights '{expected}'"
            )

    model = GalileoEncoder(variant=variant)

    if weights is not None:
        state_dict = weights.get_state_dict(progress=True, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model

