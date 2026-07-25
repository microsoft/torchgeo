# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from typing import Any

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import UniverSat_Base_Weights, universat


class TestUniverSat:
    @pytest.fixture(params=[*UniverSat_Base_Weights])
    def weights(self, request: SubRequest) -> UniverSat_Base_Weights:
        return request.param

    def test_universat(self, monkeypatch: MonkeyPatch) -> None:
        calls: list[dict[str, Any]] = []

        def fake_load(*args: Any, **kwargs: Any) -> nn.Module:
            calls.append(kwargs)
            return nn.Identity()

        monkeypatch.setattr(torch.hub, 'load', fake_load)
        model = universat()
        assert isinstance(model, nn.Module)
        # Without weights the entrypoint must build an untrained model.
        assert calls[0]['pretrained'] is False

    def test_universat_weights(
        self, weights: UniverSat_Base_Weights, monkeypatch: MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_load(*args: Any, **kwargs: Any) -> nn.Module:
            calls.append(kwargs)
            return nn.Identity()

        monkeypatch.setattr(torch.hub, 'load', fake_load)
        universat(weights=weights)
        # With weights the entrypoint must load the pre-trained checkpoint.
        assert calls[0]['pretrained'] is True

    @pytest.mark.slow
    def test_universat_download(self, weights: UniverSat_Base_Weights) -> None:
        pytest.importorskip('huggingface_hub')
        universat(weights=weights)
