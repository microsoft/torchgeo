# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import DEO_Weights, deo_base


class TestDEO:
    @pytest.fixture
    def model_swin(self) -> str:
        return 'swin_b'

    @pytest.fixture(params=[*DEO_Weights])
    def weights(self, request: SubRequest) -> DEO_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        load_state_dict_from_url: None,
        model_swin: str,
    ) -> DEO_Weights:
        weights = DEO_Weights.DEO_SWIN
        path = tmp_path / f'{weights}.pth'

        model = deo_base(model=model_swin, in_channels=10)
        torch.save(model.state_dict(), path)

        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_backbone_swin(self, model_swin: str) -> None:
        deo_base(None, model_swin, in_channels=10)

    def test_deo_weights(self, mocked_weights: DEO_Weights, model_swin: str) -> None:
        deo_base(weights=mocked_weights, model=model_swin, in_channels=10)

    def test_forward_rgb(self, model_swin: str) -> None:
        model = deo_base(None, model_swin, in_channels=3)
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, 256, 256)
            model(x)

    def test_forward_ms(self, model_swin: str) -> None:
        model = deo_base(None, model_swin, in_channels=10)
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 10, 256, 256)
            model(x)

    @pytest.mark.slow
    def test_deo_download(self, weights: DEO_Weights, model_swin: str) -> None:
        deo_base(weights=weights, model=model_swin, in_channels=10)
