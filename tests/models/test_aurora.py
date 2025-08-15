# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from datetime import datetime
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import Aurora_Weights, aurora_swin_unet

pytest.importorskip('aurora')


class TestAurora:
    @pytest.fixture(params=[*Aurora_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> WeightsEnum:
        import aurora

        weights = Aurora_Weights.HRES_T0_PRETRAINED_SMALL_AURORA
        # monkeypatch the load_checkpoint method to a no-op
        monkeypatch.setattr(
            aurora.Aurora, 'load_checkpoint', lambda self, *args, **kwargs: None
        )
        return weights

    def test_aurora_swin_unet(self) -> None:
        aurora_swin_unet()

    def test_aurora_swin_unet_weights(self, mocked_weights: WeightsEnum) -> None:
        aurora_swin_unet(weights=mocked_weights)

    @pytest.mark.slow
    def test_aurora_swin_unet_download(self, weights: WeightsEnum) -> None:
        aurora_swin_unet(weights=weights)

    @pytest.mark.slow
    @torch.inference_mode()
    def test_aurora_prediction(self, weights: WeightsEnum) -> None:
        from aurora import Batch, Metadata

        model = aurora_swin_unet(weights=weights)
        batch = Batch(
            surf_vars={k: torch.randn(1, 2, 17, 32) for k in weights.meta['surf_vars']},
            static_vars={k: torch.randn(17, 32) for k in weights.meta['static_vars']},
            atmos_vars={
                k: torch.randn(1, 2, weights.meta['patch_size'], 17, 32)
                for k in weights.meta['atmos_vars']
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, 17),
                lon=torch.linspace(0, 360, 32 + 1)[:-1],
                time=(datetime(2020, 6, 1, 12, 0),),
                atmos_levels=(100, 250, 500, 850),
            ),
        )
        model(batch)
