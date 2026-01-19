# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from datetime import datetime
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import Aurora_Weights, aurora_swin_unet

pytest.importorskip('aurora')


class TestAurora:
    @pytest.fixture(params=[*Aurora_Weights])
    def weights(self, request: SubRequest) -> Aurora_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> Aurora_Weights:
        import aurora

        weights = Aurora_Weights.HRES_T0_PRETRAINED_SMALL_AURORA
        # monkeypatch the load_checkpoint method to a no-op
        monkeypatch.setattr(
            aurora.Aurora, 'load_checkpoint', lambda self, *args, **kwargs: None
        )
        return weights

    def test_aurora_swin_unet(self) -> None:
        aurora_swin_unet()

    def test_aurora_swin_unet_weights(self, mocked_weights: Aurora_Weights) -> None:
        aurora_swin_unet(weights=mocked_weights)

    @pytest.mark.slow
    def test_aurora_swin_unet_download(self, weights: Aurora_Weights) -> None:
        aurora_swin_unet(weights=weights)

    @pytest.mark.slow
    @torch.inference_mode()
    def test_aurora_prediction(self, weights: Aurora_Weights) -> None:
        from aurora import Batch, Metadata

        model = aurora_swin_unet(weights=weights)
        patch_size = weights.meta['patch_size']
        h = 20 * patch_size
        w = 20 * patch_size
        num_atmos_levels = len((100, 250, 500, 850))
        surf_vars = {k: torch.randn(1, 2, h, w) for k in weights.meta['surf_vars']}
        if weights == Aurora_Weights.HRES_WAM0_WAVE_AURORA:
            # wave vars must be non-negative
            wave_magnitude_vars = {
                'swh',
                'mwp',
                'pp1d',
                'shww',
                'mpww',
                'mpts',
                'shts',
                'swh1',
                'mwp1',
                'swh2',
                'mwp2',
                'wind',
            }
            for k in wave_magnitude_vars:
                if k in surf_vars:
                    surf_vars[k] = torch.abs(surf_vars[k]) + 1e-3
        batch = Batch(
            surf_vars=surf_vars,
            static_vars={k: torch.randn(h, w) for k in weights.meta['static_vars']},
            atmos_vars={
                k: torch.randn(1, 2, num_atmos_levels, h, w)
                for k in weights.meta['atmos_vars']
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, h),
                lon=torch.linspace(0, 360, w + 1)[:-1],
                time=(datetime(2020, 6, 1, 12, 0),),
                atmos_levels=(100, 250, 500, 850),
            ),
        )
        model(batch)
