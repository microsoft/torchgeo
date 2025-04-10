# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    DOFA,
    DOFABase16_Weights,
    DOFALarge16_Weights,
    dofa_base_patch16_224,
    dofa_huge_patch14_224,
    dofa_large_patch16_224,
    dofa_small_patch16_224,
)


class TestDOFA:
    @pytest.mark.parametrize(
        'wavelengths',
        [
            # Gaofen
            [0.443, 0.565, 0.763, 0.765, 0.910],
            # NAIP
            [0.640, 0.560, 0.480],
            [0.480, 0.560, 0.640, 0.810],
            # Sentinel-1
            [5.405],
            [5.405, 5.405],
            # Sentinel-2
            [
                0.443,
                0.490,
                0.560,
                0.665,
                0.705,
                0.740,
                0.783,
                0.842,
                0.865,
                0.945,
                1.375,
                1.610,
                2.190,
            ],
        ],
    )
    def test_dofa(self, wavelengths: list[float]) -> None:
        batch_size = 2
        num_channels = len(wavelengths)
        num_classes = 10
        global_pool = num_channels % 2 == 0
        model = DOFA(
            embed_dim=384,
            depth=12,
            num_heads=6,
            num_classes=num_classes,
            global_pool=global_pool,
        )
        batch = torch.randn([batch_size, num_channels, 224, 224])
        out = model(batch, wavelengths)
        assert out.shape == torch.Size([batch_size, num_classes])


class TestDOFASmall16:
    def test_dofa(self) -> None:
        model = dofa_small_patch16_224()
        x = torch.rand(1, 4, 224, 224)
        wavelengths = [664.6, 559.8, 492.4, 832.8]
        model(x, wavelengths)


class TestDOFABase16:
    @pytest.fixture(params=[*DOFABase16_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = DOFABase16_Weights.DOFA_MAE
        path = tmp_path / f'{weights}.pth'
        model = dofa_base_patch16_224()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_dofa(self) -> None:
        model = dofa_base_patch16_224()
        x = torch.rand(1, 4, 224, 224)
        wavelengths = [664.6, 559.8, 492.4, 832.8]
        model(x, wavelengths)

    def test_dofa_weights(self, mocked_weights: WeightsEnum) -> None:
        dofa_base_patch16_224(weights=mocked_weights)

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = 4
        sample = {
            'image': torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        weights.transforms(sample)

    @pytest.mark.slow
    def test_dofa_download(self, weights: WeightsEnum) -> None:
        dofa_base_patch16_224(weights=weights)


class TestDOFALarge16:
    @pytest.fixture(params=[*DOFALarge16_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = DOFALarge16_Weights.DOFA_MAE
        path = tmp_path / f'{weights}.pth'
        model = dofa_large_patch16_224()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_dofa(self) -> None:
        model = dofa_large_patch16_224()
        x = torch.rand(1, 4, 224, 224)
        wavelengths = [664.6, 559.8, 492.4, 832.8]
        model(x, wavelengths)

    def test_dofa_weights(self, mocked_weights: WeightsEnum) -> None:
        dofa_large_patch16_224(weights=mocked_weights)

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = 4
        sample = {
            'image': torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        weights.transforms(sample)

    @pytest.mark.slow
    def test_dofa_download(self, weights: WeightsEnum) -> None:
        dofa_large_patch16_224(weights=weights)


class TestDOFAHuge14:
    def test_dofa(self) -> None:
        model = dofa_huge_patch14_224()
        x = torch.rand(1, 4, 224, 224)
        wavelengths = [664.6, 559.8, 492.4, 832.8]
        model(x, wavelengths)
