# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
import torchvision
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import Swin_V2_B_Weights, Swin_V2_T_Weights, swin_v2_b, swin_v2_t


class TestSwin_V2_T:
    @pytest.fixture(params=[*Swin_V2_T_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        weights: WeightsEnum,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        path = tmp_path / f'{weights}.pth'
        model = torchvision.models.swin_v2_t()
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_swin_v2_t(self) -> None:
        swin_v2_t()

    def test_swin_v2_t_weights(self, mocked_weights: WeightsEnum) -> None:
        swin_v2_t(weights=mocked_weights)

    def test_bands(self, mocked_weights: WeightsEnum) -> None:
        if 'bands' in mocked_weights.meta:
            assert len(mocked_weights.meta['bands']) == mocked_weights.meta['in_chans']

    def test_transforms(self, mocked_weights: WeightsEnum) -> None:
        c = mocked_weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        mocked_weights.transforms(sample)

    @pytest.mark.slow
    def test_swin_v2_t_download(self, weights: WeightsEnum) -> None:
        swin_v2_t(weights=weights)


class TestSwin_V2_B:
    @pytest.fixture(params=[*Swin_V2_B_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        weights: WeightsEnum,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        path = tmp_path / f'{weights}.pth'
        model = torchvision.models.swin_v2_b()
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_swin_v2_b(self) -> None:
        swin_v2_b()

    def test_swin_v2_b_weights(self, mocked_weights: WeightsEnum) -> None:
        swin_v2_b(weights=mocked_weights)

    def test_bands(self, mocked_weights: WeightsEnum) -> None:
        if 'bands' in mocked_weights.meta:
            assert len(mocked_weights.meta['bands']) == mocked_weights.meta['in_chans']

    def test_transforms(self, mocked_weights: WeightsEnum) -> None:
        c = mocked_weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        mocked_weights.transforms(sample)

    @pytest.mark.slow
    def test_swin_v2_b_download(self, weights: WeightsEnum) -> None:
        swin_v2_b(weights=weights)
