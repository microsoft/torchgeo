# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    CROMA,
    CROMABase_Weights,
    CROMALarge_Weights,
    croma_base,
    croma_large,
)


def save_model(model: torch.nn.Module, path: Path) -> None:
    state_dict = {
        's1_encoder': model.s1_encoder.state_dict(),
        's1_GAP_FFN': model.s1_GAP_FFN.state_dict(),
        's2_encoder': model.s2_encoder.state_dict(),
        's2_GAP_FFN': model.s2_GAP_FFN.state_dict(),
        'joint_encoder': model.joint_encoder.state_dict(),
    }
    torch.save(state_dict, path)


class TestCROMA:
    @pytest.mark.parametrize('modalities', [['sar'], ['optical'], ['sar', 'optical']])
    def test_croma(self, modalities: list[str]) -> None:
        batch_size = 2
        model = CROMA(modalities=modalities)
        if 'sar' in modalities:
            sar_images = torch.randn(
                [batch_size, 2, model.image_size, model.image_size]
            )
        else:
            sar_images = None
        if 'optical' in modalities:
            optical_images = torch.randn(
                [batch_size, 12, model.image_size, model.image_size]
            )
        else:
            optical_images = None
        out = model(sar_images, optical_images)
        for modality in modalities:
            assert f'{modality}_encodings' in out
        if set(modalities) == {'sar', 'optical'}:
            assert 'joint_encodings' in out


class TestCROMABase:
    @pytest.fixture(params=[*CROMABase_Weights])
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
        model = croma_base()
        save_model(model, path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_croma(self) -> None:
        croma_base()

    def test_croma_weights(self, mocked_weights: WeightsEnum) -> None:
        croma_base(weights=mocked_weights)

    @pytest.mark.slow
    def test_croma_download(self, weights: WeightsEnum) -> None:
        croma_base(weights=weights)


class TestCROMALarge:
    @pytest.fixture(params=[*CROMALarge_Weights])
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
        model = croma_large()
        save_model(model, path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_croma(self) -> None:
        croma_large()

    def test_croma_weights(self, mocked_weights: WeightsEnum) -> None:
        croma_large(weights=mocked_weights)

    @pytest.mark.slow
    def test_croma_download(self, weights: WeightsEnum) -> None:
        croma_large(weights=weights)
