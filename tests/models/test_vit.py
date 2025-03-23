# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import ViTSmall16_Weights, vit_small_patch16_224


class TestViTSmall16:
    @pytest.fixture(params=[*ViTSmall16_Weights])
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
        model = timm.create_model(
            weights.meta['model'], in_chans=weights.meta['in_chans']
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    @pytest.fixture
    def mocked_weights_features_only(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        weights: WeightsEnum,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        # features_only args implemented in timm 1.0.3
        pytest.importorskip('timm', minversion='1.0.3')
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            weights.meta['model'], in_chans=weights.meta['in_chans'], features_only=True
        )
        # the actual ViT is stored in model attribute
        torch.save(model.model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_vit(self) -> None:
        vit_small_patch16_224()

    def test_vit_weights(self, mocked_weights: WeightsEnum) -> None:
        vit_small_patch16_224(weights=mocked_weights)

    def test_vit_weights_features_only(
        self, mocked_weights_features_only: WeightsEnum
    ) -> None:
        vit_small_patch16_224(weights=mocked_weights_features_only, features_only=True)

    def test_bands(self, mocked_weights: WeightsEnum) -> None:
        if 'bands' in mocked_weights.meta:
            assert len(mocked_weights.meta['bands']) == mocked_weights.meta['in_chans']

    def test_transforms(self, mocked_weights: WeightsEnum) -> None:
        c = mocked_weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        mocked_weights.transforms(sample)

    @pytest.mark.slow
    def test_vit_download(self, weights: WeightsEnum) -> None:
        vit_small_patch16_224(weights=weights)
