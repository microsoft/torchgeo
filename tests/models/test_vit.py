# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    ViTBase16_Weights,
    ViTHuge14_Weights,
    ViTLarge16_Weights,
    ViTSmall16_Weights,
    ViTSmall14_DINOv2_Weights,
    ViTBase14_DINOv2_Weights,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch16_224,
    vit_small_patch14_dinov2,
    vit_base_patch14_dinov2,
)


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

    def test_vit(self) -> None:
        vit_small_patch16_224()

    def test_vit_weights(self, mocked_weights: WeightsEnum) -> None:
        vit_small_patch16_224(weights=mocked_weights)

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


class TestViTBase16:
    @pytest.fixture(params=[*ViTBase16_Weights])
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

    def test_vit(self) -> None:
        vit_base_patch16_224()

    def test_vit_weights(self, mocked_weights: WeightsEnum) -> None:
        vit_base_patch16_224(weights=mocked_weights)

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
        vit_base_patch16_224(weights=weights)


class TestViTLarge16:
    @pytest.fixture(params=[*ViTLarge16_Weights])
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

    def test_vit(self) -> None:
        vit_large_patch16_224()

    def test_vit_weights(self, mocked_weights: WeightsEnum) -> None:
        vit_large_patch16_224(weights=mocked_weights)

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
        vit_large_patch16_224(weights=weights)


class TestViTHuge14:
    @pytest.fixture(params=[*ViTHuge14_Weights])
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

    def test_vit(self) -> None:
        vit_huge_patch14_224()

    def test_vit_weights(self, mocked_weights: WeightsEnum) -> None:
        vit_huge_patch14_224(weights=mocked_weights)

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
        vit_huge_patch14_224(weights=weights)

class TestViTSmall14_DINOv2:
    @pytest.fixture(params=[*ViTSmall14_DINOv2_Weights])
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
            weights.meta['model'], in_chans=weights.meta['in_chans'], img_size=weights.meta['img_size']
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_vit(self) -> None:
        vit_small_patch14_dinov2()

    def test_vit_weights(self, mocked_weights: WeightsEnum) -> None:
        vit_small_patch14_dinov2(weights=mocked_weights)

    def test_bands(self, mocked_weights: WeightsEnum) -> None:
        if 'bands' in mocked_weights.meta:
            assert len(mocked_weights.meta['bands']) == mocked_weights.meta['in_chans']

    def test_transforms(self, mocked_weights: WeightsEnum) -> None:
        c = mocked_weights.meta['in_chans']
        img_size = mocked_weights.meta['img_size']
        if isinstance(img_size, int):
            h = w = img_size
        else:
            h, w = img_size
        sample = {
            'image': torch.arange(c * h * w, dtype=torch.float).view(c, h, w)
        }
        mocked_weights.transforms(sample)

    @pytest.mark.slow
    def test_vit_download(self, weights: WeightsEnum) -> None:
        vit_small_patch14_dinov2(weights=weights)

class TestViTBase14_DINOv2:
    @pytest.fixture(params=[*ViTBase14_DINOv2_Weights])
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
            weights.meta['model'], in_chans=weights.meta['in_chans'], img_size=weights.meta['img_size']
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_vit(self) -> None:
        vit_base_patch14_dinov2()

    def test_vit_weights(self, mocked_weights: WeightsEnum) -> None:
        vit_base_patch14_dinov2(weights=mocked_weights)

    def test_bands(self, mocked_weights: WeightsEnum) -> None:
        if 'bands' in mocked_weights.meta:
            assert len(mocked_weights.meta['bands']) == mocked_weights.meta['in_chans']

    def test_transforms(self, mocked_weights: WeightsEnum) -> None:
        c = mocked_weights.meta['in_chans']
        img_size = mocked_weights.meta['img_size']
        if isinstance(img_size, int):
            h = w = img_size
        else:
            h, w = img_size
        sample = {
            'image': torch.arange(c * h * w, dtype=torch.float).view(c, h, w)
        }
        mocked_weights.transforms(sample)

    @pytest.mark.slow
    def test_vit_download(self, weights: WeightsEnum) -> None:
        vit_base_patch14_dinov2(weights=weights)