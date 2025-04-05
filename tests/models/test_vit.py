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
    ViTBase14_DINOv2_Weights,
    ViTBase16_Weights,
    ViTHuge14_Weights,
    ViTLarge16_Weights,
    ViTSmall14_DINOv2_Weights,
    ViTSmall16_Weights,
    vit_base_patch14_dinov2,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch14_dinov2,
    vit_small_patch16_224,
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
        features_only: bool,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        weights = ViTSmall16_Weights.SENTINEL1_GRD_MAE
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            weights.meta['model'],
            in_chans=weights.meta['in_chans'],
            features_only=features_only,
        )
        model = model.model if features_only else model
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_vit(self) -> None:
        vit_small_patch16_224()

    def test_vit_weights(
        self, mocked_weights: WeightsEnum, features_only: bool
    ) -> None:
        vit_small_patch16_224(weights=mocked_weights, features_only=not features_only)

    def test_bands(self, weights: WeightsEnum) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        weights.transforms(sample)

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
        features_only: bool,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        weights = ViTBase16_Weights.SENTINEL1_GRD_MAE
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            weights.meta['model'],
            in_chans=weights.meta['in_chans'],
            features_only=features_only,
        )
        model = model.model if features_only else model
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_vit(self) -> None:
        vit_base_patch16_224()

    def test_vit_weights(
        self, mocked_weights: WeightsEnum, features_only: bool
    ) -> None:
        vit_base_patch16_224(weights=mocked_weights, features_only=not features_only)

    def test_bands(self, weights: WeightsEnum) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        weights.transforms(sample)

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
        features_only: bool,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        weights = ViTLarge16_Weights.SENTINEL1_GRD_MAE
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            weights.meta['model'],
            in_chans=weights.meta['in_chans'],
            features_only=features_only,
        )
        model = model.model if features_only else model
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_vit(self) -> None:
        vit_large_patch16_224()

    def test_vit_weights(
        self, mocked_weights: WeightsEnum, features_only: bool
    ) -> None:
        vit_large_patch16_224(weights=mocked_weights, features_only=not features_only)

    def test_bands(self, weights: WeightsEnum) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        weights.transforms(sample)

    # ViT-Large is too Large?
    # @pytest.mark.slow
    # def test_vit_download(self, weights: WeightsEnum) -> None:
    #     vit_large_patch16_224(weights=weights)


class TestViTHuge14:
    @pytest.fixture(params=[*ViTHuge14_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        features_only: bool,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        weights = ViTHuge14_Weights.SENTINEL1_GRD_MAE
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            weights.meta['model'],
            in_chans=weights.meta['in_chans'],
            features_only=features_only,
        )
        model = model.model if features_only else model
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_vit(self) -> None:
        vit_huge_patch14_224()

    def test_vit_weights(
        self, mocked_weights: WeightsEnum, features_only: bool
    ) -> None:
        vit_huge_patch14_224(weights=mocked_weights, features_only=not features_only)

    def test_bands(self, weights: WeightsEnum) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 224 * 224, dtype=torch.float).view(c, 224, 224)
        }
        weights.transforms(sample)

    # ViT-Huge is too Huge?
    # @pytest.mark.slow
    # def test_vit_download(self, weights: WeightsEnum) -> None:
    #     vit_huge_patch14_224(weights=weights)


class TestViTSmall14_DINOv2:
    @pytest.fixture(params=[*ViTSmall14_DINOv2_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        features_only: bool,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        weights = ViTSmall14_DINOv2_Weights.SENTINEL1_GRD_SOFTCON
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            weights.meta['model'],
            in_chans=weights.meta['in_chans'],
            img_size=weights.meta['img_size'],
            features_only=features_only,
        )
        model = model.model if features_only else model
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_vit(self) -> None:
        vit_small_patch14_dinov2()

    def test_vit_weights(
        self, mocked_weights: WeightsEnum, features_only: bool
    ) -> None:
        vit_small_patch14_dinov2(
            weights=mocked_weights, features_only=not features_only
        )

    def test_bands(self, weights: WeightsEnum) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta['in_chans']
        img_size = weights.meta['img_size']
        if isinstance(img_size, int):
            h = w = img_size
        else:
            h, w = img_size
        sample = {'image': torch.arange(c * h * w, dtype=torch.float).view(c, h, w)}
        weights.transforms(sample)

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
        features_only: bool,
        load_state_dict_from_url: None,
    ) -> WeightsEnum:
        weights = ViTBase14_DINOv2_Weights.SENTINEL1_GRD_SOFTCON
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            weights.meta['model'],
            in_chans=weights.meta['in_chans'],
            img_size=weights.meta['img_size'],
            features_only=features_only,
        )
        model = model.model if features_only else model
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_vit(self) -> None:
        vit_base_patch14_dinov2()

    def test_vit_weights(
        self, mocked_weights: WeightsEnum, features_only: bool
    ) -> None:
        vit_base_patch14_dinov2(weights=mocked_weights, features_only=not features_only)

    def test_bands(self, weights: WeightsEnum) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta['in_chans']
        img_size = weights.meta['img_size']
        if isinstance(img_size, int):
            h = w = img_size
        else:
            h, w = img_size
        sample = {'image': torch.arange(c * h * w, dtype=torch.float).view(c, h, w)}
        weights.transforms(sample)

    @pytest.mark.slow
    def test_vit_download(self, weights: WeightsEnum) -> None:
        vit_base_patch14_dinov2(weights=weights)
