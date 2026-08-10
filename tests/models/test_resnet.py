# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    resnet18,
    resnet50,
    resnet152,
)


class TestResNet18:
    @pytest.fixture(params=[*ResNet18_Weights])
    def weights(self, request: SubRequest) -> ResNet18_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        features_only: bool,
        load_state_dict_from_url: None,
    ) -> ResNet18_Weights:
        weights = ResNet18_Weights.SENTINEL2_RGB_MOCO
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            'resnet18', in_chans=weights.meta['in_chans'], features_only=features_only
        )
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_resnet(self) -> None:
        resnet18()

    def test_resnet_weights(
        self, mocked_weights: ResNet18_Weights, features_only: bool
    ) -> None:
        resnet18(weights=mocked_weights, features_only=not features_only)

    def test_bands(self, weights: ResNet18_Weights) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: ResNet18_Weights) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: ResNet18_Weights) -> None:
        """Test that the transforms have no graph breaks."""
        torch = pytest.importorskip('torch', minversion='2.6.0')
        torch.compiler.reset()
        c = weights.meta['in_chans']
        inputs = (torch.randn(1, c, 256, 256, dtype=torch.float),)
        torch.export.export(weights.transforms, inputs)

    @pytest.mark.slow
    def test_resnet_download(self, weights: ResNet18_Weights) -> None:
        resnet18(weights=weights)


class TestResNet50:
    @pytest.fixture(params=[*ResNet50_Weights])
    def weights(self, request: SubRequest) -> ResNet50_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        features_only: bool,
        load_state_dict_from_url: None,
    ) -> ResNet50_Weights:
        weights = ResNet50_Weights.SENTINEL2_RGB_MOCO
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            'resnet50', in_chans=weights.meta['in_chans'], features_only=features_only
        )
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_resnet(self) -> None:
        resnet50()

    def test_resnet_weights(
        self, mocked_weights: ResNet50_Weights, features_only: bool
    ) -> None:
        resnet50(weights=mocked_weights, features_only=not features_only)

    def test_bands(self, weights: ResNet50_Weights) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: ResNet50_Weights) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: ResNet50_Weights) -> None:
        """Test that the transforms have no graph breaks."""
        torch = pytest.importorskip('torch', minversion='2.6.0')
        torch.compiler.reset()
        c = weights.meta['in_chans']
        inputs = (torch.randn(1, c, 256, 256, dtype=torch.float),)
        torch.export.export(weights.transforms, inputs)

    @pytest.mark.slow
    def test_resnet_download(self, weights: ResNet50_Weights) -> None:
        resnet50(weights=weights)


class TestResNet152:
    @pytest.fixture(params=[*ResNet152_Weights])
    def weights(self, request: SubRequest) -> ResNet152_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        features_only: bool,
        load_state_dict_from_url: None,
    ) -> ResNet152_Weights:
        weights = ResNet152_Weights.SENTINEL2_SI_RGB_SATLAS
        path = tmp_path / f'{weights}.pth'
        model = timm.create_model(
            'resnet152', in_chans=weights.meta['in_chans'], features_only=features_only
        )
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_resnet(self) -> None:
        resnet152()

    def test_resnet_weights(
        self, mocked_weights: ResNet152_Weights, features_only: bool
    ) -> None:
        resnet152(weights=mocked_weights, features_only=not features_only)

    def test_bands(self, weights: ResNet152_Weights) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: ResNet152_Weights) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: ResNet152_Weights) -> None:
        """Test that the transforms have no graph breaks."""
        torch = pytest.importorskip('torch', minversion='2.6.0')
        torch.compiler.reset()
        c = weights.meta['in_chans']
        inputs = (torch.randn(1, c, 256, 256, dtype=torch.float),)
        torch.export.export(weights.transforms, inputs)

    @pytest.mark.slow
    def test_resnet_download(self, weights: ResNet152_Weights) -> None:
        resnet152(weights=weights)
