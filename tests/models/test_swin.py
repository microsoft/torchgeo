# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
import torchvision
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import (
    Swin_B_Weights,
    Swin_S_Weights,
    Swin_T_Weights,
    Swin_V2_B_Weights,
    Swin_V2_T_Weights,
    swin_b,
    swin_s,
    swin_t,
    swin_v2_b,
    swin_v2_t,
)


class TestSwin_T:
    @pytest.fixture(params=[*Swin_T_Weights])
    def weights(self, request: SubRequest) -> Swin_T_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Swin_T_Weights:
        weights = Swin_T_Weights.CITYSCAPES_SEMSEG
        path = tmp_path / f'{weights}.pth'
        model = torchvision.models.swin_t()
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        torch.save({'state_dict': model.state_dict()}, path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_swin_t(self) -> None:
        swin_t()

    def test_swin_t_weights(self, mocked_weights: Swin_T_Weights) -> None:
        swin_t(weights=mocked_weights)

    def test_bands(self, weights: Swin_T_Weights) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: Swin_T_Weights) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: Swin_T_Weights) -> None:
        """Test that the transforms have no graph breaks."""
        torch = pytest.importorskip('torch', minversion='2.6.0')
        torch.compiler.reset()
        c = weights.meta['in_chans']
        inputs = (torch.randn(1, c, 256, 256, dtype=torch.float),)
        torch.export.export(weights.transforms, inputs)

    @pytest.mark.slow
    def test_swin_t_download(self, weights: Swin_T_Weights) -> None:
        swin_t(weights=weights)


class TestSwin_S:
    @pytest.fixture(params=[*Swin_S_Weights])
    def weights(self, request: SubRequest) -> Swin_S_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Swin_S_Weights:
        weights = Swin_S_Weights.CITYSCAPES_SEMSEG
        path = tmp_path / f'{weights}.pth'
        model = torchvision.models.swin_s()
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        torch.save({'state_dict': model.state_dict()}, path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_swin_s(self) -> None:
        swin_s()

    def test_swin_s_weights(self, mocked_weights: Swin_S_Weights) -> None:
        swin_s(weights=mocked_weights)

    def test_bands(self, weights: Swin_S_Weights) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: Swin_S_Weights) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: Swin_S_Weights) -> None:
        """Test that the transforms have no graph breaks."""
        torch = pytest.importorskip('torch', minversion='2.6.0')
        torch.compiler.reset()
        c = weights.meta['in_chans']
        inputs = (torch.randn(1, c, 256, 256, dtype=torch.float),)
        torch.export.export(weights.transforms, inputs)

    @pytest.mark.slow
    def test_swin_s_download(self, weights: Swin_S_Weights) -> None:
        swin_s(weights=weights)


class TestSwin_B:
    @pytest.fixture(params=[*Swin_B_Weights])
    def weights(self, request: SubRequest) -> Swin_B_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Swin_B_Weights:
        weights = Swin_B_Weights.CITYSCAPES_SEMSEG
        # swin-b can have larger window size
        window_size = weights.meta['window_size']
        path = tmp_path / f'{weights}.pth'
        model = torchvision.models.SwinTransformer(
            patch_size=[4, 4],
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=[window_size, window_size],
            stochastic_depth_prob=0.5,
        )
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels  # type: ignore[not-subscriptable]
        model.features[0][0] = torch.nn.Conv2d(  # type: ignore[invalid-assignment]
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        torch.save({'state_dict': model.state_dict()}, path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_swin_b(self) -> None:
        swin_b()

    def test_swin_b_weights(self, mocked_weights: Swin_B_Weights) -> None:
        swin_b(weights=mocked_weights)

    def test_bands(self, weights: Swin_B_Weights) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: Swin_B_Weights) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: Swin_B_Weights) -> None:
        """Test that the transforms have no graph breaks."""
        torch = pytest.importorskip('torch', minversion='2.6.0')
        torch.compiler.reset()
        c = weights.meta['in_chans']
        inputs = (torch.randn(1, c, 256, 256, dtype=torch.float),)
        torch.export.export(weights.transforms, inputs)

    @pytest.mark.slow
    def test_swin_b_download(self, weights: Swin_B_Weights) -> None:
        swin_b(weights=weights)


class TestSwin_V2_T:
    @pytest.fixture(params=[*Swin_V2_T_Weights])
    def weights(self, request: SubRequest) -> Swin_V2_T_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Swin_V2_T_Weights:
        weights = Swin_V2_T_Weights.SENTINEL2_SI_RGB_SATLAS
        path = tmp_path / f'{weights}.pth'
        model = torchvision.models.swin_v2_t()
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_swin_v2_t(self) -> None:
        swin_v2_t()

    def test_swin_v2_t_weights(self, mocked_weights: Swin_V2_T_Weights) -> None:
        swin_v2_t(weights=mocked_weights)

    def test_bands(self, weights: Swin_V2_T_Weights) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: Swin_V2_T_Weights) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: Swin_V2_T_Weights) -> None:
        """Test that the transforms have no graph breaks."""
        torch = pytest.importorskip('torch', minversion='2.6.0')
        torch.compiler.reset()
        c = weights.meta['in_chans']
        inputs = (torch.randn(1, c, 256, 256, dtype=torch.float),)
        torch.export.export(weights.transforms, inputs)

    @pytest.mark.slow
    def test_swin_v2_t_download(self, weights: Swin_V2_T_Weights) -> None:
        swin_v2_t(weights=weights)


class TestSwin_V2_B:
    @pytest.fixture(params=[*Swin_V2_B_Weights])
    def weights(self, request: SubRequest) -> Swin_V2_B_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Swin_V2_B_Weights:
        weights = Swin_V2_B_Weights.SENTINEL1_SI_SATLAS
        path = tmp_path / f'{weights}.pth'
        model = torchvision.models.swin_v2_b()
        num_channels = weights.meta['in_chans']
        out_channels = model.features[0][0].out_channels
        model.features[0][0] = torch.nn.Conv2d(
            num_channels, out_channels, kernel_size=(4, 4), stride=(4, 4)
        )
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_swin_v2_b(self) -> None:
        swin_v2_b()

    def test_swin_v2_b_weights(self, mocked_weights: Swin_V2_B_Weights) -> None:
        swin_v2_b(weights=mocked_weights)

    def test_bands(self, weights: Swin_V2_B_Weights) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: Swin_V2_B_Weights) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    def test_export_transforms(self, weights: Swin_V2_B_Weights) -> None:
        """Test that the transforms have no graph breaks."""
        torch = pytest.importorskip('torch', minversion='2.6.0')
        torch.compiler.reset()
        c = weights.meta['in_chans']
        inputs = (torch.randn(1, c, 256, 256, dtype=torch.float),)
        torch.export.export(weights.transforms, inputs)

    @pytest.mark.slow
    def test_swin_v2_b_download(self, weights: Swin_V2_B_Weights) -> None:
        swin_v2_b(weights=weights)
