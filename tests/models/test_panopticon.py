# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import Panopticon, Panopticon_Weights, panopticon_vitb14
from torchgeo.models.panopticon import (
    ChnAttn,
    ChnEmb,
    Conv3dWrapper,
    CrossAttnNoQueryProj,
    PanopticonPE,
    get_1d_sincos_ipe_analytical,
    get_1d_sincos_pos_embed_from_grid_torch,
)


class TestPanopticon:
    def test_forward_shape(self) -> None:
        """Test if the forward pass produces the correct output shape."""
        model = Panopticon(attn_dim=64, embed_dim=768, img_size=28)
        x_dict = {
            'imgs': torch.randn(2, 3, 28, 28),  # (B, C, H, W)
            'chn_ids': torch.tensor([[664, 559, 493]]).repeat(2, 1),  # (B, C)
        }
        output = model(x_dict)
        assert output.shape == (2, 768)  # (B, embed_dim)

    def test_forward_with_different_image_sizes(self) -> None:
        """Test forward pass with different image sizes."""
        model = Panopticon(attn_dim=64, embed_dim=768, img_size=14)
        x_dict = {
            'imgs': torch.randn(2, 3, 14, 14),  # Smaller image size (B, C, H, W)
            'chn_ids': torch.tensor([[664, 559, 493]]).repeat(2, 1),  # (B, C)
        }
        output = model(x_dict)
        assert output.shape == (2, 768)  # (B, embed_dim)

    def test_forward_with_mask(self) -> None:
        """Test forward pass with a spectral mask."""
        model = Panopticon(attn_dim=64, embed_dim=768, img_size=28)
        x_dict = {
            'imgs': torch.randn(2, 3, 28, 28),  # (B, C, H, W)
            'chn_ids': torch.tensor([[664, 559, 493]]).repeat(2, 1),  # (B, C)
            'spec_masks': torch.randint(0, 2, (2, 3)).bool(),  # (B, C)
        }
        output = model(x_dict)
        assert output.shape == (2, 768)  # (B, embed_dim)


class TestPanopticonBase:
    @pytest.fixture(params=[*Panopticon_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> WeightsEnum:
        weights = Panopticon_Weights.VIT_BASE14
        path = tmp_path / f'{weights}.pth'
        model = panopticon_vitb14()
        state_dict = model.model.state_dict()
        state_dict['mask_token'] = None
        state_dict['pos_embed'] = torch.nn.Parameter(torch.randn(1, 1370, 768))
        torch.save(state_dict, path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_panopticon(self) -> None:
        model = panopticon_vitb14(img_size=28)
        x_dict = {
            'imgs': torch.randn(2, 3, 28, 28),  # (B, C, H, W)
            'chn_ids': torch.tensor([[664, 559, 493]]).repeat(2, 1),  # (B, C)
        }
        output = model(x_dict)
        assert output.shape == (2, 768)  # (B, embed_dim)

    def test_panopticon_weights(self, mocked_weights: WeightsEnum) -> None:
        model = panopticon_vitb14(weights=mocked_weights)
        x_dict = dict(
            imgs=torch.randn(2, 3, 224, 224),
            chn_ids=torch.tensor([[664, 559, 493]]).repeat(2, 1),
        )
        normed_cls_token = model(x_dict)
        assert tuple(normed_cls_token.shape) == (2, 768)

    @pytest.mark.slow
    def test_panopticon_download(self, weights: WeightsEnum) -> None:
        """Test forward pass with weights loaded."""
        model = panopticon_vitb14(weights)
        x_dict = dict(
            imgs=torch.randn(2, 3, 224, 224),
            chn_ids=torch.tensor([[664, 559, 493]]).repeat(2, 1),
        )
        normed_cls_token = model(x_dict)
        assert tuple(normed_cls_token.shape) == (2, 768)


class TestPanopticonPE:
    def test_initialization(self) -> None:
        """Test if PanopticonPE initializes correctly."""
        model = PanopticonPE(attn_dim=64, embed_dim=32, patch_size=14, img_size=28)
        assert model.patch_size == (14, 14)
        assert model.img_size == (28, 28)
        assert model.num_patches == (28 // 14) * (28 // 14)

    def test_forward_shape(self) -> None:
        """Test if the forward pass produces the correct output shape."""
        model = PanopticonPE(attn_dim=64, embed_dim=32, patch_size=14, img_size=28)
        x_dict = {
            'imgs': torch.randn(2, 3, 28, 28),  # (B, C, H, W)
            'chn_ids': torch.tensor([[664, 559, 493]]).repeat(2, 1),  # (B, C)
        }
        output = model(x_dict)
        assert output.shape == (2, 4, 32)  # (B, h, w, embed_dim)

    def test_forward_with_mask(self) -> None:
        """Test forward pass with a spectral mask."""
        model = PanopticonPE(attn_dim=64, embed_dim=32, patch_size=14, img_size=28)
        x_dict = {
            'imgs': torch.randn(2, 3, 28, 28),  # (B, C, H, W)
            'chn_ids': torch.tensor([[664, 559, 493]]).repeat(2, 1),  # (B, C)
            'spec_masks': torch.randint(0, 2, (2, 3)).bool(),  # (B, C)
        }
        output = model(x_dict)
        assert output.shape == (2, 4, 32)  # (B, h, w, embed_dim)

    def test_forward_with_different_image_sizes(self) -> None:
        """Test forward pass with different image sizes."""
        model = PanopticonPE(attn_dim=64, embed_dim=32, patch_size=14, img_size=28)
        x_dict = {
            'imgs': torch.randn(2, 3, 14, 14),  # (B, C, H, W)
            'chn_ids': torch.tensor([[664, 559, 493]]).repeat(2, 1),  # (B, C)
        }
        output = model(x_dict)
        assert output.shape == (2, 1, 32)  # (B, h, w, embed_dim)


class TestConv3dWrapper:
    def test_forward_shape(self) -> None:
        """Test if the forward pass produces the correct output shape."""
        model = Conv3dWrapper(patch_size=14, embed_dim=32)
        x = torch.randn(2, 3, 28, 28)  # (B, C, H, W)
        output, hp, wp = model(x)
        assert output.shape == (2, 3, 4, 32)  # (B, C, L, D)
        assert hp == 2  # H / patch_size
        assert wp == 2  # W / patch_size

    def test_forward_with_single_channel(self) -> None:
        """Test forward pass with a single input channel."""
        model = Conv3dWrapper(patch_size=14, embed_dim=32)
        x = torch.randn(2, 1, 28, 28)  # (B, C, H, W)
        output, hp, wp = model(x)
        assert output.shape == (2, 1, 4, 32)  # (B, C, L, D)
        assert hp == 2
        assert wp == 2


class TestChnAttn:
    def test_initialization(self) -> None:
        """Test if ChnAttn initializes correctly."""
        model = ChnAttn(dim=32)
        assert model.query.shape == (1, 1, 32)
        assert hasattr(model, 'chnemb')
        assert hasattr(model, 'xattn')

    def test_forward_shape(self) -> None:
        """Test if the forward pass produces the correct output shape."""
        model = ChnAttn(dim=32)
        x = torch.randn(2, 3, 4, 32)  # (B, C, L, D)
        chn_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (B, C)
        output = model(x, chn_ids)
        assert output.shape == (2, 4, 32)  # (B, L, D)

    def test_forward_with_mask(self) -> None:
        """Test forward pass with a mask."""
        model = ChnAttn(dim=32)
        x = torch.randn(2, 3, 4, 32)  # (B, C, L, D)
        chn_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (B, C)
        mask = torch.tensor([[1, 0, 1], [0, 1, 0]]).bool()  # (B, C)
        output = model(x, chn_ids, mask=mask)
        assert output.shape == (2, 4, 32)  # (B, L, D)

    def test_forward_with_layer_norm(self) -> None:
        """Test forward pass when layer normalization is enabled."""
        model = ChnAttn(dim=32, layer_norm=True)
        x = torch.randn(2, 3, 4, 32)  # (B, C, L, D)
        chn_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (B, C)
        output = model(x, chn_ids)
        assert output.shape == (2, 4, 32)  # (B, L, D)


class TestChnEmb:
    def test_initialization(self) -> None:
        """Test if ChnEmb initializes correctly."""
        model = ChnEmb(embed_dim=32)
        assert hasattr(model, 'embed_dim')
        assert model.embed_dim == 32

    def test_forward_shape(self) -> None:
        """Test if the forward pass produces the correct output shape."""
        model = ChnEmb(embed_dim=32)
        chn_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (B, C)
        output = model(chn_ids)
        assert output.shape == (2, 3, 32)  # (B, C, embed_dim)

    def test_forward_with_single_channel(self) -> None:
        """Test forward pass with a single channel."""
        model = ChnEmb(embed_dim=32)
        chn_ids = torch.tensor([[42]])  # (B=1, C=1)
        output = model(chn_ids)
        assert output.shape == (1, 1, 32)  # (B, C, embed_dim)

    def test_forward_with_full_spectra(self) -> None:
        """Test forward pass with full spectra."""
        model = ChnEmb(embed_dim=32, use_full_spectra=True)
        chn_ids = torch.ones(1, 3, 2)  # (B=1, C=3, full spectra)
        output = model(chn_ids)
        assert output.shape == (1, 3, 32)  # (B, C, embed_dim)


class TestCrossAttnNoQueryProj:
    def test_forward_shape(self) -> None:
        """Test if the forward pass produces the correct output shape."""
        model = CrossAttnNoQueryProj(dim=32, num_heads=4)
        query = torch.randn(2, 10, 32)  # (B, N, D)
        kv = torch.randn(2, 5, 32)  # (B, M, D)
        output = model(query, kv, kv)
        assert output.shape == (2, 10, 32)  # (B, N, D)

    def test_forward_with_mask(self) -> None:
        """Test forward pass with an attention mask."""
        model = CrossAttnNoQueryProj(dim=32, num_heads=4)
        query = torch.randn(2, 10, 32)  # (B, N, D)
        kv = torch.randn(2, 5, 32)  # (B, M, D)
        mask = torch.randint(0, 2, (2, 5)).bool()  # (B, M)
        output = model(query, kv, kv, key_padding_mask=mask)
        assert output.shape == (2, 10, 32)  # (B, N, D)

    def test_forward_with_different_batch_sizes(self) -> None:
        """Test forward pass with different batch sizes."""
        model = CrossAttnNoQueryProj(dim=32, num_heads=4)
        query = torch.randn(3, 10, 32)  # (B=3, N, D)
        kv = torch.randn(3, 5, 32)  # (B=3, M, D)
        output = model(query, kv, kv)
        assert output.shape == (3, 10, 32)  # (B, N, D)


class TestGet1DSincosPosEmbedFromGridTorch:
    def test_output_shape(self) -> None:
        """Test if the function produces the correct output shape."""
        embed_dim = 32
        grid = torch.linspace(0, 1, steps=10)  # 1D grid with 10 steps
        output = get_1d_sincos_pos_embed_from_grid_torch(embed_dim, grid)
        assert output.shape == (10, embed_dim)  # (grid_size, embed_dim)

    def test_invalid_embed_dim(self) -> None:
        """Test if the function raises an error for invalid embed_dim."""
        grid = torch.linspace(0, 1, steps=10)
        with pytest.raises(AssertionError):
            get_1d_sincos_pos_embed_from_grid_torch(767, grid)  # embed_dim must be even


class TestGet1DSincosIPEAnalytical:
    def test_output_shape(self) -> None:
        """Test if the function produces the correct output shape."""
        mu = torch.tensor([0.5, 1.0])  # (B,)
        sigma = torch.tensor([0.1, 0.2])  # (B,)
        D = 64
        device = torch.device('cpu')
        output = get_1d_sincos_ipe_analytical(mu, sigma, D, device)
        assert output.shape == (2, D)  # (B, D)

    def test_temperature_scaling(self) -> None:
        """Test if the function handles different temperature values."""
        mu = torch.tensor([0.5, 1.0])  # (B,)
        sigma = torch.tensor([0.1, 0.2])  # (B,)
        D = 64
        device = torch.device('cpu')
        temperature = 5000
        output = get_1d_sincos_ipe_analytical(
            mu, sigma, D, device, temperature=temperature
        )
        assert output.shape == (2, D)  # (B, D)
