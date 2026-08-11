# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import Tensor

from torchgeo.models import CopernicusFM, CopernicusFM_Base_Weights, copernicusfm_base
from torchgeo.models.copernicusfm import (
    DynamicPatchEmbed,
    FourierExpansion,
    pi_resize_patch_embed,
    resize_abs_pos_embed,
)


class TestResizeEmbeddings:
    def test_resize_abs_pos_embed(self, use_float16: None) -> None:
        pos_embed = torch.rand(1, 4, 4)
        resize_abs_pos_embed(pos_embed, 2, 2)
        output = resize_abs_pos_embed(pos_embed, 2, 4, 0)
        assert output.dtype == torch.float16

    def test_pi_resize_patch_embed(self) -> None:
        patch_embed = torch.rand(1, 1, 4, 4, dtype=torch.float64)
        output = pi_resize_patch_embed(patch_embed, (2, 2))
        assert output.dtype == torch.float64


class TestFourierExpansion:
    def test_zeros(self, use_float16: None) -> None:
        expansion = FourierExpansion(1, 2)
        x = torch.zeros(2)
        output = expansion(x, 2)
        assert output.dtype == torch.float16

    def test_range(self) -> None:
        expansion = FourierExpansion(1, 2)
        x = torch.rand(2)
        match = 'The input tensor is not within the configured range'
        with pytest.raises(AssertionError, match=match):
            expansion(x, 2)

    def test_dimensionality(self) -> None:
        expansion = FourierExpansion(0, 1)
        x = torch.rand(2)
        match = 'The dimensionality must be a multiple of two.'
        with pytest.raises(ValueError, match=match):
            expansion(x, 3)


class TestDynamicPatchEmbed:
    def test_spectral(self) -> None:
        embed = DynamicPatchEmbed(input_mode='spectral')
        x = torch.rand(1, 1, 1, 1)
        match = 'For spectral hypernet, wavelengths and bandwidths must be provided.'
        with pytest.raises(ValueError, match=match):
            embed(x)

    def test_variable(self) -> None:
        embed = DynamicPatchEmbed(input_mode='variable')
        x = torch.rand(1, 1, 1, 1)
        match = 'For variable hypernet, language_embed must be provided.'
        with pytest.raises(ValueError, match=match):
            embed(x)

    def test_kernel_size(self) -> None:
        embed = DynamicPatchEmbed(kernel_size=16)
        x = torch.rand(1, 4, 28, 28)
        wavelengths = torch.tensor([664.6, 559.8, 492.4, 832.8])
        bandwidths = torch.tensor([31, 36, 66, 106])
        embed(x, wavelengths, bandwidths, kernel_size=12)


class TestCopernicusFM:
    @pytest.fixture(
        params=[
            [0, 1, 2, 3],
            [float('nan'), 1, 2, 3],
            [0, float('nan'), 2, 3],
            [0, 1, float('nan'), 3],
            [0, 1, 2, float('nan')],
            [float('nan'), float('nan'), float('nan'), float('nan')],
        ]
    )
    def metadata(self, request: SubRequest) -> Tensor:
        return torch.tensor([request.param])

    def test_global_pool(self, metadata: Tensor) -> None:
        model = CopernicusFM(global_pool=False)
        x = torch.rand(1, 4, 28, 28)
        wavelengths = [664.6, 559.8, 492.4, 832.8]
        bandwidths = [31, 36, 66, 106]
        input_mode = 'spectral'
        model(
            x,
            metadata,
            wavelengths=wavelengths,
            bandwidths=bandwidths,
            input_mode=input_mode,
        )

    def test_embed_dim(self) -> None:
        model = CopernicusFM(embed_dim=5, num_heads=5)
        x = torch.rand(1, 4, 28, 28)
        metadata = torch.tensor([[0, 1, float('nan'), float('nan')]])
        wavelengths = [664.6, 559.8, 492.4, 832.8]
        bandwidths = [31, 36, 66, 106]
        input_mode = 'spectral'
        model(
            x,
            metadata,
            wavelengths=wavelengths,
            bandwidths=bandwidths,
            input_mode=input_mode,
        )


class TestCopernicusFMBase:
    @pytest.fixture(params=[*CopernicusFM_Base_Weights])
    def weights(self, request: SubRequest) -> CopernicusFM_Base_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> CopernicusFM_Base_Weights:
        weights = CopernicusFM_Base_Weights.CopernicusFM_ViT
        path = tmp_path / f'{weights}.pth'
        model = copernicusfm_base()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_copernicusfm_spectral(self) -> None:
        model = copernicusfm_base()
        x = torch.rand(1, 4, 28, 28)
        metadata = torch.rand(1, 4) + 1
        wavelengths = [664.6, 559.8, 492.4, 832.8]
        bandwidths = [31, 36, 66, 106]
        input_mode = 'spectral'
        model(
            x,
            metadata,
            wavelengths=wavelengths,
            bandwidths=bandwidths,
            input_mode=input_mode,
        )

    def test_copernicusfm_spectral_dtype(self, monkeypatch: MonkeyPatch) -> None:
        model = copernicusfm_base()

        def check_dtype(
            x: Tensor,
            wavelengths: Tensor,
            bandwidths: Tensor,
            kernel_size: int | None = None,
        ) -> None:
            assert wavelengths.dtype == x.dtype
            assert bandwidths.dtype == x.dtype
            raise RuntimeError('dtype checked')

        monkeypatch.setattr(model.patch_embed_spectral, 'forward', check_dtype)
        x = torch.rand(1, 4, 28, 28, dtype=torch.float16)
        metadata = torch.rand(1, 4)
        wavelengths = [664.6, 559.8, 492.4, 832.8]
        bandwidths = [31, 36, 66, 106]
        with pytest.raises(RuntimeError, match='dtype checked'):
            model.forward_features(
                x, metadata, wavelengths=wavelengths, bandwidths=bandwidths
            )

    def test_copernicusfm_variable(self) -> None:
        model = copernicusfm_base()
        x = torch.rand(1, 1, 16, 16)
        metadata = torch.rand(1, 4) + 1
        language_embed = torch.rand(2048)
        input_mode = 'variable'
        model(x, metadata, language_embed=language_embed, input_mode=input_mode)

    def test_copernicusfm_weights(
        self, mocked_weights: CopernicusFM_Base_Weights
    ) -> None:
        copernicusfm_base(weights=mocked_weights)

    @pytest.mark.slow
    def test_copernicusfm_download(self, weights: CopernicusFM_Base_Weights) -> None:
        copernicusfm_base(weights=weights)
