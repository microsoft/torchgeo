# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import Tensor
from torchvision.models._api import WeightsEnum

from torchgeo.models import CopernicusFM, CopernicusFM_Base_Weights, copernicusfm_base
from torchgeo.models.copernicusfm import (
    DynamicPatchEmbed,
    FourierExpansion,
    pi_resize_patch_embed,
    resize_abs_pos_embed,
)


class TestResizeEmbeddings:
    def test_resize_abs_pos_embed(self) -> None:
        pos_embed = torch.rand(1, 4, 4)
        resize_abs_pos_embed(pos_embed, 2, 2)
        resize_abs_pos_embed(pos_embed, 2, 4, 0)

    def test_pi_resize_patch_embed(self) -> None:
        patch_embed = torch.rand(1, 1, 4, 4)
        pi_resize_patch_embed(patch_embed, (4, 4))


class TestFourierExpansion:
    def test_zeros(self) -> None:
        expansion = FourierExpansion(1, 2)
        x = torch.zeros(2)
        expansion(x, 2)

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
        embed = DynamicPatchEmbed(hypernet='spectral')
        img_feat = torch.rand(1, 1, 1, 1)
        match = 'For spectral hypernet, wvs and bandwidths must be provided.'
        with pytest.raises(ValueError, match=match):
            embed(img_feat)

    def test_variable(self) -> None:
        embed = DynamicPatchEmbed(hypernet='variable')
        img_feat = torch.rand(1, 1, 1, 1)
        match = 'For variable hypernet, language_embed must be provided.'
        with pytest.raises(ValueError, match=match):
            embed(img_feat)

    def test_kernel_size(self) -> None:
        embed = DynamicPatchEmbed(kernel_size=16)
        img_feat = torch.rand(1, 4, 28, 28)
        wvs = torch.tensor([664.6, 559.8, 492.4, 832.8])
        bandwidths = torch.tensor([31, 36, 66, 106])
        embed(img_feat, wvs, bandwidths, kernel_size=12)


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
    def meta_info(self, request: SubRequest) -> Tensor:
        return torch.tensor([request.param])

    def test_global_pool(self, meta_info: Tensor) -> None:
        model = CopernicusFM(global_pool=False)
        x = torch.rand(1, 4, 28, 28)
        wave_list = [664.6, 559.8, 492.4, 832.8]
        bandwidth = [31, 36, 66, 106]
        input_mode = 'spectral'
        model(
            x,
            meta_info,
            wave_list=wave_list,
            bandwidth=bandwidth,
            input_mode=input_mode,
        )

    def test_embed_dim(self) -> None:
        model = CopernicusFM(embed_dim=5, num_heads=5)
        x = torch.rand(1, 4, 28, 28)
        meta_info = torch.tensor([[0, 1, float('nan'), float('nan')]])
        wave_list = [664.6, 559.8, 492.4, 832.8]
        bandwidth = [31, 36, 66, 106]
        input_mode = 'spectral'
        model(
            x,
            meta_info,
            wave_list=wave_list,
            bandwidth=bandwidth,
            input_mode=input_mode,
        )


class TestCopernicusFMBase:
    @pytest.fixture(params=[*CopernicusFM_Base_Weights])
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
        model = copernicusfm_base()
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, 'url', str(path))
        except AttributeError:
            monkeypatch.setattr(weights, 'url', str(path))
        return weights

    def test_copernicusfm_spectral(self) -> None:
        model = copernicusfm_base()
        x = torch.rand(1, 4, 28, 28)
        meta_info = torch.rand(1, 4)
        wave_list = [664.6, 559.8, 492.4, 832.8]
        bandwidth = [31, 36, 66, 106]
        input_mode = 'spectral'
        model(
            x,
            meta_info,
            wave_list=wave_list,
            bandwidth=bandwidth,
            input_mode=input_mode,
        )

    def test_copernicusfm_variable(self) -> None:
        model = copernicusfm_base()
        x = torch.rand(1, 1, 96, 96)
        meta_info = torch.rand(1, 4)
        language_embed = torch.rand(2048)
        input_mode = 'variable'
        model(x, meta_info, language_embed=language_embed, input_mode=input_mode)

    def test_copernicusfm_weights(self, mocked_weights: WeightsEnum) -> None:
        copernicusfm_base(weights=mocked_weights)

    @pytest.mark.slow
    def test_copernicusfm_download(self, weights: WeightsEnum) -> None:
        copernicusfm_base(weights=weights)
