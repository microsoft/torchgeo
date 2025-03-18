# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import Tensor
from torchvision.models._api import WeightsEnum

from torchgeo.models import CopernicusFM_Base_Weights, copernicusfm_base


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

    def test_copernicusfm_spectral(self, meta_info: Tensor) -> None:
        model = copernicusfm_base()
        x = torch.rand(1, 4, 224, 224)
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

    def test_copernicusfm_builtin_variable(self, meta_info: Tensor) -> None:
        model = copernicusfm_base()
        x = torch.rand(1, 4, 224, 224)
        key = 's5p_co'
        # TODO: why are these required?
        wave_list = [664.6, 559.8, 492.4, 832.8]
        bandwidth = [31, 36, 66, 106]
        input_mode = 'variable'
        model(
            x,
            meta_info,
            key=key,
            wave_list=wave_list,
            bandwidth=bandwidth,
            input_mode=input_mode,
        )

    def test_copernicusfm_custom_variable(self, meta_info: Tensor) -> None:
        model = copernicusfm_base()
        x = torch.rand(1, 4, 224, 224)
        key = 'lulc'
        # TODO: why are these required?
        wave_list = [664.6, 559.8, 492.4, 832.8]
        bandwidth = [31, 36, 66, 106]
        language_embed = torch.rand(2048)
        input_mode = 'variable'
        model(
            x,
            meta_info,
            key=key,
            wave_list=wave_list,
            bandwidth=bandwidth,
            language_embed=language_embed,
            input_mode=input_mode,
        )

    def test_copernicusfm_weights(self, mocked_weights: WeightsEnum) -> None:
        copernicusfm_base(weights=mocked_weights)

    @pytest.mark.slow
    def test_copernicusfm_download(self, weights: WeightsEnum) -> None:
        copernicusfm_base(weights=weights)
