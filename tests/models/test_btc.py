# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Literal

import pytest
import torch
from pytest import MonkeyPatch

import torchgeo
import torchgeo.models
from torchgeo.models import BTC
from torchgeo.models.btc import SwinBackbone

BACKBONES = ['swin_tiny', 'swin_small', 'swin_base']


class TestBTC:
    @pytest.mark.parametrize('backbone', BACKBONES)
    def test_btc_sizes(
        self, backbone: Literal['swin_tiny', 'swin_small', 'swin_base']
    ) -> None:
        model = BTC(backbone=backbone, backbone_pretrained=False)
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 2 * 3, 256, 256)
            model(x)

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> str:
        backbone = SwinBackbone('swin_tiny', backbone_pretrained=False)
        path = tmp_path / 'mocked_swin_backbone.pth'
        torch.save(
            {
                'state_dict': backbone.feature_extractor.state_dict(),
                'feat_norms_state_dict': backbone.norms.state_dict(),
            },
            path,
        )
        monkeypatch.setattr(
            torchgeo.models.Swin_T_Weights.CITYSCAPES_SEMSEG.value, 'url', str(path)
        )
        return 'swin_tiny'

    def test_backbone_weight_load(self, mocked_weights: str) -> None:
        """Test to cover backbone and layernorm weight loading."""
        mocked_size = mocked_weights
        SwinBackbone(mocked_size, backbone_pretrained=True)

    def test_btc_invalid_size(self) -> None:
        with pytest.raises(
            ValueError,
            match=r'Invalid swin size: fail_test. Possible options: swin_\[tiny | small | base\]',
        ):
            SwinBackbone(model_size='fail_test')

    @pytest.mark.slow
    @pytest.mark.parametrize('backbone', BACKBONES)
    def test_btc_backbone_download(
        self, backbone: Literal['swin_tiny', 'swin_small', 'swin_base']
    ) -> None:
        BTC(backbone=backbone, backbone_pretrained=True)
