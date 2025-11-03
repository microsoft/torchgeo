# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Literal

import pytest
import torch

from torchgeo.models import BTC
from torchgeo.models.btc import SwinBackbone

BACKBONES = ['swin_tiny', 'swin_small', 'swin_base']


class TestBTC:
    @pytest.mark.parametrize('backbone', BACKBONES)
    def test_btc_sizes(
        self, backbone: Literal['swin_tiny', 'swin_small', 'swin_base']
    ) -> None:
        model = BTC(backbone=backbone)
        model.eval()
        with torch.no_grad():
            x = torch.randn(2, 2 * 3, 256, 256)
            model(x)

    def test_btc_invalid_size(self) -> None:
        with pytest.raises(
            ValueError,
            match=r'Invalid swin size: fail_test. Possible options: swin_\[tiny | small | base\]',
        ):
            SwinBackbone(model_size='fail_test')

    def test_unexpected_keys_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeWeights:
            @staticmethod
            def get_state_dict(progress: bool) -> dict[str, Any]:
                return {'state_dict': {'unexpected_keys': []}}

        class FakeSwinWeights:
            CITYSCAPES_SEMSEG_TINY = FakeWeights()

        monkeypatch.setattr('torchgeo.models.btc.SwinBackbone_Weights', FakeSwinWeights)

        with pytest.raises(
            RuntimeError,
            match=r'Failed to load pretrained weights for backbone: unexpected keys: ',
        ):
            SwinBackbone(model_size='swin_tiny')

    def test_missing_keys_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeWeights:
            @staticmethod
            def get_state_dict(progress: bool) -> dict[str, Any]:
                return {'state_dict': {}}

        class FakeSwinWeights:
            CITYSCAPES_SEMSEG_TINY = FakeWeights()

        monkeypatch.setattr('torchgeo.models.btc.SwinBackbone_Weights', FakeSwinWeights)
        with pytest.raises(RuntimeError, match=r'Missing keys in pretrained weights'):
            SwinBackbone(model_size='swin_tiny')
