# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from unittest.mock import MagicMock, patch

import pytest
import torch

from torchgeo.models import BTC
from torchgeo.models.btc import SwinBackbone

BACKBONES = ['swin_tiny', 'swin_small', 'swin_base']


class TestBTC:
    @pytest.mark.parametrize('backbone', BACKBONES)
    def test_btc_sizes(self, backbone: str) -> None:
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

    def test_unexpected_keys_raises(self) -> None:
        fake_weights = MagicMock()
        fake_weights.get_state_dict.return_value = {
            'state_dict': {'unexpected_keys': []}
        }

        with patch(
            'torchgeo.models.btc.SwinBackbone_Weights',  # patch the class, not the enum field
            new=MagicMock(CITYSCAPES_SEMSEG_TINY=fake_weights),
        ):
            with pytest.raises(
                RuntimeError,
                match=r'Failed to load pretrained weights for backbone: unexpected keys: ',
            ):
                SwinBackbone(model_size='swin_tiny')

    def test_missing_keys_raises(self) -> None:
        fake_weights = MagicMock()
        fake_weights.get_state_dict.return_value = {'state_dict': {}}

        with patch(
            'torchgeo.models.btc.SwinBackbone_Weights',  # patch the class, not the enum field
            new=MagicMock(CITYSCAPES_SEMSEG_TINY=fake_weights),
        ):
            with pytest.raises(
                RuntimeError, match=r'Missing keys in pretrained weights'
            ):
                SwinBackbone(model_size='swin_tiny')
