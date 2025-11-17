# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
from pathlib import Path
from typing import Literal

import pytest
import torch
import torchvision

from torchgeo.models import BTC
from torchgeo.models.btc import SwinBackbone
from torchgeo.models.swin import SwinBackbone_Weights

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

    @pytest.fixture
    def patched_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        load_state_dict_from_url: None,
    ) -> Path:
        """Patch the weight enum URL to point to a fake checkpoint file."""
        ckpt_path = tmp_path / 'fake_swin_tiny.pth'
        monkeypatch.setattr(
            SwinBackbone_Weights.CITYSCAPES_SEMSEG_TINY.value, 'url', str(ckpt_path)
        )
        return ckpt_path

    def test_unexpected_keys_raises(
        self, patched_url: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = torchvision.models.swin_t(weights=None)
        state = model.state_dict()

        state['unexpected_keys'] = torch.tensor([1, 2, 3])

        torch.save({'state_dict': state}, patched_url)

        with pytest.raises(
            RuntimeError,
            match=r'Failed to load pretrained weights for backbone: unexpected keys: ',
        ):
            SwinBackbone(model_size='swin_tiny')

    def test_missing_keys_raises(
        self, patched_url: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = torchvision.models.swin_t(weights=None)
        state = model.state_dict()

        del state[next(iter(state))]

        torch.save({'state_dict': state}, patched_url)

        with pytest.raises(RuntimeError, match=r'Missing keys in pretrained weights'):
            SwinBackbone(model_size='swin_tiny')
