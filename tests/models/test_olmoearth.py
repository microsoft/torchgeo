# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import re
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import OlmoEarthV1_Weights, olmoearth_v1, olmoearth_v1_unet_decoder

pytest.importorskip('olmoearth_pretrain_minimal')


class TestOlmoEarthV1:
    @pytest.fixture(params=[*OlmoEarthV1_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthV1_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> OlmoEarthV1_Weights:
        weights = OlmoEarthV1_Weights.NANO
        path = tmp_path / 'weights.pth'
        model = olmoearth_v1(model_size='nano')
        # Save the *inner* state dict (encoder.*/decoder.*), which is how the published
        # checkpoints are keyed. Saving the wrapper's dict would already carry the model.*
        # prefix and so would not exercise the re-keying that loading requires.
        torch.save(model.model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_olmoearth_v1(self) -> None:
        olmoearth_v1()

    def test_olmoearth_v1_weights(self, mocked_weights: OlmoEarthV1_Weights) -> None:
        olmoearth_v1(weights=mocked_weights)

    def test_olmoearth_v1_weights_are_applied(
        self, mocked_weights: OlmoEarthV1_Weights
    ) -> None:
        """Two models built from the same weights must hold identical tensors.

        Regression test for the ``model.*`` prefix mismatch: the checkpoint keys and the
        parameter names were disjoint, so ``strict=False`` dropped every tensor and each call
        returned a freshly randomized model.
        """
        one = olmoearth_v1(weights=mocked_weights).state_dict()
        two = olmoearth_v1(weights=mocked_weights).state_dict()
        assert one.keys() == two.keys()
        for key, value in one.items():
            assert torch.equal(value, two[key]), key

    @pytest.mark.slow
    def test_olmoearth_v1_download(self, weights: OlmoEarthV1_Weights) -> None:
        olmoearth_v1(weights=weights)


class TestOlmoEarthV1UNetDecoder:
    def test_olmoearth_v1_unet_decoder(self) -> None:
        olmoearth_v1_unet_decoder()

    def test_forward(self) -> None:
        in_dim, num_classes, patch_size = 32, 5, 8
        decoder = olmoearth_v1_unet_decoder(
            in_dim=in_dim, num_classes=num_classes, patch_size=patch_size
        )
        # Patch tokens: (B, H_p, W_p, in_dim) -> logits (B, num_classes, H, W).
        x = torch.randn(2, 4, 4, in_dim)
        out = decoder(x)
        assert out.shape == (2, num_classes, 4 * patch_size, 4 * patch_size)

    def test_invalid_patch_size(self) -> None:
        with pytest.raises(ValueError, match='patch_size must be a power of two'):
            olmoearth_v1_unet_decoder(patch_size=6)
