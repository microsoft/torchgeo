# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch
from _pytest.fixtures import SubRequest

from torchgeo.models import (
    OlmoEarthBase_Weights,
    OlmoEarthLarge_Weights,
    OlmoEarthNano_Weights,
    OlmoEarthSmall_Weights,
    OlmoEarthTiny_Weights,
    olmoearth_base,
    olmoearth_large,
    olmoearth_nano,
    olmoearth_small,
    olmoearth_tiny,
    olmoearth_v1_unet_decoder,
)

pytest.importorskip('olmoearth_pretrain_minimal')


class TestOlmoEarthNano:
    @pytest.fixture(params=[*OlmoEarthNano_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthNano_Weights:
        return request.param

    def test_olmoearth(self) -> None:
        olmoearth_nano()

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthNano_Weights) -> None:
        olmoearth_nano(weights=weights)


class TestOlmoEarthTiny:
    @pytest.fixture(params=[*OlmoEarthTiny_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthTiny_Weights:
        return request.param

    def test_olmoearth(self) -> None:
        olmoearth_tiny()

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthTiny_Weights) -> None:
        olmoearth_tiny(weights=weights)


class TestOlmoEarthSmall:
    @pytest.fixture(params=[*OlmoEarthSmall_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthSmall_Weights:
        return request.param

    def test_olmoearth(self) -> None:
        olmoearth_small()

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthSmall_Weights) -> None:
        olmoearth_small(weights=weights)


class TestOlmoEarthBase:
    @pytest.fixture(params=[*OlmoEarthBase_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthBase_Weights:
        return request.param

    def test_olmoearth(self) -> None:
        olmoearth_base()

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthBase_Weights) -> None:
        olmoearth_base(weights=weights)


class TestOlmoEarthLarge:
    @pytest.fixture(params=[*OlmoEarthLarge_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthLarge_Weights:
        return request.param

    def test_olmoearth(self) -> None:
        olmoearth_large()

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthLarge_Weights) -> None:
        olmoearth_large(weights=weights)


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
