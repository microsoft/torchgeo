# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import importlib
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    OlmoEarthV1_1_Weights,
    OlmoEarthV1_2_Weights,
    OlmoEarthV1_Weights,
    olmoearth_v1,
    olmoearth_v1_1,
    olmoearth_v1_2,
    olmoearth_v1_unet_decoder,
)

pytest.importorskip('olmoearth_pretrain_minimal')


class TestOlmoEarth:
    @pytest.fixture(
        params=[
            (olmoearth_v1, OlmoEarthV1_Weights),
            (olmoearth_v1_1, OlmoEarthV1_1_Weights),
            (olmoearth_v1_2, OlmoEarthV1_2_Weights),
        ]
    )
    def family(
        self, request: SubRequest
    ) -> tuple[Callable[..., nn.Module], type[WeightsEnum]]:
        return request.param

    @pytest.fixture(
        params=[
            (builder, weights)
            for builder, enum in (
                (olmoearth_v1, OlmoEarthV1_Weights),
                (olmoearth_v1_1, OlmoEarthV1_1_Weights),
                (olmoearth_v1_2, OlmoEarthV1_2_Weights),
            )
            for weights in enum
        ]
    )
    def weights(
        self, request: SubRequest
    ) -> tuple[Callable[..., nn.Module], WeightsEnum]:
        return request.param

    def test_random_init(
        self, family: tuple[Callable[..., nn.Module], type[WeightsEnum]]
    ) -> None:
        builder, _ = family
        builder()

    def test_every_weight_is_pinned(
        self, family: tuple[Callable[..., nn.Module], type[WeightsEnum]]
    ) -> None:
        """Each entry must pin a commit so its architecture and weights cannot change."""
        _, enum = family
        for weights in enum:
            revision = weights.meta['revision']
            assert re.fullmatch(r'[0-9a-f]{40}', revision), weights
            assert f'/resolve/{revision}/' in weights.url, weights

    def test_every_weight_carries_digests(
        self, family: tuple[Callable[..., nn.Module], type[WeightsEnum]]
    ) -> None:
        """Each entry must record a sha256 for both artifacts it downloads."""
        _, enum = family
        for weights in enum:
            for key in ('config_sha256', 'weights_sha256'):
                assert re.fullmatch(r'[0-9a-f]{64}', weights.meta[key]), (weights, key)

    def test_pinned_url_and_digests_are_used(
        self,
        family: tuple[Callable[..., nn.Module], type[WeightsEnum]],
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Both artifacts must be fetched at the pinned commit with their own digest."""
        builder, enum = family
        weights = next(iter(enum))
        requested: list[tuple[str, str | None]] = []

        def fake_download(url: str, dst: str, hash_prefix: str | None = None) -> None:
            requested.append((url, hash_prefix))
            Path(dst).write_text('{}')

        monkeypatch.setattr(torch.hub, 'get_dir', lambda: str(tmp_path))
        monkeypatch.setattr(torch.hub, 'download_url_to_file', fake_download)
        olmoearth = importlib.import_module('olmoearth_pretrain_minimal')
        monkeypatch.setattr(
            olmoearth,
            'load_model_from_path',
            lambda directory: olmoearth.OlmoEarthPretrain_v1(model_size='nano'),
        )

        builder(weights=weights)
        repo = weights.meta['hf_repo']
        revision = weights.meta['revision']
        base = f'https://huggingface.co/{repo}/resolve/{revision}'
        assert requested == [
            (f'{base}/config.json', weights.meta['config_sha256']),
            (f'{base}/weights.pth', weights.meta['weights_sha256']),
        ]

    @pytest.mark.slow
    def test_download(
        self, weights: tuple[Callable[..., nn.Module], WeightsEnum]
    ) -> None:
        builder, w = weights
        builder(weights=w)


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
