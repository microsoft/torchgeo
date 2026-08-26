# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import hashlib
import importlib
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

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
from torchgeo.models import olmoearth as olmoearth_module

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

    def test_pinned_revision_is_used(
        self,
        family: tuple[Callable[..., nn.Module], type[WeightsEnum]],
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """The pinned commit must reach the Hub for both the config and the weights."""
        builder, enum = family
        weights = next(iter(enum))
        requested: list[tuple[str, str, str]] = []

        def fake_download(repo_id: str, filename: str, revision: str) -> str:
            requested.append((repo_id, filename, revision))
            return str(tmp_path / filename)

        monkeypatch.setattr(olmoearth_module, '_verified_download', fake_download)
        olmoearth = importlib.import_module('olmoearth_pretrain_minimal')
        monkeypatch.setattr(
            olmoearth,
            'load_model_from_path',
            lambda directory: olmoearth.OlmoEarthPretrain_v1(model_size='nano'),
        )

        builder(weights=weights)
        assert requested == [
            (weights.meta['hf_repo'], 'config.json', weights.meta['revision']),
            (weights.meta['hf_repo'], 'weights.pth', weights.meta['revision']),
        ]

    def test_corrupted_download_is_rejected(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """A file of the right size but wrong contents must fail.

        hf_hub_download only checks size, so this is the case it cannot catch.
        """
        path = tmp_path / 'weights.pth'
        path.write_bytes(b'pretend checkpoint')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hub = importlib.import_module('huggingface_hub')
        monkeypatch.setattr(hub, 'hf_hub_download', lambda **kwargs: str(path))
        monkeypatch.setattr(hub, 'hf_hub_url', lambda **kwargs: 'https://example')
        monkeypatch.setattr(
            hub, 'get_hf_file_metadata', lambda url: SimpleNamespace(etag=digest)
        )

        olmoearth_module._verified_download('allenai/repo', 'weights.pth', 'abc')

        path.write_bytes(b'pretend checkpoinT')  # same length, one byte different
        with pytest.raises(RuntimeError, match='failed its integrity check'):
            olmoearth_module._verified_download('allenai/repo', 'weights.pth', 'abc')

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
