# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    OlmoEarthBase_Weights,
    OlmoEarthLarge_Weights,
    OlmoEarthNano_Weights,
    OlmoEarthSmall_Weights,
    OlmoEarthTiny_Weights,
    OlmoEarthV1_Weights,
    olmoearth_base,
    olmoearth_large,
    olmoearth_nano,
    olmoearth_small,
    olmoearth_tiny,
    olmoearth_v1,
    olmoearth_v1_unet_decoder,
)

constants = pytest.importorskip(
    'olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants'
)


@pytest.mark.parametrize(
    'builder,weights',
    [
        *[(olmoearth_nano, w) for w in OlmoEarthNano_Weights],
        *[(olmoearth_tiny, w) for w in OlmoEarthTiny_Weights],
        *[(olmoearth_small, w) for w in OlmoEarthSmall_Weights],
        *[(olmoearth_base, w) for w in OlmoEarthBase_Weights],
        *[(olmoearth_large, w) for w in OlmoEarthLarge_Weights],
    ],
)
def test_olmoearth_meta(
    builder: Callable[..., nn.Module], weights: WeightsEnum
) -> None:
    """The weights metadata matches the architecture the library builds."""
    meta = weights.meta
    encoder = builder(model_version=meta['model_version']).model.encoder
    assert encoder.embedding_size == meta['embed_dim']
    blocks = encoder.blocks
    assert isinstance(blocks, nn.ModuleList)
    assert len(blocks) == meta['depth']
    assert blocks[0].attn.num_heads == meta['num_heads']
    assert (encoder.min_patch_size, encoder.max_patch_size) == meta['patch_size']
    assert encoder.max_sequence_length == meta['max_sequence_length']
    for modality in meta['modalities']:
        spec = constants.Modality.get(modality)
        assert meta['bands'][modality] == list(spec.band_order)


class TestOlmoEarthNano:
    @pytest.fixture(params=[*OlmoEarthNano_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthNano_Weights:
        return request.param

    @pytest.fixture(params=[*OlmoEarthNano_Weights])
    def mocked_weights(
        self,
        request: SubRequest,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        load_state_dict_from_url: None,
    ) -> OlmoEarthNano_Weights:
        weights = request.param
        path = tmp_path / f'{weights}.pth'
        model = olmoearth_nano(model_version=weights.meta['model_version'])
        torch.save(model.model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_olmoearth(self) -> None:
        olmoearth_nano()

    def test_olmoearth_weights(self, mocked_weights: OlmoEarthNano_Weights) -> None:
        olmoearth_nano(weights=mocked_weights)

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthNano_Weights) -> None:
        olmoearth_nano(weights=weights)


class TestOlmoEarthTiny:
    @pytest.fixture(params=[*OlmoEarthTiny_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthTiny_Weights:
        return request.param

    @pytest.fixture(params=[*OlmoEarthTiny_Weights])
    def mocked_weights(
        self,
        request: SubRequest,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        load_state_dict_from_url: None,
    ) -> OlmoEarthTiny_Weights:
        weights = request.param
        path = tmp_path / f'{weights}.pth'
        model = olmoearth_tiny(model_version=weights.meta['model_version'])
        torch.save(model.model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_olmoearth(self) -> None:
        olmoearth_tiny()

    def test_olmoearth_weights(self, mocked_weights: OlmoEarthTiny_Weights) -> None:
        olmoearth_tiny(weights=mocked_weights)

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthTiny_Weights) -> None:
        olmoearth_tiny(weights=weights)


class TestOlmoEarthSmall:
    @pytest.fixture(params=[*OlmoEarthSmall_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthSmall_Weights:
        return request.param

    @pytest.fixture(params=[*OlmoEarthSmall_Weights])
    def mocked_weights(
        self,
        request: SubRequest,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        load_state_dict_from_url: None,
    ) -> OlmoEarthSmall_Weights:
        weights = request.param
        path = tmp_path / f'{weights}.pth'
        model = olmoearth_small(model_version=weights.meta['model_version'])
        torch.save(model.model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_olmoearth(self) -> None:
        olmoearth_small()

    def test_olmoearth_weights(self, mocked_weights: OlmoEarthSmall_Weights) -> None:
        olmoearth_small(weights=mocked_weights)

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthSmall_Weights) -> None:
        olmoearth_small(weights=weights)


class TestOlmoEarthBase:
    @pytest.fixture(params=[*OlmoEarthBase_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthBase_Weights:
        return request.param

    @pytest.fixture(params=[*OlmoEarthBase_Weights])
    def mocked_weights(
        self,
        request: SubRequest,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        load_state_dict_from_url: None,
    ) -> OlmoEarthBase_Weights:
        weights = request.param
        path = tmp_path / f'{weights}.pth'
        model = olmoearth_base(model_version=weights.meta['model_version'])
        torch.save(model.model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_olmoearth(self) -> None:
        olmoearth_base()

    def test_olmoearth_weights(self, mocked_weights: OlmoEarthBase_Weights) -> None:
        olmoearth_base(weights=mocked_weights)

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthBase_Weights) -> None:
        olmoearth_base(weights=weights)


class TestOlmoEarthLarge:
    @pytest.fixture(params=[*OlmoEarthLarge_Weights])
    def weights(self, request: SubRequest) -> OlmoEarthLarge_Weights:
        return request.param

    @pytest.fixture(params=[*OlmoEarthLarge_Weights])
    def mocked_weights(
        self,
        request: SubRequest,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        load_state_dict_from_url: None,
    ) -> OlmoEarthLarge_Weights:
        weights = request.param
        path = tmp_path / f'{weights}.pth'
        model = olmoearth_large(model_version=weights.meta['model_version'])
        torch.save(model.model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_olmoearth(self) -> None:
        olmoearth_large()

    def test_olmoearth_weights(self, mocked_weights: OlmoEarthLarge_Weights) -> None:
        olmoearth_large(weights=mocked_weights)

    @pytest.mark.slow
    def test_olmoearth_download(self, weights: OlmoEarthLarge_Weights) -> None:
        olmoearth_large(weights=weights)


@pytest.mark.filterwarnings(
    'ignore:Use torchgeo.models.olmoearth_nano.* instead:DeprecationWarning'
)
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
        torch.save(model.model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_olmoearth_v1(self) -> None:
        olmoearth_v1()

    def test_olmoearth_v1_deprecated(self) -> None:
        with pytest.warns(DeprecationWarning, match='olmoearth_nano'):
            olmoearth_v1()

    def test_olmoearth_v1_weights(self, mocked_weights: OlmoEarthV1_Weights) -> None:
        olmoearth_v1(weights=mocked_weights)

    def test_olmoearth_v1_weights_are_applied(
        self, mocked_weights: OlmoEarthV1_Weights
    ) -> None:
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
