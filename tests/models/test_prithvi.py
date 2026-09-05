# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import Prithvi, PrithviV2_Weights, prithvi_eo_v2_300


def save_model(model: Prithvi, path: Path) -> None:
    """Save a model in the format of the upstream Prithvi-EO checkpoints."""
    state_dict = {'encoder.' + k: v for k, v in model.state_dict().items()}
    state_dict['decoder.embed.weight'] = torch.zeros(4, 4)
    state_dict['decoder.embed.bias'] = torch.zeros(4)
    torch.save(state_dict, path)


class TestPrithvi:
    def test_prithvi(self) -> None:
        model = Prithvi(
            img_size=32,
            patch_size=(1, 8, 8),
            num_frames=2,
            in_chans=6,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )
        assert model.patch_size == (1, 8, 8)
        assert model.img_size == (32, 32)
        assert model.out_channels == [32 * 2] * 2
        x = torch.randn(2, 6, 2, 32, 32)
        features = model(x)
        assert len(features) == 2
        for feature in features:
            assert feature.shape == (2, 1 + 2 * 4 * 4, 32)
        assert torch.allclose(features[-1].mean(dim=-1), torch.zeros(2, 33), atol=1e-5)

    def test_4d_input(self) -> None:
        model = Prithvi(
            img_size=32,
            patch_size=(1, 8, 8),
            num_frames=1,
            in_chans=6,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )
        x4 = torch.randn(2, 6, 32, 32)
        with torch.no_grad():
            features = model(x4)
            expected = model(x4.unsqueeze(2))
        for feature, exp in zip(features, expected):
            torch.testing.assert_close(feature, exp)

    def test_temporal_encoding(self) -> None:
        model = Prithvi(
            img_size=32,
            patch_size=(1, 8, 8),
            num_frames=2,
            in_chans=6,
            embed_dim=32,
            depth=2,
            num_heads=4,
            coords_encoding=['time'],
        )
        x = torch.randn(2, 6, 2, 32, 32)
        coords = torch.randn(2, 2, 2)
        with torch.no_grad():
            features = model(x, temporal_coords=coords)
            expected = model(x)
        for feature, exp in zip(features, expected):
            assert feature.shape == exp.shape
            assert not torch.allclose(feature, exp)
        embedding = model.temporal_embed_enc(coords)
        assert embedding.shape == (2, 2, 32)

    def test_location_encoding(self) -> None:
        model = Prithvi(
            img_size=32,
            patch_size=(1, 8, 8),
            num_frames=2,
            in_chans=6,
            embed_dim=32,
            depth=2,
            num_heads=4,
            coords_encoding=['location'],
        )
        x = torch.randn(2, 6, 2, 32, 32)
        coords = torch.randn(2, 2)
        with torch.no_grad():
            features = model(x, location_coords=coords)
            expected = model(x)
        for feature, exp in zip(features, expected):
            assert feature.shape == exp.shape
            assert not torch.allclose(feature, exp)

    @pytest.mark.parametrize('coords_scale_learn', [True, False])
    def test_coords_scale_learn(self, coords_scale_learn: bool) -> None:
        model = Prithvi(
            img_size=32,
            patch_size=(1, 8, 8),
            num_frames=2,
            in_chans=6,
            embed_dim=32,
            depth=2,
            num_heads=4,
            coords_encoding=['time', 'location'],
            coords_scale_learn=coords_scale_learn,
        )
        scales = [
            module.scale
            for module in (model.temporal_embed_enc, model.location_embed_enc)
        ]
        if coords_scale_learn:
            for scale in scales:
                assert scale in set(model.parameters())
        else:
            for scale in scales:
                assert scale in set(model.buffers())

    def test_patch_size_int(self) -> None:
        model = Prithvi(
            img_size=32,
            patch_size=8,
            num_frames=2,
            in_chans=6,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )
        assert model.patch_size == (1, 8, 8)

    def test_interpolate_pos_embedding(self) -> None:
        model = Prithvi(
            img_size=32,
            patch_size=(1, 8, 8),
            num_frames=2,
            in_chans=6,
            embed_dim=32,
            depth=2,
            num_heads=4,
            coords_encoding=['time', 'location'],
        )
        coords = torch.randn(2, 3, 2)
        location = torch.randn(2, 2)
        with torch.no_grad():
            # Changed number of frames and square grid
            features = model(torch.randn(2, 6, 3, 32, 32), coords, location)
            assert features[-1].shape == (2, 1 + 3 * 4 * 4, 32)
            # Changed spatial size
            features = model(torch.randn(2, 6, 2, 16, 24))
            assert features[-1].shape == (2, 1 + 2 * 2 * 3, 32)
            # Changed number of frames and spatial size
            features = model(torch.randn(2, 6, 4, 16, 24))
            assert features[-1].shape == (2, 1 + 4 * 2 * 3, 32)

    def test_patch_embed_warning(self) -> None:
        model = Prithvi(
            img_size=32,
            patch_size=(1, 8, 8),
            num_frames=2,
            in_chans=6,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )
        x = torch.randn(2, 6, 2, 30, 32)
        with pytest.warns(UserWarning, match='not divisible by patch size'):
            features = model(x)
        assert features[-1].shape == (2, 1 + 2 * 3 * 4, 32)

    def test_invalid_embed_dim(self) -> None:
        with pytest.raises(ValueError, match='divisible by 16'):
            Prithvi(
                img_size=32,
                patch_size=(1, 8, 8),
                num_frames=2,
                in_chans=6,
                embed_dim=36,
                depth=2,
                num_heads=4,
            )

    def test_invalid_patch_size(self) -> None:
        with pytest.raises(ValueError, match='bigger than input size'):
            Prithvi(
                img_size=4,
                patch_size=(1, 8, 8),
                num_frames=2,
                in_chans=6,
                embed_dim=32,
                depth=2,
                num_heads=4,
            )

    def test_prepare_features_for_image_model(self) -> None:
        model = Prithvi(
            img_size=32,
            patch_size=(1, 8, 8),
            num_frames=2,
            in_chans=6,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )
        x = torch.randn(2, 6, 2, 32, 32)
        features = model(x)
        features = model.prepare_features_for_image_model(features)
        assert features[0].shape == (2, 2 * 32, 4, 4)

    def test_float64(self) -> None:
        model = Prithvi(
            img_size=32,
            patch_size=(1, 8, 8),
            num_frames=2,
            in_chans=6,
            embed_dim=32,
            depth=2,
            num_heads=4,
        )
        model.double()
        x = torch.randn(2, 6, 2, 32, 32, dtype=torch.float64)
        features = model(x)
        assert features[-1].dtype == torch.float64


@pytest.mark.xdist_group('memory_intensive')
class TestPrithviEOV2_300:
    @pytest.fixture(params=[*PrithviV2_Weights])
    def weights(self, request: SubRequest) -> PrithviV2_Weights:
        return request.param

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> PrithviV2_Weights:
        weights = PrithviV2_Weights.EO_V2_300
        path = tmp_path / f'{weights}.pth'
        model = prithvi_eo_v2_300()
        save_model(model, path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_prithvi_eo_v2_300(self) -> None:
        prithvi_eo_v2_300()

    def test_prithvi_eo_v2_300_weights(self, mocked_weights: PrithviV2_Weights) -> None:
        prithvi_eo_v2_300(weights=mocked_weights)

    @pytest.mark.slow
    def test_prithvi_eo_v2_300_download(self, weights: PrithviV2_Weights) -> None:
        prithvi_eo_v2_300(weights=weights)
