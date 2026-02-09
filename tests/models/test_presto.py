# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import Presto, Presto_Weights, presto
from torchgeo.models.presto import (
    BANDS_GROUPS_IDX,
    NUM_DYNAMIC_WORLD_CLASSES,
    get_sinusoid_encoding_table,
    month_to_tensor,
)


@pytest.fixture
def make_inputs() -> Callable[
    [int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    def _make_inputs(
        batch_size: int, seq_len: int, channels: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(batch_size, seq_len, channels)
        dynamic_world = torch.zeros(batch_size, seq_len, dtype=torch.long)
        latlons = torch.tensor([[0.0, 0.0], [10.0, -20.0]])[:batch_size]
        mask = torch.zeros_like(x)
        return x, dynamic_world, latlons, mask

    return _make_inputs


class TestPrestoHelpers:
    def test_get_sinusoid_encoding_table(self) -> None:
        device = torch.device('cpu')
        table = get_sinusoid_encoding_table(positions=[0, 1, 2], device=device, d_hid=4)
        assert table.shape == torch.Size([3, 4])

    def test_month_to_tensor_2d(self) -> None:
        device = torch.device('cpu')
        month_tensor_2d = torch.tensor([[1, 2, 3]])
        assert torch.equal(
            month_to_tensor(month_tensor_2d, batch_size=1, seq_len=3, device=device),
            month_tensor_2d,
        )


class TestPresto:
    @pytest.fixture(params=[*Presto_Weights])
    def weights(self, request: SubRequest) -> Presto_Weights:
        return request.param  # type: ignore[no-any-return]

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Presto_Weights:
        weights = Presto_Weights.PRESTO
        path = tmp_path / f'{weights}.pth'
        model = Presto()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights  # type: ignore[no-any-return]

    def test_presto(
        self,
        make_inputs: Callable[
            [int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ) -> None:
        model = Presto(
            encoder_embedding_size=16,
            channel_embed_ratio=0.25,
            month_embed_ratio=0.25,
            encoder_depth=1,
            mlp_ratio=2,
            encoder_num_heads=2,
            decoder_embedding_size=16,
            decoder_depth=1,
            decoder_num_heads=2,
            max_sequence_length=4,
        )
        channels = sum(len(group) for group in BANDS_GROUPS_IDX.values())
        x, dynamic_world, latlons, _ = make_inputs(2, 3, channels)
        reconstructed, dw_output = model(x, dynamic_world, latlons)
        assert reconstructed.shape == torch.Size([2, 3, channels])
        assert dw_output is not None
        assert dw_output.shape == torch.Size([2, 3, NUM_DYNAMIC_WORLD_CLASSES])

    def test_presto_with_mask(
        self,
        make_inputs: Callable[
            [int, int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ) -> None:
        model = Presto(
            encoder_embedding_size=16,
            channel_embed_ratio=0.25,
            month_embed_ratio=0.25,
            encoder_depth=1,
            mlp_ratio=2,
            encoder_num_heads=2,
            decoder_embedding_size=16,
            decoder_depth=1,
            decoder_num_heads=2,
            max_sequence_length=4,
        )
        channels = sum(len(group) for group in BANDS_GROUPS_IDX.values())
        x, dynamic_world, latlons, mask = make_inputs(2, 3, channels)
        mask[:, 0, :] = 1
        reconstructed, dw_output = model(
            x, dynamic_world, latlons, mask=mask, month=torch.tensor([0, 2])
        )
        assert reconstructed.shape == torch.Size([2, 3, channels])
        assert dw_output is not None
        assert dw_output.shape == torch.Size([2, 3, NUM_DYNAMIC_WORLD_CLASSES])

    def test_presto_no_weights(self) -> None:
        presto()

    def test_presto_weights(self, mocked_weights: Presto_Weights) -> None:
        presto(weights=mocked_weights)

    @pytest.mark.slow
    def test_presto_download(self, weights: Presto_Weights) -> None:
        presto(weights=weights)
