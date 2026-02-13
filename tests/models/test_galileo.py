# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.models import Galileo, Galileo_Weights, galileo
from torchgeo.models.galileo import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
)


class TestGalileo:
    @pytest.fixture
    def inputs(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, height, width, timesteps = 2, 4, 4, 3
        s_t_x = torch.randn(batch_size, height, width, timesteps, len(SPACE_TIME_BANDS))
        sp_x = torch.randn(batch_size, height, width, len(SPACE_BANDS))
        t_x = torch.randn(batch_size, timesteps, len(TIME_BANDS))
        st_x = torch.randn(batch_size, len(STATIC_BANDS))

        s_t_m = torch.zeros(
            batch_size,
            height,
            width,
            timesteps,
            len(SPACE_TIME_BANDS_GROUPS_IDX),
            dtype=torch.long,
        )
        sp_m = torch.zeros(
            batch_size, height, width, len(SPACE_BAND_GROUPS_IDX), dtype=torch.long
        )
        t_m = torch.zeros(
            batch_size, timesteps, len(TIME_BAND_GROUPS_IDX), dtype=torch.long
        )
        st_m = torch.zeros(batch_size, len(STATIC_BAND_GROUPS_IDX), dtype=torch.long)
        months = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)

        return s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months

    @pytest.fixture(params=[*Galileo_Weights])
    def weights(self, request: SubRequest) -> Galileo_Weights:
        return request.param  # type: ignore[no-any-return]

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, load_state_dict_from_url: None
    ) -> Galileo_Weights:
        weights = Galileo_Weights.GALILEO_NANO
        path = tmp_path / f'{weights}.pth'
        model = Galileo()
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights  # type: ignore[no-any-return]

    def test_galileo(
        self,
        inputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        model = Galileo(
            max_patch_size=2,
            embedding_size=32,
            depth=1,
            mlp_ratio=2,
            num_heads=4,
            max_sequence_length=4,
            freeze_projections=False,
            drop_path=0.0,
        )

        outputs = model(*inputs, patch_size=2)

        assert len(outputs) == 9
        s_t_x, sp_x, t_x, st_x = outputs[0], outputs[1], outputs[2], outputs[3]
        assert s_t_x.shape == torch.Size(
            [2, 2, 2, 3, len(SPACE_TIME_BANDS_GROUPS_IDX), 32]
        )
        assert sp_x.shape == torch.Size([2, 2, 2, len(SPACE_BAND_GROUPS_IDX), 32])
        assert t_x.shape == torch.Size([2, 3, len(TIME_BAND_GROUPS_IDX), 32])
        assert st_x.shape == torch.Size([2, len(STATIC_BAND_GROUPS_IDX), 32])

    def test_galileo_no_weights(self) -> None:
        galileo()

    def test_galileo_weights(self, mocked_weights: Galileo_Weights) -> None:
        galileo(weights=mocked_weights)

    @pytest.mark.slow
    def test_galileo_download(self, weights: Galileo_Weights) -> None:
        galileo(weights=weights)
