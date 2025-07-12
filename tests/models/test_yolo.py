# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torchvision.models._api import WeightsEnum

from torchgeo.models import YOLO_Weights, yolo

pytest.importorskip('ultralytics', minversion='8.3')


class TestYOLO:
    @pytest.fixture(params=[*YOLO_Weights])
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

    @pytest.fixture
    def mocked_weights(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> WeightsEnum:
        import ultralytics

        weights = YOLO_Weights.DELINEATE_ANYTHING
        path = tmp_path / f'{weights}.pth'
        model = ultralytics.YOLO(
            model=f'{weights.meta["model"]}.yaml', task=weights.meta['task']
        )
        model.ckpt = model.state_dict()
        model.save(path)
        monkeypatch.setattr(weights.value, 'url', str(path))
        return weights

    def test_yolo(self) -> None:
        yolo(weights=None, model='yolo11n.yaml', task='segment')

    def test_yolo_weights(self, mocked_weights: WeightsEnum) -> None:
        yolo(weights=mocked_weights)

    def test_bands(self, weights: WeightsEnum) -> None:
        if 'bands' in weights.meta:
            assert len(weights.meta['bands']) == weights.meta['in_chans']

    def test_transforms(self, weights: WeightsEnum) -> None:
        c = weights.meta['in_chans']
        sample = {
            'image': torch.arange(c * 256 * 256, dtype=torch.float).view(c, 256, 256)
        }
        weights.transforms(sample)

    @pytest.mark.slow
    def test_yolo_download(self, weights: WeightsEnum) -> None:
        yolo(weights=weights)
