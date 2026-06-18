# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, NASAMarineDebris
from torchgeo.datasets.utils import (
    Executable,
    PointToBoundingBoxAdapter,
    Sample,
    boxes_to_points,
)


class BoxesToPointTargets:
    """Convert detection boxes to point targets for point-detection tests."""

    def __init__(self, box_size: int = 40) -> None:
        """Initialize the transform.

        Args:
            box_size: Proxy box size for point-detection training.
        """
        self.adapter = PointToBoundingBoxAdapter(box_size=box_size)

    def __call__(self, sample: Sample) -> Sample:
        """Convert boxes to point targets and regenerate proxy boxes."""
        boxes = sample.pop('bbox_xyxy')
        sample['points'] = boxes_to_points(boxes)
        return self.adapter(sample)


class TestNASAMarineDebris:
    @pytest.fixture
    def dataset(
        self, azcopy: Executable, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> NASAMarineDebris:
        url = os.path.join('tests', 'data', 'nasa_marine_debris')
        monkeypatch.setattr(NASAMarineDebris, 'url', url)
        transforms = nn.Identity()
        return NASAMarineDebris(tmp_path, transforms, download=True)

    def test_getitem(self, dataset: NASAMarineDebris) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert x['image'].shape[0] == 3
        assert x['bbox_xyxy'].shape[-1] == 4

    def test_point_detection_targets(
        self, azcopy: Executable, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        url = os.path.join('tests', 'data', 'nasa_marine_debris')
        monkeypatch.setattr(NASAMarineDebris, 'url', url)
        transforms = BoxesToPointTargets(box_size=40)
        dataset = NASAMarineDebris(tmp_path, transforms, download=True)

        x = dataset[0]

        assert isinstance(x['points'], torch.Tensor)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['points'].shape[-1] == 2
        assert x['bbox_xyxy'].shape[-1] == 4
        assert x['bbox_xyxy'].shape[0] == x['points'].shape[0]
        assert torch.all(x['label'] == 1)

    def test_len(self, dataset: NASAMarineDebris) -> None:
        assert len(dataset) == 5

    def test_already_downloaded(
        self, dataset: NASAMarineDebris, tmp_path: Path
    ) -> None:
        NASAMarineDebris(tmp_path, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            NASAMarineDebris(tmp_path)

    def test_plot(self, dataset: NASAMarineDebris) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction_bbox_xyxy'] = x['bbox_xyxy'].clone()
        dataset.plot(x)
        plt.close()
