# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from torch import nn

from torchgeo.datasets import DatasetNotFoundError, FMoW


class TestFMoW:
    def test_classes(self) -> None:
        assert len(FMoW.classes) == 62
        assert FMoW.classes[3] == 'amusement_park'

    @pytest.fixture(params=['train', 'val'])
    def dataset(self, tmp_path: Path, request: SubRequest) -> FMoW:
        split = request.param
        src_dir = os.path.join('tests', 'data', 'fmow', split)
        dst_dir = os.path.join(str(tmp_path), split)
        shutil.copytree(src_dir, dst_dir)

        transforms = nn.Identity()
        return FMoW(root=tmp_path, split=split, transforms=transforms)

    def test_getitem(self, dataset: FMoW) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert x['image'].shape == (3, 32, 32)
        assert x['image'].dtype == torch.float32
        assert x['label'].item() == 0
        assert x['bbox_xyxy'].dtype == torch.float32
        assert torch.equal(
            x['bbox_xyxy'], torch.tensor([[1.0, 2.0, 4.0, 6.0], [5.0, 6.0, 12.0, 14.0]])
        )

    def test_len(self, dataset: FMoW) -> None:
        assert len(dataset) == 1

    def test_empty_bbox(self, tmp_path: Path) -> None:
        src_dir = os.path.join('tests', 'data', 'fmow', 'train')
        dst_dir = os.path.join(str(tmp_path), 'train')
        shutil.copytree(src_dir, dst_dir)

        json_path = os.path.join(
            dst_dir, 'airport', 'airport_0', 'airport_0_0_rgb.json'
        )
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'bounding_boxes': []}, f)

        ds = FMoW(root=tmp_path, split='train')
        assert ds[0]['bbox_xyxy'].shape == (0, 4)

    def test_missing_metadata(self, dataset: FMoW) -> None:
        Path(dataset.image_paths[0]).with_suffix('.json').unlink()
        with pytest.raises(FileNotFoundError):
            dataset[0]

    def test_invalid_metadata(self, dataset: FMoW) -> None:
        path = Path(dataset.image_paths[0]).with_suffix('.json')
        path.write_text('{}', encoding='utf-8')
        with pytest.raises(KeyError, match='bounding_boxes'):
            dataset[0]

    def test_invalid_split(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='Split must be one of'):
            FMoW(root=tmp_path, split='test')  # ty: ignore[invalid-argument-type]

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FMoW(root=tmp_path, split='train')

    def test_false_detection_ignored(self, tmp_path: Path) -> None:
        src_dir = Path('tests', 'data', 'fmow', 'train', 'airport', 'airport_0')
        dst_dir = tmp_path / 'train' / 'false_detection' / 'false_detection_0'
        shutil.copytree(src_dir, dst_dir)

        with pytest.raises(DatasetNotFoundError):
            FMoW(root=tmp_path, split='train')

    def test_plot(self, dataset: FMoW) -> None:
        x = dataset[0].copy()
        fig = dataset.plot(x, suptitle='Test')
        assert fig.axes[0].get_title() == 'Label: airport'
        assert fig.get_suptitle() == 'Test'
        assert len(fig.axes[0].patches) == 2
        plt.close(fig)

        fig = dataset.plot(x, show_titles=False)
        assert fig.axes[0].get_title() == ''
        plt.close(fig)

        x['prediction'] = x['label'].clone()
        fig = dataset.plot(x)
        assert fig.axes[0].get_title() == 'Label: airport\nPrediction: airport'
        plt.close(fig)
