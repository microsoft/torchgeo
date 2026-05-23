# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import VHR10, DatasetNotFoundError


class TestVHR10:
    @pytest.fixture(params=['positive', 'negative'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> VHR10:
        url = os.path.join('tests', 'data', 'vhr10', 'NWPU VHR-10 dataset.zip')
        monkeypatch.setitem(VHR10.image_meta, 'url', url)
        url = os.path.join('tests', 'data', 'vhr10', 'annotations.json')
        monkeypatch.setitem(VHR10.target_meta, 'url', url)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return VHR10(root, split, transforms, download=True)

    def test_getitem(self, dataset: VHR10) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        if dataset.split == 'positive':
            assert isinstance(x['label'], torch.Tensor)
            assert isinstance(x['bbox_xyxy'], torch.Tensor)
            assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: VHR10) -> None:
        if dataset.split == 'positive':
            assert len(dataset) == 650
        elif dataset.split == 'negative':
            assert len(dataset) == 150

    def test_already_downloaded(self, dataset: VHR10) -> None:
        VHR10(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            VHR10(tmp_path)

    def test_plot(self, dataset: VHR10) -> None:
        x = dataset[0]
        dataset.plot(x, show_titles=False, suptitle='Test')
        plt.close()
        if dataset.split == 'positive':
            x['prediction_label'] = x['label']
            x['prediction_bbox_xyxy'] = x['bbox_xyxy']
            x['prediction_mask'] = x['mask']
            scores = [0.3, 0.7]
            for score in scores:
                x['prediction_score'] = torch.tensor([score])
                dataset.plot(x)
                plt.close()
