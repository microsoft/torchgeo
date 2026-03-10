# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, SentinelKilnDB

pytest.importorskip('pyarrow')


class TestSentinelKilnDB:
    @pytest.fixture(
        params=product(['train', 'val', 'test'], ['horizontal', 'oriented'])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SentinelKilnDB:
        url = os.path.join('tests', 'data', 'sentinelkilndb', '{}.parquet')
        monkeypatch.setattr(SentinelKilnDB, 'url', url)

        root = tmp_path
        split, bbox_orientation = request.param

        transforms = nn.Identity()

        return SentinelKilnDB(
            root,
            split=split,
            bbox_orientation=bbox_orientation,
            transforms=transforms,
            download=True,
            checksum=False,
        )

    def test_getitem(self, dataset: SentinelKilnDB) -> None:
        for i in range(len(dataset)):
            x = dataset[i]
            assert isinstance(x, dict)
            assert isinstance(x['image'], torch.Tensor)
            assert isinstance(x['labels'], torch.Tensor)

            # Check image shape (3 channels, 32x32 for test data)
            assert x['image'].shape == (3, 32, 32)

            if dataset.bbox_orientation == 'oriented':
                bbox_key = 'bbox'
            else:
                bbox_key = 'bbox_xyxy'
            assert isinstance(x[bbox_key], torch.Tensor)

            # Check bbox dimensions
            if dataset.bbox_orientation == 'oriented':
                assert x[bbox_key].ndim == 2
                if x[bbox_key].shape[0] > 0:
                    assert x[bbox_key].shape[1] == 8
            else:
                assert x[bbox_key].ndim == 2
                if x[bbox_key].shape[0] > 0:
                    assert x[bbox_key].shape[1] == 4

            assert x['labels'].shape[0] == x[bbox_key].shape[0]

    def test_len(self, dataset: SentinelKilnDB) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 4
        else:
            assert len(dataset) == 2

    def test_already_downloaded(self, dataset: SentinelKilnDB) -> None:
        SentinelKilnDB(root=dataset.root, split=dataset.split, download=True)

    def test_checksum(self, dataset: SentinelKilnDB) -> None:
        SentinelKilnDB(
            root=dataset.root, split=dataset.split, download=True, checksum=True
        )

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SentinelKilnDB(tmp_path)

    def test_invalid_split(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='not supported'):
            SentinelKilnDB(tmp_path, split='invalid')  # type: ignore[arg-type]

    def test_invalid_bbox_orientation(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='Bounding box orientation'):
            SentinelKilnDB(tmp_path, bbox_orientation='invalid')  # type: ignore[arg-type]

    def test_plot(self, dataset: SentinelKilnDB) -> None:
        # Use sample with boxes (index 1 has boxes in test data)
        x = dataset[1]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_no_boxes(self, dataset: SentinelKilnDB) -> None:
        # First sample in each split has no boxes (negative sample)
        x = dataset[0]
        dataset.plot(x, suptitle='Negative Sample')
        plt.close()
