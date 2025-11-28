# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from collections.abc import Sized
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from torchgeo.datamodules import CloudCoverDetectionDataModule
from torchgeo.datasets import CloudCoverDetection
from torchgeo.datasets.utils import Executable, which


@pytest.fixture
def azcopy(monkeypatch: MonkeyPatch) -> Executable:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'datasets')
    path = os.path.normpath(path)
    monkeypatch.setenv('PATH', path, prepend=os.pathsep)
    return which('azcopy')


@pytest.fixture
def datamodule(
    tmp_path: Path, monkeypatch: MonkeyPatch, azcopy: Executable
) -> CloudCoverDetectionDataModule:
    url = os.path.join('tests', 'data', 'ref_cloud_cover_detection_challenge_v1')
    monkeypatch.setattr(CloudCoverDetection, 'url', url)
    dm = CloudCoverDetectionDataModule(
        root=tmp_path, batch_size=2, num_workers=0, val_split_pct=0.5, download=True
    )
    return dm


class TestCloudCoverDetectionDataModule:
    def test_invalid_val_split_pct(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='val_split_pct'):
            CloudCoverDetectionDataModule(root=tmp_path, val_split_pct=0.0)
        with pytest.raises(ValueError, match='val_split_pct'):
            CloudCoverDetectionDataModule(root=tmp_path, val_split_pct=1.0)

    def test_setup_fit(self, datamodule: CloudCoverDetectionDataModule) -> None:
        datamodule.setup('fit')
        assert datamodule.dataset is not None
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None

        dataset = datamodule.dataset
        train_dataset = datamodule.train_dataset
        val_dataset = datamodule.val_dataset

        assert isinstance(dataset, Sized)
        assert isinstance(train_dataset, Sized)
        assert isinstance(val_dataset, Sized)

        assert len(train_dataset) + len(val_dataset) == len(dataset)

    def test_setup_test(self, datamodule: CloudCoverDetectionDataModule) -> None:
        datamodule.setup('test')
        assert datamodule.test_dataset is not None
        test_dataset = datamodule.test_dataset
        assert isinstance(test_dataset, Sized)
        assert len(test_dataset) == 1
