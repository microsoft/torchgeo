# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import (
    CloudCoverDetection,
    DatasetNotFoundError,
    RGBBandsMissingError,
)
from torchgeo.datasets.utils import Executable


class TestCloudCoverDetection:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self,
        azcopy: Executable,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        request: SubRequest,
    ) -> CloudCoverDetection:
        url = os.path.join('tests', 'data', 'ref_cloud_cover_detection_challenge_v1')
        monkeypatch.setattr(CloudCoverDetection, 'url', url)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return CloudCoverDetection(
            root=root, split=split, transforms=transforms, download=True
        )

    def test_invalid_band(self, dataset: CloudCoverDetection) -> None:
        with pytest.raises(AssertionError):
            CloudCoverDetection(root=dataset.root, split=dataset.split, bands=['B09'])

    def test_getitem(self, dataset: CloudCoverDetection) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: CloudCoverDetection) -> None:
        assert len(dataset) == 1

    def test_already_downloaded(self, dataset: CloudCoverDetection) -> None:
        CloudCoverDetection(root=dataset.root, split=dataset.split, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CloudCoverDetection(tmp_path)

    def test_plot(self, dataset: CloudCoverDetection) -> None:
        sample = dataset[0]
        dataset.plot(sample, suptitle='Test')
        plt.close()
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='Pred')
        plt.close()

    def test_plot_rgb(self, dataset: CloudCoverDetection) -> None:
        dataset = CloudCoverDetection(
            root=dataset.root, split=dataset.split, bands=['B08'], download=True
        )
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            dataset.plot(dataset[0], suptitle='Single Band')
