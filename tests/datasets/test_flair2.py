# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import FLAIR2, DatasetNotFoundError


class TestFLAIR2:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> FLAIR2:
        # TODO: Update md5 checksums
        md5s = {"test": '73c0aba603c356b2cce9ebf952fb7be0'}
        monkeypatch.setattr(FLAIR2, "md5s", md5s)
        url = os.path.join("tests", "data", "flair2", "FLAIR2.zip")
        monkeypatch.setattr(FLAIR2, "url", url)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return FLAIR2(root, split, transforms, download=True, checksum=True)
    
    def test_per_band_statistics(self, dataset: FLAIR2) -> None:
        if dataset.split != 'train': return
        
        mins, maxs, means, stdvs = dataset.per_band_statistics()
        for stats in [mins, maxs, means, stdvs]:
            assert isinstance(stats, list)
            assert len(stats) == dataset.get_num_bands()
            assert all(isinstance(stat, float) for stat in stats)
    
    def test_getitem(self, dataset: FLAIR2) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape == (len(dataset.all_bands), 512, 512)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].shape[-2:] == x['mask_zones'].shape[-2:]
    
    def test_get_num_bands(self, dataset: FLAIR2) -> None:
        assert dataset.get_num_bands() == len(dataset.all_bands)

    def test_len(self, dataset: FLAIR2) -> None:
        # TODO: find out how many samples are in the dataset
        if dataset.split == 'train':
            assert len(dataset) == 3
        else:
            assert len(dataset) == 3

    def test_already_downloaded(self, dataset: FLAIR2) -> None:
        FLAIR2(root=dataset.root)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        filename = 'FLAIR2.zip'
        dir = os.path.join('tests', 'data', 'caffe')
        shutil.copyfile(
            os.path.join(dir, filename), os.path.join(str(tmp_path), filename)
        )
        FLAIR2(root=str(tmp_path))

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            FLAIR2(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FLAIR2(tmp_path)

    def test_plot(self, dataset: FLAIR2) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = torch.clone(sample['mask_zones'])
        dataset.plot(sample, suptitle='Prediction')
        plt.close()