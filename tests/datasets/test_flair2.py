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
        md5s = {
            "flair-2_centroids_sp_to_patch": "c308ca502fbe23ef6eaa2efc33e7b199",
            "flair_aerial_train": "ace5712f03f799e9f3e0cc74c27d84c8",
            "flair_sen_train": "7468da7cda8d0c0c1120dec8dd857a1f",
            "flair_labels_train": "87bc5bbeb2d04b788fe733e353fa7112",
            "flair_2_aerial_test": "e5a6d4f5c812edc6cf59bff656fd81b6",
            "flair_2_sen_test": "ae230be3a50c02e62606632b7d72b48b",
            "flair_2_labels_test": "644d81311d0ac2aaecfdc5721dcd03f1",
        }

        monkeypatch.setattr(FLAIR2, "md5s", md5s)
        url_prefix = os.path.join("tests", "data", "flair2", "FLAIR2")
        monkeypatch.setattr(FLAIR2, "url_prefix", url_prefix)
        
        root = tmp_path
        split = request.param
        bands = ("B01", "B02", "B03", "B04", "B05")
        transforms = nn.Identity()
        
        return FLAIR2(root, split, bands, transforms, download=True, checksum=True)
    
    def test_get_num_bands(self, dataset: FLAIR2) -> None:
        assert dataset.get_num_bands() == len(dataset.all_bands)
        
    def test_per_band_statistics(self, dataset: FLAIR2) -> None:
        if dataset.split != 'train': return
        
        mins, maxs, means, stdvs = dataset.per_band_statistics(dataset.split)
        for stats in [mins, maxs, means, stdvs]:
            assert isinstance(stats, list)
            assert stats.__len__() == dataset.get_num_bands()
            assert all(isinstance(stat, float) for stat in stats)
    
    def test_getitem(self, dataset: FLAIR2) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape == (len(dataset.all_bands), 512, 512)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].shape[-2:] == x['mask'].shape[-2:]

    def test_len(self, dataset: FLAIR2) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 10
        else:
            assert len(dataset) == 5

    def test_already_downloaded(self, dataset: FLAIR2) -> None:
        FLAIR2(root=dataset.root, split=dataset.split)

    def test_not_yet_extracted(self, dataset: FLAIR2, tmp_path: Path) -> None:
        filenames = list(dataset.dir_names[dataset.split].values())
        filenames.append(dataset.centroids_file)
        dir = os.path.join('tests', 'data', 'flair2', "FLAIR2")
        for filename in filenames:
            shutil.copyfile(
                os.path.join(dir, f"{filename}.zip"), os.path.join(str(tmp_path), f"{filename}.zip")
            )
            
        FLAIR2(root=str(tmp_path), split=dataset.split)

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
        sample['prediction'] = torch.clone(sample['mask'])
        dataset.plot(sample, suptitle='Prediction')
        plt.close()

    # TODO: test toy option