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

from torchgeo.datasets import FLAIR2, FLAIR2Toy, DatasetNotFoundError


class TestFLAIR2:
    @pytest.fixture(params=[(split, init_class) for split in ["train", "test"] for init_class in [FLAIR2, FLAIR2Toy]])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> FLAIR2:
        md5s = {
            "flair_2_toy_dataset": "9a058c40635d7a25f2ffbf39f59e668c",
            "flair-2_centroids_sp_to_patch": "20c7648f423b42a0a537d1871752aca7",
            "flair_aerial_train": "411d34d0f23204a9f8d4bf5c695c4e4a",
            "flair_sen_train": "5173c991c131ef5429bb195b785a4746",
            "flair_labels_train": "6b361d14c0ef491dd37267674a238edf",
            "flair_2_aerial_test": "b7f971ef943e9c16e83c9849dd66309f",
            "flair_2_sen_test": "5c5a7fc78097340bafcee7a01ef3bf48",
            "flair_2_labels_test": "b46050736699d2f16639a5951fb1e9f8",
        }

        monkeypatch.setattr(FLAIR2, "md5s", md5s)
        url_prefix = os.path.join("tests", "data", "flair2", "FLAIR2")
        monkeypatch.setattr(FLAIR2, "url_prefix", url_prefix)
        
        root = tmp_path
        split = request.param[0]
        init_class = request.param[1]
        bands = ("B01", "B02", "B03", "B04", "B05")
        transforms = nn.Identity()
                
        return init_class(root, split, bands, transforms, download=True, checksum=True, use_sentinel=True)
    
    def test_get_num_bands(self, dataset: FLAIR2) -> None:
        assert dataset.get_num_bands() == len(dataset.all_bands)
        
    def test_per_band_statistics(self, dataset: FLAIR2) -> None:
        if dataset.split != 'train':
            return
        
        mins, maxs, means, stdvs = dataset.per_band_statistics(dataset.split)
        for stats in [mins, maxs, means, stdvs]:
            assert isinstance(stats, list)
            assert stats.__len__() == dataset.get_num_bands()
            assert all(isinstance(stat, float) for stat in stats)
    
    @pytest.mark.parametrize("use_sentinel", [True, False])
    def test_getitem(self, dataset: FLAIR2, use_sentinel: bool) -> None:
        dataset.use_sentinel = use_sentinel
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape == (len(dataset.all_bands), 32, 32)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].shape[-2:] == x['mask'].shape[-2:]

    def test_len(self, dataset: FLAIR2) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 10
        else:
            assert len(dataset) == 5

    def test_already_downloaded(self, dataset: FLAIR2) -> None:
        init_class = type(dataset)
        init_class(root=dataset.root, split=dataset.split)

    def test_not_yet_extracted(self, dataset: FLAIR2, tmp_path: Path) -> None:
        if type(dataset) == FLAIR2Toy:
            shutil.copyfile(
                os.path.join('tests', 'data', 'flair2', "FLAIR2", f"flair_2_toy_dataset.zip"),
                os.path.join(str(tmp_path), "flair_2_toy_dataset.zip")
            )
        else:
            filenames = list(dataset.dir_names[dataset.split].values())
            filenames.append(dataset.centroids_file)
            dir = os.path.join('tests', 'data', 'flair2', "FLAIR2")
            for filename in filenames:
                shutil.copyfile(
                    os.path.join(dir, f"{filename}.zip"), os.path.join(str(tmp_path), f"{filename}.zip")
                )
            
        init_class = type(dataset)
        init_class(root=str(tmp_path), split=dataset.split)

    def test_invalid_split(self, dataset: FLAIR2) -> None:
        with pytest.raises(AssertionError):
            init_class = type(dataset)
            init_class(split="foo")

    def test_not_downloaded(self, dataset: FLAIR2, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            init_class = type(dataset)
            init_class(str(tmp_path)+"tmp", download=False)

    @pytest.mark.parametrize("use_sentinel", [True, False])
    def test_plot(self, dataset: FLAIR2, use_sentinel: bool) -> None:
        dataset.use_sentinel = use_sentinel
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = torch.clone(sample['mask'])
        dataset.plot(sample, suptitle='Prediction')
        plt.close()