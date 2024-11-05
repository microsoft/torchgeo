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
            "flair-2_centroids_sp_to_patch": "af243a4c6ed95dd2b97d07261c7cc3dd",
            "flair_aerial_train": "70c558a7000e3671f3a3ed2ba187d795",
            "flair_sen_train": "661e20e39ffc7eca196b6cab8d1c9e27",
            "flair_labels_train": "525011487e0c282d22a91d3798768e38",
            "flair_2_aerial_test": "8b414c21e48e87331fed1d9835b17726",
            "flair_2_sen_test": "6247e940974e247be1c8187aaf35281c",
            "flair_2_labels_test": "9285086a7aa5085e3b8f2fc2f8618ad0",
        }

        monkeypatch.setattr(FLAIR2, "md5s", md5s)
        url_prefix = os.path.join("tests", "data", "flair2", "FLAIR2")
        monkeypatch.setattr(FLAIR2, "url_prefix", url_prefix)
        
        root = tmp_path
        split = request.param
        bands = ("B01", "B02", "B03", "B04", "B05")
        transforms = nn.Identity()
        
        return FLAIR2(root, split, bands, transforms, download=True, checksum=True, use_sentinel=True)
    
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

    @pytest.mark.parametrize("use_sentinel", [True, False])
    def test_plot(self, dataset: FLAIR2, use_sentinel: bool) -> None:
        dataset.use_sentinel = use_sentinel
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = torch.clone(sample['mask'])
        dataset.plot(sample, suptitle='Prediction')
        plt.close()

    # TODO: test toy option