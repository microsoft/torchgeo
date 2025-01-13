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

from torchgeo.datasets import (
    FLAIR2,
    DatasetNotFoundError,
    FLAIR2Toy,
    RGBBandsMissingError,
)


class TestFLAIR2:
    @pytest.fixture(
        params=[
            (split, init_class)
            for split in ['train', 'test']
            for init_class in [FLAIR2, FLAIR2Toy]
        ]
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> FLAIR2:
        md5s = {
            'flair_2_labels_test.zip': '00718f3c0dc071dc9eaa7f9cfa977707',
            'flair_2_toy_dataset.zip': '45b9b9c85b5b52833d4c844765c54750',
            'flair_2_sen_test.zip': '56aa00085ccaa9350ae294adad406933',
            'flair_2_aerial_test.zip': 'e82210d08b9d636c083c54107de4fc68',
            'flair_sen_train.zip': '99a76c3fa598930f7339a530fe29010b',
            'flair_aerial_train.zip': 'bde5f1da7b51da76a87d61ede9c560c5',
            'flair_labels_train.zip': 'b221c7cd69cc3e160c4ad3e7b00ed2d9',
            'flair_2_centroids_sp_to_patch.zip': '35318aa3b984f02a9049b7eab5f1fecb',
        }

        monkeypatch.setattr(FLAIR2, 'md5s', md5s)
        url_prefix = os.path.join('tests', 'data', 'flair2', 'FLAIR2')
        monkeypatch.setattr(FLAIR2, 'url_prefix', url_prefix)

        root: Path = tmp_path
        split: str = request.param[0]
        init_class = request.param[1]
        bands: tuple[str, ...] = ('B01', 'B02', 'B03', 'B04', 'B05')
        transforms = nn.Identity()

        flair_class: FLAIR2 = init_class(
            root,
            split,
            bands,
            transforms,
            download=True,
            checksum=True,
            use_sentinel=True,
        )

        return flair_class

    def test_get_num_bands(self, dataset: FLAIR2) -> None:
        assert dataset.get_num_bands() == len(dataset.aerial_all_bands)

    def test_per_band_statistics(self, dataset: FLAIR2) -> None:
        if dataset.split != 'train':
            return

        mins, maxs, means, stdvs = dataset.per_band_statistics(dataset.split)
        for stats in [mins, maxs, means, stdvs]:
            assert isinstance(stats, list)
            assert stats.__len__() == dataset.get_num_bands()
            assert all(isinstance(stat, float) for stat in stats)

    @pytest.mark.parametrize('use_sentinel', [True, False])
    def test_getitem(self, dataset: FLAIR2, use_sentinel: bool) -> None:
        dataset.use_sentinel = use_sentinel
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape == (len(dataset.aerial_all_bands), 32, 32)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].shape[-2:] == x['mask'].shape[-2:]

    def test_len(self, dataset: FLAIR2) -> None:
        assert len(dataset) == 4


        # Test the 1:n mapping of images to sentinels (i.e. 1 sentinel to 2 images)
        if not isinstance(dataset, FLAIR2Toy):
            assert dataset.files[0]['sentinel'] == dataset.files[1]['sentinel']

    def test_already_downloaded(self, dataset: FLAIR2) -> None:
        init_class = type(dataset)
        init_class(root=dataset.root, split=dataset.split)

    def test_not_yet_extracted(self, dataset: FLAIR2, tmp_path: Path) -> None:
        if isinstance(dataset, FLAIR2Toy):
            shutil.copyfile(
                os.path.join(
                    'tests', 'data', 'flair2', 'FLAIR2', 'flair_2_toy_dataset.zip'
                ),
                os.path.join(str(tmp_path), 'flair_2_toy_dataset.zip'),
            )
        else:
            filenames = list(dataset.dir_names[dataset.split].values())
            filenames.append(dataset.centroids_file)
            dir = os.path.join('tests', 'data', 'flair2', 'FLAIR2')
            for filename in filenames:
                filename = filename.replace('-', '_')
                shutil.copyfile(
                    os.path.join(dir, f'{filename}.zip'),
                    os.path.join(str(tmp_path), f'{filename}.zip'),
                )

        init_class = type(dataset)
        init_class(root=str(tmp_path), split=dataset.split)

    def test_invalid_split(self, dataset: FLAIR2) -> None:
        with pytest.raises(AssertionError):
            init_class = type(dataset)
            init_class(split='foo')

    def test_not_downloaded(self, dataset: FLAIR2, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            init_class = type(dataset)
            init_class(str(tmp_path) + 'tmp', download=False)

    @pytest.mark.parametrize('use_sentinel', [True, False])
    def test_plot(self, dataset: FLAIR2, use_sentinel: bool) -> None:
        dataset.use_sentinel = use_sentinel
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = torch.clone(sample['mask'])
        dataset.plot(sample, suptitle='Prediction')
        plt.close()

    def test_plot_rgb_missing(self, dataset: FLAIR2) -> None:
        dataset.aerial_bands = ('B02', 'B03', 'B04', 'B05')
        with pytest.raises(RGBBandsMissingError):
            dataset.plot(dataset[0], suptitle='Test')
