# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from itertools import product
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest

from torchgeo.datasets import DatasetNotFoundError, MMEarth

pytest.importorskip('h5py', minversion='3.6')

data_dir_dict = {
    'MMEarth': os.path.join('tests', 'data', 'mmearth', 'data_1M_v001'),
    'MMEarth64': os.path.join('tests', 'data', 'mmearth', 'data_1M_v001_64'),
    'MMEarth100k': os.path.join('tests', 'data', 'mmearth', 'data_100k_v001'),
}


class TestMMEarth:
    @pytest.fixture(
        params=list(
            product(['train', 'val', 'test'], ['MMEarth', 'MMEarth64', 'MMEarth100k'])
        )
    )
    def dataset(self, tmp_path: Path, request: SubRequest) -> MMEarth:
        root = tmp_path
        split, version = request.param
        shutil.copytree(
            data_dir_dict[version], root / Path(data_dir_dict[version]).name
        )
        transforms = nn.Identity()
        return MMEarth(root, split=split, ds_version=version, transforms=transforms)

    def test_getitem(self, dataset: MMEarth) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        for modality in dataset.modalities:
            assert modality in x
            assert isinstance(x[modality], torch.Tensor)

    def test_subset_modalities(self, dataset: MMEarth) -> None:
        specified_modalities = ['sentinel2', 'dynamic_world']
        dataset = MMEarth(
            dataset.root,
            split=dataset.split,
            ds_version=dataset.ds_version,
            modalities=specified_modalities,
        )
        x = dataset[0]
        assert isinstance(x, dict)

        for modality in dataset.modalities:
            if modality in specified_modalities:
                assert modality in x
            else:
                assert modality not in x

    def test_dataset_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            MMEarth(tmp_path)

    def test_invalid_modalities(self, dataset: MMEarth) -> None:
        with pytest.raises(ValueError, match='is an invalid modality name'):
            MMEarth(
                dataset.root,
                split=dataset.split,
                ds_version=dataset.ds_version,
                modalities=['invalid'],
            )

    def test_invalid_modality_bands_modality_name(self, dataset: MMEarth) -> None:
        with pytest.raises(ValueError, match='is an invalid modality name'):
            MMEarth(
                dataset.root,
                split=dataset.split,
                ds_version=dataset.ds_version,
                modality_bands={'invalid': ['invalid']},
            )

    def test_invalid_modality_bands(self, dataset: MMEarth) -> None:
        with pytest.raises(ValueError, match='is an invalid band name for modality'):
            MMEarth(
                dataset.root,
                split=dataset.split,
                ds_version=dataset.ds_version,
                modality_bands={'sentinel2': ['invalid']},
            )

    def test_subset_modaliy_bands(self, dataset: MMEarth) -> None:
        modality_bands = {'sentinel2': ['B2', 'B3']}
        dataset = MMEarth(
            dataset.root,
            split=dataset.split,
            ds_version=dataset.ds_version,
            modalities=['sentinel2'],
            modality_bands=modality_bands,
        )
        x = dataset[0]
        assert isinstance(x, dict)

        for modality in dataset.modalities:
            if modality in modality_bands:
                assert modality in x
                assert x[modality].shape[0] == len(modality_bands[modality])
            else:
                assert modality not in x

    @pytest.mark.parametrize('normalization_mode', ['z-score', 'min-max'])
    def test_normalization_mode(
        self, dataset: MMEarth, normalization_mode: str
    ) -> None:
        dataset = MMEarth(
            dataset.root,
            split=dataset.split,
            ds_version=dataset.ds_version,
            normalization_mode=normalization_mode,
        )
        x = dataset[0]
        assert isinstance(x, dict)

    def test_len(self, dataset: MMEarth) -> None:
        assert len(dataset) >= 2
