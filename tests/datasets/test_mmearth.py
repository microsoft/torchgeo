# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
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
    @pytest.fixture(params=['MMEarth', 'MMEarth64', 'MMEarth100k'])
    def dataset(self, tmp_path: Path, request: SubRequest) -> MMEarth:
        root = tmp_path
        subset = request.param
        shutil.copytree(data_dir_dict[subset], root / Path(data_dir_dict[subset]).name)
        transforms = nn.Identity()
        return MMEarth(root, subset=subset, transforms=transforms)

    def test_getitem(self, dataset: MMEarth) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        for modality in dataset.modalities:
            modality_name = dataset.modality_category_name.get(modality, '') + modality
            assert modality_name in x
            assert isinstance(x[modality_name], torch.Tensor)
            assert x[modality_name].shape[0] == len(dataset.modality_bands[modality])

    def test_subset_modalities(self, dataset: MMEarth) -> None:
        specified_modalities = ['sentinel2', 'dynamic_world']
        dataset = MMEarth(
            dataset.root, subset=dataset.subset, modalities=specified_modalities
        )
        x = dataset[0]
        assert isinstance(x, dict)

        for modality in dataset.modalities:
            modality_name = dataset.modality_category_name.get(modality, '') + modality
            if modality in specified_modalities:
                assert modality_name in x
            else:
                assert modality_name not in x

    def test_dataset_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            MMEarth(tmp_path)

    def test_invalid_modalities(self, dataset: MMEarth) -> None:
        with pytest.raises(ValueError, match='is an invalid modality'):
            MMEarth(dataset.root, subset=dataset.subset, modalities=['invalid'])

    def test_invalid_modality_bands_modality_name(self, dataset: MMEarth) -> None:
        with pytest.raises(ValueError, match='is an invalid modality name'):
            MMEarth(
                dataset.root,
                subset=dataset.subset,
                modality_bands={'invalid': ['invalid']},
            )

    def test_invalid_modality_bands(self, dataset: MMEarth) -> None:
        with pytest.raises(ValueError, match='is an invalid band name for modality'):
            MMEarth(
                dataset.root,
                subset=dataset.subset,
                modality_bands={'sentinel2': ['invalid']},
            )

    @pytest.mark.parametrize(
        'modality_bands, modalities',
        [
            ({'sentinel2': ['B2', 'B3']}, ['sentinel2']),
            (
                {'sentinel1_asc': ['VV'], 'sentinel1_desc': ['VH']},
                ['sentinel1_asc', 'sentinel1_desc'],
            ),
        ],
    )
    def test_subset_modaliy_bands(
        self,
        dataset: MMEarth,
        modality_bands: dict[str, list[str]],
        modalities: list[str],
    ) -> None:
        dataset = MMEarth(
            dataset.root,
            subset=dataset.subset,
            modalities=modalities,
            modality_bands=modality_bands,
        )
        x = dataset[0]
        assert isinstance(x, dict)

        for modality in dataset.modalities:
            modality_name = dataset.modality_category_name.get(modality, '') + modality
            if modality in modality_bands:
                assert modality_name in x
                assert x[modality_name].shape[0] == len(modality_bands[modality])
            else:
                assert modality_name not in x

    def test_sentinel1_asc_desc(self, dataset: MMEarth) -> None:
        modality_bands = {'sentinel1_asc': ['VV'], 'sentinel1_desc': ['VH']}
        dataset = MMEarth(
            dataset.root,
            subset=dataset.subset,
            modalities=['sentinel1_asc', 'sentinel1_desc'],
            modality_bands=modality_bands,
        )
        x = dataset[0]
        assert isinstance(x, dict)

        for modality in dataset.modalities:
            modality_name = dataset.modality_category_name.get(modality, '') + modality
            if modality in modality_bands:
                assert modality_name in x
                assert x[modality_name].shape[0] == len(modality_bands[modality])
            else:
                assert modality_name not in x

    @pytest.mark.parametrize('normalization_mode', ['z-score', 'min-max'])
    def test_normalization_mode(
        self, dataset: MMEarth, normalization_mode: str
    ) -> None:
        dataset = MMEarth(
            dataset.root, subset=dataset.subset, normalization_mode=normalization_mode
        )
        x = dataset[0]
        assert isinstance(x, dict)

    def test_len(self, dataset: MMEarth) -> None:
        assert len(dataset) >= 2
