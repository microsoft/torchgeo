# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pyproj import CRS
from pytest import MonkeyPatch

from torchgeo.datasets import (
    DatasetNotFoundError,
    EuroCrops,
    IntersectionDataset,
    UnionDataset,
)


class TestEuroCrops:
    @pytest.fixture(params=[None, ['1000000010'], ['1000000000'], ['2000000000']])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> EuroCrops:
        classes = request.param
        monkeypatch.setattr(
            EuroCrops, 'zenodo_files', [('AA.zip', 'b2ef5cac231294731c1dfea47cba544d')]
        )
        monkeypatch.setattr(EuroCrops, 'hcat_md5', '22d61cf3b316c8babfd209ae81419d8f')
        base_url = os.path.join('tests', 'data', 'eurocrops') + os.sep
        monkeypatch.setattr(EuroCrops, 'base_url', base_url)
        monkeypatch.setattr(plt, 'show', lambda *args: None)
        root = tmp_path
        transforms = nn.Identity()
        return EuroCrops(
            root, classes=classes, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: EuroCrops) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: EuroCrops) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: EuroCrops) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: EuroCrops) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_downloaded(self, dataset: EuroCrops) -> None:
        EuroCrops(dataset.paths, download=True)

    def test_plot(self, dataset: EuroCrops) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle='Test')

    def test_plot_prediction(self, dataset: EuroCrops) -> None:
        query = dataset.bounds
        x = dataset[query]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EuroCrops(tmp_path)

    def test_invalid_query(self, dataset: EuroCrops) -> None:
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[200:200, 200:200, pd.Timestamp.min : pd.Timestamp.min]

    def test_get_label_with_none_hcat_code(self, dataset: EuroCrops) -> None:
        mock_feature = {'properties': {dataset.label_name: None}}
        label = dataset.get_label(mock_feature)
        assert label == 0, "Expected label to be 0 when 'EC_hcat_c' is None."

    def test_integrity_error(self, dataset: EuroCrops) -> None:
        dataset.zenodo_files = (('AA.zip', 'invalid'),)
        assert not dataset._check_integrity()
