# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pyproj import CRS
from pytest import MonkeyPatch

from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    OpenBuildings,
    UnionDataset,
)


class TestOpenBuildings:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> OpenBuildings:
        root = tmp_path
        shutil.copy(
            os.path.join('tests', 'data', 'openbuildings', 'tiles.geojson'), root
        )
        shutil.copy(
            os.path.join('tests', 'data', 'openbuildings', '000_buildings.csv.gz'), root
        )

        md5s = {'000_buildings.csv.gz': '20aeeec9d45a0ce4d772a26e0bcbc25f'}

        monkeypatch.setattr(OpenBuildings, 'md5s', md5s)
        transforms = nn.Identity()
        return OpenBuildings(root, transforms=transforms)

    def test_no_shapes_to_rasterize(
        self, dataset: OpenBuildings, tmp_path: Path
    ) -> None:
        # empty csv buildings file
        path = os.path.join(tmp_path, '000_buildings.csv.gz')
        df = pd.read_csv(path)
        df = pd.DataFrame(columns=df.columns)
        df.to_csv(path, compression='gzip')
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_not_download(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            OpenBuildings(tmp_path)

    def test_corrupted(self, dataset: OpenBuildings, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, '000_buildings.csv.gz'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            OpenBuildings(dataset.paths, checksum=True)

    def test_nothing_in_index(self, dataset: OpenBuildings, tmp_path: Path) -> None:
        # change meta data to another 'title_url' so that there is no match found
        with open(os.path.join(tmp_path, 'tiles.geojson')) as f:
            content = json.load(f)
            content['features'][0]['properties']['tile_url'] = 'mismatch.csv.gz'

        with open(os.path.join(tmp_path, 'tiles.geojson'), 'w') as f:
            json.dump(content, f)

        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            OpenBuildings(dataset.paths)

    def test_getitem(self, dataset: OpenBuildings) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: OpenBuildings) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: OpenBuildings) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: OpenBuildings) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_invalid_query(self, dataset: OpenBuildings) -> None:
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[100:100, 100:100, pd.Timestamp.min : pd.Timestamp.min]

    def test_plot(self, dataset: OpenBuildings) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='test')
        plt.close()

    def test_plot_prediction(self, dataset: OpenBuildings) -> None:
        x = dataset[dataset.bounds]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_float_res(self, dataset: OpenBuildings) -> None:
        OpenBuildings(dataset.paths, res=0.0001)
