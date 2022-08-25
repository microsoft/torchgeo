# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import (
    BoundingBox,
    IntersectionDataset,
    OpenBuildings,
    UnionDataset,
)

pd = pytest.importorskip("pandas", minversion="0.23.2")


class TestOpenBuildings:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> OpenBuildings:

        root = str(tmp_path)
        shutil.copy(
            os.path.join("tests", "data", "openbuildings", "tiles.geojson"), root
        )
        shutil.copy(
            os.path.join("tests", "data", "openbuildings", "000_buildings.csv.gz"), root
        )

        md5s = {"000_buildings.csv.gz": "20aeeec9d45a0ce4d772a26e0bcbc25f"}

        monkeypatch.setattr(OpenBuildings, "md5s", md5s)
        transforms = nn.Identity()
        return OpenBuildings(root=root, transforms=transforms)

    @pytest.fixture(params=["pandas"])
    def mock_missing_module(self, monkeypatch: MonkeyPatch, request: SubRequest) -> str:
        import_orig = builtins.__import__
        package = str(request.param)

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == package:
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)
        return package

    def test_mock_missing_module(
        self, dataset: OpenBuildings, mock_missing_module: str
    ) -> None:
        package = mock_missing_module

        with pytest.raises(
            ImportError,
            match=f"{package} is not installed and is required to use this dataset",
        ):
            OpenBuildings(root=dataset.root)

    def test_no_shapes_to_rasterize(
        self, dataset: OpenBuildings, tmp_path: Path
    ) -> None:
        # empty csv buildings file
        path = os.path.join(tmp_path, "000_buildings.csv.gz")
        df = pd.read_csv(path)
        df = pd.DataFrame(columns=df.columns)
        df.to_csv(path, compression="gzip")
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_no_building_data_found(self, tmp_path: Path) -> None:
        false_root = os.path.join(tmp_path, "empty")
        os.makedirs(false_root)
        shutil.copy(
            os.path.join("tests", "data", "openbuildings", "tiles.geojson"), false_root
        )
        with pytest.raises(
            RuntimeError, match="have manually downloaded the dataset as suggested "
        ):
            OpenBuildings(root=false_root)

    def test_corrupted(self, dataset: OpenBuildings, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, "000_buildings.csv.gz"), "w") as f:
            f.write("bad")
        with pytest.raises(RuntimeError, match="Dataset found, but corrupted."):
            OpenBuildings(dataset.root, checksum=True)

    def test_no_meta_data_found(self, tmp_path: Path) -> None:
        false_root = os.path.join(tmp_path, "empty")
        os.makedirs(false_root)
        with pytest.raises(FileNotFoundError, match="Meta data file"):
            OpenBuildings(root=false_root)

    def test_nothing_in_index(self, dataset: OpenBuildings, tmp_path: Path) -> None:
        # change meta data to another 'title_url' so that there is no match found
        with open(os.path.join(tmp_path, "tiles.geojson")) as f:
            content = json.load(f)
            content["features"][0]["properties"]["tile_url"] = "mismatch.csv.gz"

        with open(os.path.join(tmp_path, "tiles.geojson"), "w") as f:
            json.dump(content, f)

        with pytest.raises(FileNotFoundError, match="data was found in"):
            OpenBuildings(dataset.root)

    def test_getitem(self, dataset: OpenBuildings) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: OpenBuildings) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: OpenBuildings) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_invalid_query(self, dataset: OpenBuildings) -> None:
        query = BoundingBox(100, 100, 100, 100, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]

    def test_plot(self, dataset: OpenBuildings) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle="test")
        plt.close()

    def test_plot_prediction(self, dataset: OpenBuildings) -> None:
        x = dataset[dataset.bounds]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()
