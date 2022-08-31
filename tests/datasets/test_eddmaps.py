# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import os
from pathlib import Path
from typing import Any

import pytest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets import BoundingBox, EDDMapS, IntersectionDataset, UnionDataset

pytest.importorskip("pandas", minversion="0.23.2")


class TestEDDMapS:
    @pytest.fixture(scope="class")
    def dataset(self) -> EDDMapS:
        root = os.path.join("tests", "data", "eddmaps")
        return EDDMapS(root)

    def test_getitem(self, dataset: EDDMapS) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)

    def test_len(self, dataset: EDDMapS) -> None:
        assert len(dataset) == 2

    def test_and(self, dataset: EDDMapS) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: EDDMapS) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            EDDMapS(str(tmp_path))

    @pytest.fixture
    def mock_missing_module(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pandas":
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    def test_mock_missing_module(
        self, dataset: EDDMapS, mock_missing_module: None
    ) -> None:
        with pytest.raises(
            ImportError,
            match="pandas is not installed and is required to use this dataset",
        ):
            EDDMapS(dataset.root)

    def test_invalid_query(self, dataset: EDDMapS) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
