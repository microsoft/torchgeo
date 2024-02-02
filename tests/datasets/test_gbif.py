# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import pytest

from torchgeo.datasets import (
    GBIF,
    BoundingBox,
    DatasetNotFoundError,
    IntersectionDataset,
    UnionDataset,
)


class TestGBIF:
    @pytest.fixture(scope="class")
    def dataset(self) -> GBIF:
        root = os.path.join("tests", "data", "gbif")
        return GBIF(root)

    def test_getitem(self, dataset: GBIF) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)

    def test_len(self, dataset: GBIF) -> None:
        assert len(dataset) == 5

    def test_and(self, dataset: GBIF) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: GBIF) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            GBIF(str(tmp_path))

    def test_invalid_query(self, dataset: GBIF) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
