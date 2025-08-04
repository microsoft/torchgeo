# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import torch

from torchgeo.datasets import (
    BoundingBox,
    IntersectionDataset,
    RioXarrayDataset,
    UnionDataset,
)

pytest.importorskip('rioxarray')


class TestRioXarrayDataset:
    @pytest.fixture(scope='class')
    def dataset(self) -> RioXarrayDataset:
        root = os.path.join('tests', 'data', 'rioxr', 'data')
        return RioXarrayDataset(root=root, data_variables=['zos', 'tos'])

    def test_getitem(self, dataset: RioXarrayDataset) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_and(self, dataset: RioXarrayDataset) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: RioXarrayDataset) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_invalid_query(self, dataset: RioXarrayDataset) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]
