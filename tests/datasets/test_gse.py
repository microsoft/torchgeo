# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest

from torchgeo.datasets import (
    DatasetNotFoundError,
    GoogleSatelliteEmbedding,
    IntersectionDataset,
    UnionDataset,
)


class TestGoogleSatelliteEmbedding:
    @pytest.fixture(
        params=[
            os.path.join('2024', '10N'),
            os.path.join('2024', 'U', '1', 'L', '7'),
            'x086q72fv2f9q1x4a-0000000000-0000000000.tiff',
        ]
    )
    def dataset(self, request: SubRequest) -> GoogleSatelliteEmbedding:
        paths = os.path.join('tests', 'data', 'gse', request.param)
        transforms = nn.Identity()
        return GoogleSatelliteEmbedding(paths, transforms=transforms)

    def test_len(self, dataset: GoogleSatelliteEmbedding) -> None:
        assert len(dataset) == 1

    def test_getitem(self, dataset: GoogleSatelliteEmbedding) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        # Rasterio may return nodata for upside down rasters
        assert not torch.all(x['image'] == -128.0)

    def test_and(self, dataset: GoogleSatelliteEmbedding) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: GoogleSatelliteEmbedding) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: GoogleSatelliteEmbedding) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            GoogleSatelliteEmbedding(tmp_path)

    def test_invalid_query(self, dataset: GoogleSatelliteEmbedding) -> None:
        with pytest.raises(
            IndexError, match=r'query: .* not found in index with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]
