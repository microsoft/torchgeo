# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch.nn as nn
from torch import Tensor

from torchgeo.datasets import DatasetNotFoundError, EarthEmbeddings


class TestEarthEmbeddings:
    pytest.importorskip('pyarrow')

    @pytest.fixture
    def dataset(self) -> EarthEmbeddings:
        root = os.path.join(
            'tests', 'data', 'earth_embeddings', 'embeddings', 'embeddings'
        )
        transforms = nn.Identity()
        return EarthEmbeddings(root, transforms=transforms)

    def test_getitem(self, dataset: EarthEmbeddings) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['embedding'], Tensor)

    def test_len(self, dataset: EarthEmbeddings) -> None:
        assert len(dataset) == 4

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EarthEmbeddings(tmp_path)

    def test_plot(self, dataset: EarthEmbeddings) -> None:
        x = dataset[0]
        dataset.plot(x)
        plt.close()
