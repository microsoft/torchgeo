# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch.nn as nn
from torch import Tensor

from torchgeo.datasets import DatasetNotFoundError, EarthIndexEmbeddings

pytest.importorskip('pyarrow')


class TestEarthIndexEmbeddings:
    @pytest.fixture
    def dataset(self) -> EarthIndexEmbeddings:
        root = os.path.join('tests', 'data', 'earth_index', '2024')
        transforms = nn.Identity()
        return EarthIndexEmbeddings(root, transforms=transforms)

    def test_getitem(self, dataset: EarthIndexEmbeddings) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['embedding'], Tensor)

    def test_len(self, dataset: EarthIndexEmbeddings) -> None:
        assert len(dataset) == 4

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EarthIndexEmbeddings(tmp_path)

    def test_plot(self, dataset: EarthIndexEmbeddings) -> None:
        x = dataset[0]
        dataset.plot(x)
        plt.close()
