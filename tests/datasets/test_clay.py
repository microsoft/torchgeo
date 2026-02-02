# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch.nn as nn
from _pytest.fixtures import SubRequest
from torch import Tensor

from torchgeo.datasets import ClayEmbeddings, DatasetNotFoundError

pytest.importorskip('pyarrow')


class TestClayEmbeddings:
    @pytest.fixture(params=['v0', 'v1.5'])
    def dataset(self, request: SubRequest) -> ClayEmbeddings:
        root = os.path.join('tests', 'data', 'clay', request.param)
        transforms = nn.Identity()
        return ClayEmbeddings(root, transforms=transforms)

    def test_getitem(self, dataset: ClayEmbeddings) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['embedding'], Tensor)

    def test_len(self, dataset: ClayEmbeddings) -> None:
        assert len(dataset) == 4

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            ClayEmbeddings(tmp_path)

    def test_plot(self, dataset: ClayEmbeddings) -> None:
        x = dataset[0]
        dataset.plot(x)
        plt.close()
