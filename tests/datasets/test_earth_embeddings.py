# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import pytest
from torch import Tensor, nn

from torchgeo.datasets import DatasetNotFoundError, EarthEmbeddings


class TestEarthEmbeddings:
    pytest.importorskip('pyarrow')

    @pytest.fixture
    def dataset(self, test_data: Callable[[str], str]) -> EarthEmbeddings:
        root = os.path.join(test_data('earth_embeddings'), 'dinov2')
        transforms = nn.Identity()
        return EarthEmbeddings(root, transforms=transforms)

    def test_getitem(self, dataset: EarthEmbeddings) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['embedding'], Tensor)

    def test_len(self, dataset: EarthEmbeddings) -> None:
        assert len(dataset) == 4

    def test_no_data(self, test_data: Callable[[str], str]) -> None:
        root = test_data('earth_embeddings')
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EarthEmbeddings(root)

    def test_plot(self, dataset: EarthEmbeddings) -> None:
        x = dataset[0]
        dataset.plot(x)
        plt.close()
