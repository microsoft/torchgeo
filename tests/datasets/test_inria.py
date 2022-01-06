import shutil

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest

from torchgeo.datasets.inria import InriaBuildings

from ..data.inria.data import generate_test_data

TEST_DATA_DIR = "tests/data/inria"

generate_test_data(TEST_DATA_DIR, 2)


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestInriaBuildings:
    @pytest.fixture(params=["train", "test"])
    def dataset(self, request: SubRequest) -> InriaBuildings:

        root = TEST_DATA_DIR
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return InriaBuildings(root, split=request.param, transforms=transforms)

    def test_getitem(self, dataset: InriaBuildings) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        if dataset.split == "train":
            assert isinstance(x["mask"], torch.Tensor)
        assert x["image"].shape[0] == 3

    def test_len(self, dataset: InriaBuildings) -> None:
        assert len(dataset) == 2

    def test_plot(self, dataset: InriaBuildings) -> None:
        x = dataset[0].copy()
        if dataset.split == "train":
            x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
