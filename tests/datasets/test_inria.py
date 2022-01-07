from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets.inria import InriaBuildings

from ..data.inria.data import generate_test_data

TEST_DATA_DIR = "tests/data/inria"
md5_hash = generate_test_data(TEST_DATA_DIR, 2)


class TestInriaBuildings:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, request: SubRequest, monkeypatch: Generator[MonkeyPatch, None, None]
    ) -> InriaBuildings:

        root = TEST_DATA_DIR
        monkeypatch.setattr(  # type: ignore[attr-defined]
            InriaBuildings, "md5", md5_hash
        )
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return InriaBuildings(
            root, split=request.param, transforms=transforms, checksum=True
        )

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
