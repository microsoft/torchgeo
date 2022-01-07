import os
import shutil
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datasets.inria import InriaBuildings

from ..data.inria.data import generate_test_data


class TestInriaBuildings:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self,
        request: SubRequest,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
    ) -> InriaBuildings:

        root = str(tmp_path)
        test_md5 = generate_test_data(root, 2)
        monkeypatch.setattr(  # type: ignore[attr-defined]
            InriaBuildings, "md5", test_md5
        )
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

    def test_already_downloaded(self, dataset: InriaBuildings) -> None:
        InriaBuildings(root=dataset.root)

    def test_not_downloaded(self, tmp_path: str) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            InriaBuildings(str(tmp_path))

    def test_dataset_checksum(self, dataset: InriaBuildings) -> None:
        InriaBuildings.md5 = "randommd5hash123"
        shutil.rmtree(os.path.join(dataset.root, dataset.foldername))
        with pytest.raises(RuntimeError, match="Dataset corrupted"):
            InriaBuildings(root=dataset.root, checksum=True)

    def test_plot(self, dataset: InriaBuildings) -> None:
        x = dataset[0].copy()
        if dataset.split == "train":
            x["prediction"] = x["mask"]
        dataset.plot(x, suptitle="Test")
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
