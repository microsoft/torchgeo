import os
import shutil
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets import BoundingBox, Chesapeake13, ZipDataset
from torchgeo.transforms import Identity


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestChesapeake13:
    @pytest.fixture
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> Chesapeake13:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5 = "8363639b51651cc1de2bdbeb2be4f9b1"
        monkeypatch.setattr(Chesapeake13, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join(
            "tests", "data", "chesapeake", "BAYWIDE", "Baywide_13Class_20132014.zip"
        )
        monkeypatch.setattr(Chesapeake13, "url", url)  # type: ignore[attr-defined]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            plt, "show", lambda *args: None
        )
        (tmp_path / "chesapeake" / "BAYWIDE").mkdir(parents=True)
        root = str(tmp_path)
        transforms = Identity()
        return Chesapeake13(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: Chesapeake13) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["masks"], torch.Tensor)

    def test_add(self, dataset: Chesapeake13) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ZipDataset)

    def test_already_downloaded(self, dataset: Chesapeake13) -> None:
        Chesapeake13(root=os.path.dirname(os.path.dirname(dataset.root)), download=True)

    def test_plot(self, dataset: Chesapeake13) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["masks"])

    def test_url(self) -> None:
        ds = Chesapeake13(os.path.join("tests", "data"))
        assert "cicwebresources.blob.core.windows.net" in ds.url

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            Chesapeake13(str(tmp_path))

    def test_invalid_query(self, dataset: Chesapeake13) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* is not within bounds of the index:"
        ):
            dataset[query]
