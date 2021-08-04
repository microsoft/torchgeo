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
from torchgeo.datasets import BoundingBox, CanadianBuildingFootprints, ZipDataset
from torchgeo.transforms import Identity


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestCanadianBuildingFootprints:
    @pytest.fixture
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> CanadianBuildingFootprints:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5s = [
            "aef9a3deb3297f225d6cdb221cb48527",
            "2b7872c4121116fda8f96490daf89d29",
            "c71ded923e22569b62b00da2d2a61076",
            "75a8f652531790c3c3aefc0655400d6d",
            "89ff9c6257efa99365a8b709dde9579b",
            "d4d6a36ed834df5cbf5254effca78a4d",
            "cce85f6183427e3034704cf35919c985",
            "0149c7ec5101c0309c79b7e695dcb394",
            "b05216155725f48937804371b945f8ae",
            "72d0e6d7196345ca520c825697cc4947",
            "77e1c6c71ff0efbdd221b7e7d4a5f2df",
            "86e32374f068c7bbb76aa81af0736733",
            "5e453a3426b0bb986b2837b85e8b8850",
        ]
        monkeypatch.setattr(  # type: ignore[attr-defined]
            CanadianBuildingFootprints, "md5s", md5s
        )
        url = os.path.join("tests", "data", "cbf") + os.sep
        monkeypatch.setattr(  # type: ignore[attr-defined]
            CanadianBuildingFootprints, "url", url
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            plt, "show", lambda *args: None
        )
        root = str(tmp_path)
        transforms = Identity()
        return CanadianBuildingFootprints(
            root, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: CanadianBuildingFootprints) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["masks"], torch.Tensor)

    def test_add(self, dataset: CanadianBuildingFootprints) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ZipDataset)

    def test_already_downloaded(self, dataset: CanadianBuildingFootprints) -> None:
        CanadianBuildingFootprints(root=dataset.root, download=True)

    def test_plot(self, dataset: CanadianBuildingFootprints) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["masks"])

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            CanadianBuildingFootprints(str(tmp_path))

    def test_invalid_query(self, dataset: CanadianBuildingFootprints) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* is not within bounds of the index:"
        ):
            dataset[query]
