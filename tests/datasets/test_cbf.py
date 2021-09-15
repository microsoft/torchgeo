# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
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
    ) -> CanadianBuildingFootprints:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.utils, "download_url", download_url
        )
        md5s = [
            "8a4a0a57367f67c69608d1452e30df13",
            "1829f4054a9a81bb23871ca797a3895c",
            "4358a0076fd43e9a2f436e74348813b0",
            "ae3726b1263727d72565ecacfed56fb8",
            "6861876d3a3ca7e79b28c61ab5906de4",
            "d289c9ea49801bb287ddbde1ea5f31ef",
            "3a940288297631b4e6a365266bfb949a",
            "6b43b3632b165ff79c1ca0c693a61398",
            "36283e0b29088ec281e77c989cbee100",
            "773da9d33e3766b7237a1d7db0811832",
            "cc833a65137c8a046c8f45bb695092b1",
            "067664d066c4152fb96a5c129cbabadf",
            "474bc084bc41b124aa4919e7a37a9648",
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
            root, res=0.1, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: CanadianBuildingFootprints) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_add(self, dataset: CanadianBuildingFootprints) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ZipDataset)

    def test_already_downloaded(self, dataset: CanadianBuildingFootprints) -> None:
        CanadianBuildingFootprints(root=dataset.root, download=True)

    def test_plot(self, dataset: CanadianBuildingFootprints) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["mask"])

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            CanadianBuildingFootprints(str(tmp_path))

    def test_invalid_query(self, dataset: CanadianBuildingFootprints) -> None:
        query = BoundingBox(2, 2, 2, 2, 2, 2)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
