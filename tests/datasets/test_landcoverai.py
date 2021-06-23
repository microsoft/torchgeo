import os
from pathlib import Path
import shutil
from typing import Generator

from _pytest.fixtures import SubRequest
import pytest
from pytest import MonkeyPatch
import torch
from torch.utils.data import ConcatDataset
import torchvision.datasets.utils

from torchgeo.datasets import LandCoverAI
from torchgeo.transforms import Identity


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestLandCoverAI:
    @pytest.fixture(params=["train", "val", "test"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> LandCoverAI:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchvision.datasets.utils, "download_url", download_url
        )
        md5 = "8a84857267619c8cf22730193b3b1ada"
        monkeypatch.setattr(LandCoverAI, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "landcoverai", "landcover.ai.v1.zip")
        monkeypatch.setattr(LandCoverAI, "url", url)  # type: ignore[attr-defined]
        sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        monkeypatch.setattr(LandCoverAI, "sha256", sha256)  # type: ignore[attr-defined]
        (tmp_path / "landcoverai").mkdir()
        root = str(tmp_path)
        split = request.param
        transforms = Identity()
        return LandCoverAI(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: LandCoverAI) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: LandCoverAI) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: LandCoverAI) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_downloaded(self, dataset: LandCoverAI) -> None:
        LandCoverAI(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            LandCoverAI(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            LandCoverAI(str(tmp_path))
