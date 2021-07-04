import os
import shutil
import sys
from pathlib import Path
from typing import Generator

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

import torchgeo.datasets.nwpu
from torchgeo.datasets import VHR10
from torchgeo.transforms import Identity

pytest.importorskip("rarfile")
pytest.importorskip("pycocotools")


def download_file_from_google_drive(file_id: str, root: str, *args: str) -> None:
    shutil.copy(file_id, root)


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


# TODO: figure out how to install unrar on Windows in GitHub Actions
@pytest.mark.skipif(sys.platform == "win32", reason="requires unrar executable")
class TestVHR10:
    @pytest.fixture(params=["positive", "negative"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> VHR10:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.nwpu,
            "download_file_from_google_drive",
            download_file_from_google_drive,
        )
        monkeypatch.setattr(  # type: ignore[attr-defined]
            torchgeo.datasets.nwpu, "download_url", download_url
        )
        file_id = os.path.join("tests", "data", "vhr10", "NWPU VHR-10 dataset.rar")
        monkeypatch.setitem(  # type: ignore[attr-defined]
            VHR10.image_meta, "file_id", file_id
        )
        md5 = "e5c38351bd948479fe35a71136aedbc4"
        monkeypatch.setitem(VHR10.image_meta, "md5", md5)  # type: ignore[attr-defined]
        url = os.path.join("tests", "data", "vhr10", "annotations.json")
        monkeypatch.setitem(VHR10.target_meta, "url", url)  # type: ignore[attr-defined]
        md5 = "16fc6aa597a19179dad84151cc221873"
        monkeypatch.setitem(VHR10.target_meta, "md5", md5)  # type: ignore[attr-defined]
        (tmp_path / "vhr10").mkdir()
        root = str(tmp_path)
        split = request.param
        transforms = Identity()
        return VHR10(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: VHR10) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], dict)

    def test_len(self, dataset: VHR10) -> None:
        if dataset.split == "positive":
            assert len(dataset) == 650
        elif dataset.split == "negative":
            assert len(dataset) == 150

    def test_add(self, dataset: VHR10) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        if dataset.split == "positive":
            assert len(ds) == 1300
        elif dataset.split == "negative":
            assert len(ds) == 300

    def test_already_downloaded(self, dataset: VHR10) -> None:
        VHR10(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            VHR10(split="train")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            VHR10(str(tmp_path))
