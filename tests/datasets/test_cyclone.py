import glob
import os
from pathlib import Path
import shutil
from typing import Generator

from _pytest.fixtures import SubRequest
import pytest
from pytest import MonkeyPatch
import torch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import TropicalCycloneWindEstimation
from torchgeo.transforms import Identity


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        for tarball in glob.iglob(os.path.join("tests", "data", "cyclone", "*.tar.gz")):
            shutil.copy(tarball, output_dir)


def fetch(collection_id: str, **kwargs: str) -> Dataset:
    return Dataset()


class TestTropicalCycloneWindEstimation:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self,
        monkeypatch: Generator[MonkeyPatch, None, None],
        tmp_path: Path,
        request: SubRequest,
    ) -> TropicalCycloneWindEstimation:
        radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
        monkeypatch.setattr(  # type: ignore[attr-defined]
            radiant_mlhub.Dataset, "fetch", fetch
        )
        md5s = {
            "train": {
                "source": "d90be51a42ebafe234eb2b3a653ff03d",
                "labels": "2569bbe7d2b4702c3c9b6270f2bba900",
            },
            "test": {
                "source": "b205da6e23dad1ffe74f5821c38f3b10",
                "labels": "b90e0363f6a5f6b1138bc5d35711e6e9",
            },
        }
        monkeypatch.setattr(  # type: ignore[attr-defined]
            TropicalCycloneWindEstimation, "md5s", md5s
        )
        (tmp_path / "cyclone").mkdir()
        root = str(tmp_path)
        split = request.param
        transforms = Identity()
        return TropicalCycloneWindEstimation(
            root, split, transforms, download=True, api_key="", checksum=True
        )

    def test_getitem(self, dataset: TropicalCycloneWindEstimation) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["storm_id"], str)
        assert isinstance(x["relative_time"], str)
        assert isinstance(x["ocean"], str)
        assert isinstance(x["wind_speed"], str)

    def test_len(self, dataset: TropicalCycloneWindEstimation) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: TropicalCycloneWindEstimation) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 4

    def test_already_downloaded(self, dataset: TropicalCycloneWindEstimation) -> None:
        TropicalCycloneWindEstimation(root=dataset.root, download=True, api_key="")

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            TropicalCycloneWindEstimation(split="foo")

    def test_missing_api_key(self) -> None:
        match = "You must pass an MLHub API key if download=True."
        with pytest.raises(RuntimeError, match=match):
            TropicalCycloneWindEstimation(download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            TropicalCycloneWindEstimation(str(tmp_path))
