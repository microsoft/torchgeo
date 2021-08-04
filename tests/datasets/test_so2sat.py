import os
from pathlib import Path
from typing import Generator

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import So2Sat
from torchgeo.transforms import Identity


class TestSo2Sat:
    @pytest.fixture(params=["train", "validation", "test"])
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], request: SubRequest
    ) -> So2Sat:
        md5s = {
            "train": "086c5fa964a401d4194d09ab161c39f1",
            "validation": "dd864f1af0cd495af99d7de80103f49e",
            "test": "320102c5c15f3cee7691f203824028ce",
        }

        monkeypatch.setattr(So2Sat, "md5s", md5s)  # type: ignore[attr-defined]
        root = os.path.join("tests", "data", "so2sat")
        split = request.param
        transforms = Identity()
        return So2Sat(root, split, transforms, checksum=True)

    def test_getitem(self, dataset: So2Sat) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], int)

    def test_len(self, dataset: So2Sat) -> None:
        assert len(dataset) == 10

    def test_out_of_bounds(self, dataset: So2Sat) -> None:
        # h5py at version 2.10.0 raises a ValueError instead of an IndexError so we
        # check for both here
        with pytest.raises((IndexError, ValueError)):
            dataset[10]

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            So2Sat(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            So2Sat(str(tmp_path))
