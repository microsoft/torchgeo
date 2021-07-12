import os
from pathlib import Path
from typing import Generator

import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import SEN12MS
from torchgeo.transforms import Identity


class TestSEN12MS:
    @pytest.fixture(params=["train", "test"])
    def dataset(
        self, monkeypatch: Generator[MonkeyPatch, None, None], request: SubRequest
    ) -> SEN12MS:
        md5s = [
            "3079d1c5038fa101ec2072657f2cb1ab",
            "f11487a4b2e641b64ed80a031c4d121d",
            "299691b948b37028398d4506d0195c6d",
            "76e6847b10ee9323ce022508721e2c6c",
            "dfbe57486455c31ae6f4d243186a8da5",
            "8d0aae3b12d420cab2feff5035400cbf",
            "f524074dcd90b9a770031cbfec50db71",
            "5256cf09bd2a0ec44bdff78f28e6653d",
            "b85b1641971444c87fedbc7134c437ac",
            "af28777ee277e3f9577c10a3c6d952eb",
            "44d18ee9efeb83f921b3b7aa6d511bbf",
            "00e18016c6af1e55528c535d9b06c35a",
            "02d5128ac1fc2bf8762091b4f319762d",
            "02d5128ac1fc2bf8762091b4f319762d",
        ]

        monkeypatch.setattr(SEN12MS, "md5s", md5s)  # type: ignore[attr-defined]
        root = os.path.join("tests", "data")
        split = request.param
        transforms = Identity()
        return SEN12MS(root, split, transforms, checksum=True)

    def test_getitem(self, dataset: SEN12MS) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_len(self, dataset: SEN12MS) -> None:
        assert len(dataset) == 8

    def test_add(self, dataset: SEN12MS) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 16

    def test_out_of_bounds(self, dataset: SEN12MS) -> None:
        with pytest.raises(IndexError):
            dataset[8]

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SEN12MS(split="foo")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            SEN12MS(str(tmp_path))
