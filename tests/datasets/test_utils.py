# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import glob
import math
import os
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Tuple

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets.utils import (
    BoundingBox,
    collate_dict,
    disambiguate_timestamp,
    download_and_extract_archive,
    download_radiant_mlhub,
    download_radiant_mlhub_collection,
    extract_archive,
    working_dir,
)


@pytest.fixture
def mock_missing_module(monkeypatch: Generator[MonkeyPatch, None, None]) -> None:
    import_orig = builtins.__import__

    def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ["rarfile", "radiant_mlhub"]:
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(  # type: ignore[attr-defined]
        builtins, "__import__", mocked_import
    )


class Dataset:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(
            "tests", "data", "ref_african_crops_kenya_02", "*.tar.gz"
        )
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


class Collection:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join(
            "tests", "data", "ref_african_crops_kenya_02", "*.tar.gz"
        )
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch_dataset(dataset_id: str, **kwargs: str) -> Dataset:
    return Dataset()


def fetch_collection(collection_id: str, **kwargs: str) -> Collection:
    return Collection()


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


def test_mock_missing_module(mock_missing_module: None) -> None:
    import sys  # noqa: F401


# TODO: figure out how to install unrar on Windows in GitHub Actions
@pytest.mark.skipif(sys.platform == "win32", reason="requires unrar executable")
@pytest.mark.parametrize(
    "src",
    [
        os.path.join("cowc_detection", "COWC_Detection_Columbus_CSUAV_AFRL.tbz"),
        os.path.join("cowc_detection", "COWC_test_list_detection.txt.bz2"),
        os.path.join("vhr10", "NWPU VHR-10 dataset.rar"),
        os.path.join("landcoverai", "landcover.ai.v1.zip"),
        os.path.join("sen12ms", "ROIs1158_spring_lc.tar.gz"),
    ],
)
def test_extract_archive(src: str, tmp_path: Path) -> None:
    pytest.importorskip("rarfile")
    extract_archive(os.path.join("tests", "data", src), str(tmp_path))


def test_missing_rarfile(mock_missing_module: None) -> None:
    with pytest.raises(
        ImportError,
        match="rarfile is not installed and is required to extract this dataset",
    ):
        extract_archive(
            os.path.join("tests", "data", "vhr10", "NWPU VHR-10 dataset.rar")
        )


def test_unsupported_scheme() -> None:
    with pytest.raises(
        RuntimeError, match="src file has unknown archival/compression scheme"
    ):
        extract_archive("foo.bar")


def test_download_and_extract_archive(
    tmp_path: Path, monkeypatch: Generator[MonkeyPatch, None, None]
) -> None:
    monkeypatch.setattr(  # type: ignore[attr-defined]
        torchgeo.datasets.utils, "download_url", download_url
    )
    download_and_extract_archive(
        os.path.join("tests", "data", "landcoverai", "landcover.ai.v1.zip"),
        str(tmp_path),
    )


def test_download_radiant_mlhub(
    tmp_path: Path, monkeypatch: Generator[MonkeyPatch, None, None]
) -> None:
    radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
    monkeypatch.setattr(  # type: ignore[attr-defined]
        radiant_mlhub.Dataset, "fetch", fetch_dataset
    )
    download_radiant_mlhub("", str(tmp_path))


def test_download_radiant_mlhub_collection(
    tmp_path: Path, monkeypatch: Generator[MonkeyPatch, None, None]
) -> None:
    radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
    monkeypatch.setattr(  # type: ignore[attr-defined]
        radiant_mlhub.Collection, "fetch", fetch_collection
    )
    download_radiant_mlhub_collection("", str(tmp_path))


def test_missing_radiant_mlhub(mock_missing_module: None) -> None:
    with pytest.raises(
        ImportError,
        match="radiant_mlhub is not installed and is required to download this dataset",
    ):
        download_radiant_mlhub("", "")


class TestBoundingBox:
    def test_new_init(self) -> None:
        bbox = BoundingBox(0, 1, 2, 3, 4, 5)

        assert bbox.minx == 0
        assert bbox.maxx == 1
        assert bbox.miny == 2
        assert bbox.maxy == 3
        assert bbox.mint == 4
        assert bbox.maxt == 5

        assert bbox[0] == 0
        assert bbox[-1] == 5
        assert bbox[1:3] == (1, 2)

    def test_repr_str(self) -> None:
        bbox = BoundingBox(0, 1, 2.0, 3.0, -5, -4)
        expected = "BoundingBox(minx=0, maxx=1, miny=2.0, maxy=3.0, mint=-5, maxt=-4)"
        assert repr(bbox) == expected
        assert str(bbox) == expected

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            # Same box
            ((0, 1, 0, 1, 0, 1), True),
            ((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), True),
            # bbox1 strictly within bbox2
            ((-1, 2, -1, 2, -1, 2), True),
            # bbox2 strictly within bbox1
            ((0.25, 0.75, 0.25, 0.75, 0.25, 0.75), True),
            # One corner of bbox1 within bbox2
            ((0.5, 1.5, 0.5, 1.5, 0.5, 1.5), True),
            ((0.5, 1.5, -0.5, 0.5, 0.5, 1.5), True),
            ((0.5, 1.5, 0.5, 1.5, -0.5, 0.5), True),
            ((0.5, 1.5, -0.5, 0.5, -0.5, 0.5), True),
            ((-0.5, 0.5, 0.5, 1.5, 0.5, 1.5), True),
            ((-0.5, 0.5, -0.5, 0.5, 0.5, 1.5), True),
            ((-0.5, 0.5, 0.5, 1.5, -0.5, 0.5), True),
            ((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), True),
            # No overlap
            ((0.5, 1.5, 0.5, 1.5, 2, 3), False),
            ((0.5, 1.5, 2, 3, 0.5, 1.5), False),
            ((2, 3, 0.5, 1.5, 0.5, 1.5), False),
            ((2, 3, 2, 3, 2, 3), False),
        ],
    )
    def test_intersects(
        self,
        test_input: Tuple[float, float, float, float, float, float],
        expected: bool,
    ) -> None:
        bbox1 = BoundingBox(0, 1, 0, 1, 0, 1)
        bbox2 = BoundingBox(*test_input)
        assert bbox1.intersects(bbox2) == bbox2.intersects(bbox1) == expected

    def test_picklable(self) -> None:
        bbox = BoundingBox(0, 1, 2, 3, 4, 5)
        x = pickle.dumps(bbox)
        y = pickle.loads(x)
        assert bbox == y

    def test_invalid_x(self) -> None:
        with pytest.raises(
            ValueError, match="Bounding box is invalid: 'minx=1' > 'maxx=0'"
        ):
            BoundingBox(1, 0, 2, 3, 4, 5)

    def test_invalid_y(self) -> None:
        with pytest.raises(
            ValueError, match="Bounding box is invalid: 'miny=3' > 'maxy=2'"
        ):
            BoundingBox(0, 1, 3, 2, 4, 5)

    def test_invalid_t(self) -> None:
        with pytest.raises(
            ValueError, match="Bounding box is invalid: 'mint=5' > 'maxt=4'"
        ):
            BoundingBox(0, 1, 2, 3, 5, 4)


@pytest.mark.parametrize(
    "date_string,format,min_datetime,max_datetime",
    [
        ("", "", 0, sys.maxsize),
        (
            "2021",
            "%Y",
            datetime(2021, 1, 1, 0, 0, 0, 0).timestamp(),
            datetime(2021, 12, 31, 23, 59, 59, 999999).timestamp(),
        ),
        (
            "2021-09",
            "%Y-%m",
            datetime(2021, 9, 1, 0, 0, 0, 0).timestamp(),
            datetime(2021, 9, 30, 23, 59, 59, 999999).timestamp(),
        ),
        (
            "2021-09-13",
            "%Y-%m-%d",
            datetime(2021, 9, 13, 0, 0, 0, 0).timestamp(),
            datetime(2021, 9, 13, 23, 59, 59, 999999).timestamp(),
        ),
        (
            "2021-09-13 17",
            "%Y-%m-%d %H",
            datetime(2021, 9, 13, 17, 0, 0, 0).timestamp(),
            datetime(2021, 9, 13, 17, 59, 59, 999999).timestamp(),
        ),
        (
            "2021-09-13 17:21",
            "%Y-%m-%d %H:%M",
            datetime(2021, 9, 13, 17, 21, 0, 0).timestamp(),
            datetime(2021, 9, 13, 17, 21, 59, 999999).timestamp(),
        ),
        (
            "2021-09-13 17:21:53",
            "%Y-%m-%d %H:%M:%S",
            datetime(2021, 9, 13, 17, 21, 53, 0).timestamp(),
            datetime(2021, 9, 13, 17, 21, 53, 999999).timestamp(),
        ),
        (
            "2021-09-13 17:21:53:000123",
            "%Y-%m-%d %H:%M:%S:%f",
            datetime(2021, 9, 13, 17, 21, 53, 123).timestamp(),
            datetime(2021, 9, 13, 17, 21, 53, 123).timestamp(),
        ),
    ],
)
def test_disambiguate_timestamp(
    date_string: str, format: str, min_datetime: float, max_datetime: float
) -> None:
    mint, maxt = disambiguate_timestamp(date_string, format)
    assert math.isclose(mint, min_datetime)
    assert math.isclose(maxt, max_datetime)


def test_collate_dict() -> None:
    samples = [
        {
            "foo": torch.tensor(1),  # type: ignore[attr-defined]
            "bar": torch.tensor(2),  # type: ignore[attr-defined]
            "crs": CRS.from_epsg(3005),
        },
        {
            "foo": torch.tensor(3),  # type: ignore[attr-defined]
            "bar": torch.tensor(4),  # type: ignore[attr-defined]
            "crs": CRS.from_epsg(3005),
        },
    ]
    sample = collate_dict(samples)
    assert torch.allclose(  # type: ignore[attr-defined]
        sample["foo"], torch.tensor([1, 3])  # type: ignore[attr-defined]
    )
    assert torch.allclose(  # type: ignore[attr-defined]
        sample["bar"], torch.tensor([2, 4])  # type: ignore[attr-defined]
    )


def test_existing_directory(tmp_path: Path) -> None:
    subdir = tmp_path / "foo" / "bar"
    subdir.mkdir(parents=True)

    assert subdir.exists()

    with working_dir(str(subdir)):
        assert subdir.cwd() == subdir


def test_nonexisting_directory(tmp_path: Path) -> None:
    subdir = tmp_path / "foo" / "bar"

    assert not subdir.exists()

    with working_dir(str(subdir), create=True):
        assert subdir.cwd() == subdir
