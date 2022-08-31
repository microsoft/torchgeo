# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import glob
import math
import os
import pickle
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets.utils import (
    BoundingBox,
    concat_samples,
    disambiguate_timestamp,
    download_and_extract_archive,
    download_radiant_mlhub_collection,
    download_radiant_mlhub_dataset,
    extract_archive,
    merge_samples,
    percentile_normalization,
    stack_samples,
    unbind_samples,
    working_dir,
)


@pytest.fixture
def mock_missing_module(monkeypatch: MonkeyPatch) -> None:
    import_orig = builtins.__import__

    def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ["radiant_mlhub", "rarfile", "zipfile_deflate64"]:
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mocked_import)


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


@pytest.mark.parametrize(
    "src",
    [
        os.path.join("cowc_detection", "COWC_Detection_Columbus_CSUAV_AFRL.tbz"),
        os.path.join("cowc_detection", "COWC_test_list_detection.txt.bz2"),
        os.path.join("vhr10", "NWPU VHR-10 dataset.rar"),
        os.path.join("landcoverai", "landcover.ai.v1.zip"),
        os.path.join("chesapeake", "BAYWIDE", "Baywide_13Class_20132014.zip"),
        os.path.join("sen12ms", "ROIs1158_spring_lc.tar.gz"),
    ],
)
def test_extract_archive(src: str, tmp_path: Path) -> None:
    if src.endswith(".rar"):
        pytest.importorskip("rarfile", minversion="3")
    if src.startswith("chesapeake"):
        pytest.importorskip("zipfile_deflate64")
    extract_archive(os.path.join("tests", "data", src), str(tmp_path))


def test_missing_rarfile(mock_missing_module: None) -> None:
    with pytest.raises(
        ImportError,
        match="rarfile is not installed and is required to extract this dataset",
    ):
        extract_archive(
            os.path.join("tests", "data", "vhr10", "NWPU VHR-10 dataset.rar")
        )


def test_missing_zipfile_deflate64(mock_missing_module: None) -> None:
    # Should fallback on Python builtin zipfile
    extract_archive(os.path.join("tests", "data", "landcoverai", "landcover.ai.v1.zip"))


def test_unsupported_scheme() -> None:
    with pytest.raises(
        RuntimeError, match="src file has unknown archival/compression scheme"
    ):
        extract_archive("foo.bar")


def test_download_and_extract_archive(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
    download_and_extract_archive(
        os.path.join("tests", "data", "landcoverai", "landcover.ai.v1.zip"),
        str(tmp_path),
    )


def test_download_radiant_mlhub_dataset(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
    monkeypatch.setattr(radiant_mlhub.Dataset, "fetch", fetch_dataset)
    download_radiant_mlhub_dataset("", str(tmp_path))


def test_download_radiant_mlhub_collection(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    radiant_mlhub = pytest.importorskip("radiant_mlhub", minversion="0.2.1")
    monkeypatch.setattr(radiant_mlhub.Collection, "fetch", fetch_collection)
    download_radiant_mlhub_collection("", str(tmp_path))


def test_missing_radiant_mlhub(mock_missing_module: None) -> None:
    with pytest.raises(
        ImportError,
        match="radiant_mlhub is not installed and is required to download this dataset",
    ):
        download_radiant_mlhub_dataset("", "")

    with pytest.raises(
        ImportError,
        match="radiant_mlhub is not installed and is required to download this"
        + " collection",
    ):
        download_radiant_mlhub_collection("", "")


class TestBoundingBox:
    def test_repr_str(self) -> None:
        bbox = BoundingBox(0, 1, 2.0, 3.0, -5, -4)
        expected = "BoundingBox(minx=0, maxx=1, miny=2.0, maxy=3.0, mint=-5, maxt=-4)"
        assert repr(bbox) == expected
        assert str(bbox) == expected

    def test_getitem(self) -> None:
        bbox = BoundingBox(0, 1, 2, 3, 4, 5)

        assert bbox.minx == 0
        assert bbox.maxx == 1
        assert bbox.miny == 2
        assert bbox.maxy == 3
        assert bbox.mint == 4
        assert bbox.maxt == 5

        assert bbox[0] == 0
        assert bbox[-1] == 5
        assert bbox[1:3] == [1, 2]

    def test_iter(self) -> None:
        bbox = BoundingBox(0, 1, 2, 3, 4, 5)

        assert tuple(bbox) == (0, 1, 2, 3, 4, 5)

        i = 0
        for _ in bbox:
            i += 1
        assert i == 6

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            # Same box
            ((0, 1, 0, 1, 0, 1), True),
            ((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), True),
            # bbox1 strictly within bbox2
            ((-1, 2, -1, 2, -1, 2), True),
            # bbox2 strictly within bbox1
            ((0.25, 0.75, 0.25, 0.75, 0.25, 0.75), False),
            # One corner of bbox1 within bbox2
            ((0.5, 1.5, 0.5, 1.5, 0.5, 1.5), False),
            ((0.5, 1.5, -0.5, 0.5, 0.5, 1.5), False),
            ((0.5, 1.5, 0.5, 1.5, -0.5, 0.5), False),
            ((0.5, 1.5, -0.5, 0.5, -0.5, 0.5), False),
            ((-0.5, 0.5, 0.5, 1.5, 0.5, 1.5), False),
            ((-0.5, 0.5, -0.5, 0.5, 0.5, 1.5), False),
            ((-0.5, 0.5, 0.5, 1.5, -0.5, 0.5), False),
            ((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), False),
            # No overlap
            ((0.5, 1.5, 0.5, 1.5, 2, 3), False),
            ((0.5, 1.5, 2, 3, 0.5, 1.5), False),
            ((2, 3, 0.5, 1.5, 0.5, 1.5), False),
            ((2, 3, 2, 3, 2, 3), False),
        ],
    )
    def test_contains(
        self,
        test_input: Tuple[float, float, float, float, float, float],
        expected: bool,
    ) -> None:
        bbox1 = BoundingBox(0, 1, 0, 1, 0, 1)
        bbox2 = BoundingBox(*test_input)
        assert (bbox1 in bbox2) == expected

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            # Same box
            ((0, 1, 0, 1, 0, 1), (0, 1, 0, 1, 0, 1)),
            ((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (0, 1, 0, 1, 0, 1)),
            # bbox1 strictly within bbox2
            ((-1, 2, -1, 2, -1, 2), (-1, 2, -1, 2, -1, 2)),
            # bbox2 strictly within bbox1
            ((0.25, 0.75, 0.25, 0.75, 0.25, 0.75), (0, 1, 0, 1, 0, 1)),
            # One corner of bbox1 within bbox2
            ((0.5, 1.5, 0.5, 1.5, 0.5, 1.5), (0, 1.5, 0, 1.5, 0, 1.5)),
            ((0.5, 1.5, -0.5, 0.5, 0.5, 1.5), (0, 1.5, -0.5, 1, 0, 1.5)),
            ((0.5, 1.5, 0.5, 1.5, -0.5, 0.5), (0, 1.5, 0, 1.5, -0.5, 1)),
            ((0.5, 1.5, -0.5, 0.5, -0.5, 0.5), (0, 1.5, -0.5, 1, -0.5, 1)),
            ((-0.5, 0.5, 0.5, 1.5, 0.5, 1.5), (-0.5, 1, 0, 1.5, 0, 1.5)),
            ((-0.5, 0.5, -0.5, 0.5, 0.5, 1.5), (-0.5, 1, -0.5, 1, 0, 1.5)),
            ((-0.5, 0.5, 0.5, 1.5, -0.5, 0.5), (-0.5, 1, 0, 1.5, -0.5, 1)),
            ((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), (-0.5, 1, -0.5, 1, -0.5, 1)),
            # No overlap
            ((0.5, 1.5, 0.5, 1.5, 2, 3), (0, 1.5, 0, 1.5, 0, 3)),
            ((0.5, 1.5, 2, 3, 0.5, 1.5), (0, 1.5, 0, 3, 0, 1.5)),
            ((2, 3, 0.5, 1.5, 0.5, 1.5), (0, 3, 0, 1.5, 0, 1.5)),
            ((2, 3, 2, 3, 2, 3), (0, 3, 0, 3, 0, 3)),
        ],
    )
    def test_or(
        self,
        test_input: Tuple[float, float, float, float, float, float],
        expected: Tuple[float, float, float, float, float, float],
    ) -> None:
        bbox1 = BoundingBox(0, 1, 0, 1, 0, 1)
        bbox2 = BoundingBox(*test_input)
        bbox3 = BoundingBox(*expected)
        assert (bbox1 | bbox2) == bbox3

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            # Same box
            ((0, 1, 0, 1, 0, 1), (0, 1, 0, 1, 0, 1)),
            ((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), (0, 1, 0, 1, 0, 1)),
            # bbox1 strictly within bbox2
            ((-1, 2, -1, 2, -1, 2), (0, 1, 0, 1, 0, 1)),
            # bbox2 strictly within bbox1
            (
                (0.25, 0.75, 0.25, 0.75, 0.25, 0.75),
                (0.25, 0.75, 0.25, 0.75, 0.25, 0.75),
            ),
            # One corner of bbox1 within bbox2
            ((0.5, 1.5, 0.5, 1.5, 0.5, 1.5), (0.5, 1, 0.5, 1, 0.5, 1)),
            ((0.5, 1.5, -0.5, 0.5, 0.5, 1.5), (0.5, 1, 0, 0.5, 0.5, 1)),
            ((0.5, 1.5, 0.5, 1.5, -0.5, 0.5), (0.5, 1, 0.5, 1, 0, 0.5)),
            ((0.5, 1.5, -0.5, 0.5, -0.5, 0.5), (0.5, 1, 0, 0.5, 0, 0.5)),
            ((-0.5, 0.5, 0.5, 1.5, 0.5, 1.5), (0, 0.5, 0.5, 1, 0.5, 1)),
            ((-0.5, 0.5, -0.5, 0.5, 0.5, 1.5), (0, 0.5, 0, 0.5, 0.5, 1)),
            ((-0.5, 0.5, 0.5, 1.5, -0.5, 0.5), (0, 0.5, 0.5, 1, 0, 0.5)),
            ((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), (0, 0.5, 0, 0.5, 0, 0.5)),
        ],
    )
    def test_and_intersection(
        self,
        test_input: Tuple[float, float, float, float, float, float],
        expected: Tuple[float, float, float, float, float, float],
    ) -> None:
        bbox1 = BoundingBox(0, 1, 0, 1, 0, 1)
        bbox2 = BoundingBox(*test_input)
        bbox3 = BoundingBox(*expected)
        assert (bbox1 & bbox2) == bbox3

    @pytest.mark.parametrize(
        "test_input",
        [
            # No overlap
            (0.5, 1.5, 0.5, 1.5, 2, 3),
            (0.5, 1.5, 2, 3, 0.5, 1.5),
            (2, 3, 0.5, 1.5, 0.5, 1.5),
            (2, 3, 2, 3, 2, 3),
        ],
    )
    def test_and_no_intersection(
        self, test_input: Tuple[float, float, float, float, float, float]
    ) -> None:
        bbox1 = BoundingBox(0, 1, 0, 1, 0, 1)
        bbox2 = BoundingBox(*test_input)
        with pytest.raises(
            ValueError,
            match=re.escape(f"Bounding boxes {bbox1} and {bbox2} do not overlap"),
        ):
            bbox1 & bbox2

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            # Rectangular prism
            ((0, 1, 0, 1, 0, 1), 1),
            ((0, 2, 0, 3, 0, 4), 6),
            # Plane
            ((0, 0, 0, 1, 0, 1), 0),
            # Line
            ((0, 0, 0, 0, 0, 1), 0),
            # Point
            ((0, 0, 0, 0, 0, 0), 0),
        ],
    )
    def test_area(
        self, test_input: Tuple[float, float, float, float, float, float], expected: int
    ) -> None:
        bbox = BoundingBox(*test_input)
        assert bbox.area == expected

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            # Rectangular prism
            ((0, 1, 0, 1, 0, 1), 1),
            ((0, 2, 0, 3, 0, 4), 24),
            # Plane
            ((0, 0, 0, 1, 0, 1), 0),
            # Line
            ((0, 0, 0, 0, 0, 1), 0),
            # Point
            ((0, 0, 0, 0, 0, 0), 0),
        ],
    )
    def test_volume(
        self, test_input: Tuple[float, float, float, float, float, float], expected: int
    ) -> None:
        bbox = BoundingBox(*test_input)
        assert bbox.volume == expected

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
            "Dec 21",
            "%b %y",
            datetime(2021, 12, 1, 0, 0, 0, 0).timestamp(),
            datetime(2021, 12, 31, 23, 59, 59, 999999).timestamp(),
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


class TestCollateFunctionsMatchingKeys:
    @pytest.fixture(scope="class")
    def samples(self) -> List[Dict[str, Any]]:
        return [
            {"image": torch.tensor([1, 2, 0]), "crs": CRS.from_epsg(2000)},
            {"image": torch.tensor([0, 0, 3]), "crs": CRS.from_epsg(2001)},
        ]

    def test_stack_unbind_samples(self, samples: List[Dict[str, Any]]) -> None:
        sample = stack_samples(samples)
        assert sample["image"].size() == torch.Size([2, 3])
        assert torch.allclose(sample["image"], torch.tensor([[1, 2, 0], [0, 0, 3]]))
        assert sample["crs"] == [CRS.from_epsg(2000), CRS.from_epsg(2001)]

        new_samples = unbind_samples(sample)
        for i in range(2):
            assert torch.allclose(samples[i]["image"], new_samples[i]["image"])
            assert samples[i]["crs"] == new_samples[i]["crs"]

    def test_concat_samples(self, samples: List[Dict[str, Any]]) -> None:
        sample = concat_samples(samples)
        assert sample["image"].size() == torch.Size([6])
        assert torch.allclose(sample["image"], torch.tensor([1, 2, 0, 0, 0, 3]))
        assert sample["crs"] == CRS.from_epsg(2000)

    def test_merge_samples(self, samples: List[Dict[str, Any]]) -> None:
        sample = merge_samples(samples)
        assert sample["image"].size() == torch.Size([3])
        assert torch.allclose(sample["image"], torch.tensor([1, 2, 3]))
        assert sample["crs"] == CRS.from_epsg(2001)


class TestCollateFunctionsDifferingKeys:
    @pytest.fixture(scope="class")
    def samples(self) -> List[Dict[str, Any]]:
        return [
            {"image": torch.tensor([1, 2, 0]), "crs1": CRS.from_epsg(2000)},
            {"mask": torch.tensor([0, 0, 3]), "crs2": CRS.from_epsg(2001)},
        ]

    def test_stack_unbind_samples(self, samples: List[Dict[str, Any]]) -> None:
        sample = stack_samples(samples)
        assert sample["image"].size() == torch.Size([1, 3])
        assert sample["mask"].size() == torch.Size([1, 3])
        assert torch.allclose(sample["image"], torch.tensor([[1, 2, 0]]))
        assert torch.allclose(sample["mask"], torch.tensor([[0, 0, 3]]))
        assert sample["crs1"] == [CRS.from_epsg(2000)]
        assert sample["crs2"] == [CRS.from_epsg(2001)]

        new_samples = unbind_samples(sample)
        assert torch.allclose(samples[0]["image"], new_samples[0]["image"])
        assert samples[0]["crs1"] == new_samples[0]["crs1"]
        assert torch.allclose(samples[1]["mask"], new_samples[0]["mask"])
        assert samples[1]["crs2"] == new_samples[0]["crs2"]

    def test_concat_samples(self, samples: List[Dict[str, Any]]) -> None:
        sample = concat_samples(samples)
        assert sample["image"].size() == torch.Size([3])
        assert sample["mask"].size() == torch.Size([3])
        assert torch.allclose(sample["image"], torch.tensor([1, 2, 0]))
        assert torch.allclose(sample["mask"], torch.tensor([0, 0, 3]))
        assert sample["crs1"] == CRS.from_epsg(2000)
        assert sample["crs2"] == CRS.from_epsg(2001)

    def test_merge_samples(self, samples: List[Dict[str, Any]]) -> None:
        sample = merge_samples(samples)
        assert sample["image"].size() == torch.Size([3])
        assert sample["mask"].size() == torch.Size([3])
        assert torch.allclose(sample["image"], torch.tensor([1, 2, 0]))
        assert torch.allclose(sample["mask"], torch.tensor([0, 0, 3]))
        assert sample["crs1"] == CRS.from_epsg(2000)
        assert sample["crs2"] == CRS.from_epsg(2001)


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


def test_percentile_normalization() -> None:
    img: "np.typing.NDArray[np.int_]" = np.array([[1, 2], [98, 100]])

    img = percentile_normalization(img, 2, 98)
    assert img.min() == 0
    assert img.max() == 1
