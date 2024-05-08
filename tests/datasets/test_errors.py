# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any

import pytest
from torch.utils.data import Dataset

from torchgeo.datasets import DatasetNotFoundError, RGBBandsMissingError


class TestDatasetNotFoundError:
    def test_none(self) -> None:
        ds: Dataset[Any] = Dataset()
        match = 'Dataset not found.'
        with pytest.raises(DatasetNotFoundError, match=match):
            raise DatasetNotFoundError(ds)

    def test_root(self) -> None:
        ds: Dataset[Any] = Dataset()
        ds.root = 'foo'  # type: ignore[attr-defined]
        match = "Dataset not found in `root='foo'` and cannot be automatically "
        match += 'downloaded, either specify a different `root` or manually '
        match += 'download the dataset.'
        with pytest.raises(DatasetNotFoundError, match=match):
            raise DatasetNotFoundError(ds)

    def test_paths(self) -> None:
        ds: Dataset[Any] = Dataset()
        ds.paths = 'foo'  # type: ignore[attr-defined]
        match = "Dataset not found in `paths='foo'` and cannot be automatically "
        match += 'downloaded, either specify a different `paths` or manually '
        match += 'download the dataset.'
        with pytest.raises(DatasetNotFoundError, match=match):
            raise DatasetNotFoundError(ds)

    def test_root_download(self) -> None:
        ds: Dataset[Any] = Dataset()
        ds.root = 'foo'  # type: ignore[attr-defined]
        ds.download = False  # type: ignore[attr-defined]
        match = "Dataset not found in `root='foo'` and `download=False`, either "
        match += 'specify a different `root` or use `download=True` to automatically '
        match += 'download the dataset.'
        with pytest.raises(DatasetNotFoundError, match=match):
            raise DatasetNotFoundError(ds)

    def test_paths_download(self) -> None:
        ds: Dataset[Any] = Dataset()
        ds.paths = 'foo'  # type: ignore[attr-defined]
        ds.download = False  # type: ignore[attr-defined]
        match = "Dataset not found in `paths='foo'` and `download=False`, either "
        match += 'specify a different `paths` or use `download=True` to automatically '
        match += 'download the dataset.'
        with pytest.raises(DatasetNotFoundError, match=match):
            raise DatasetNotFoundError(ds)


def test_rgb_bands_missing() -> None:
    match = 'Dataset does not contain some of the RGB bands'
    with pytest.raises(RGBBandsMissingError, match=match):
        raise RGBBandsMissingError()
