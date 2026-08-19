# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from collections.abc import Generator
from typing import Any

import pytest
from _pytest.fixtures import SubRequest
from _pytest.tmpdir import TempPathFactory
from pytest import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets.utils import Executable, Path, which


def copy(url: str, root: Path, *args: Any, **kwargs: Any) -> None:
    os.makedirs(root, exist_ok=True)
    shutil.copy(url, root)


@pytest.fixture(autouse=True)
def download_url(monkeypatch: MonkeyPatch, request: SubRequest) -> None:
    monkeypatch.setattr(torchgeo.datasets.utils, 'download_url', copy)
    _, filename = os.path.split(request.path)
    module = filename[5:-3]
    try:
        monkeypatch.setattr(f'torchgeo.datasets.{module}.download_url', copy)
    except AttributeError:
        pass
    monkeypatch.setattr('torchgeo.datasets.copernicus.embed.download_url', copy)
    monkeypatch.setattr('torchgeo.datasets.copernicus.lcz_s2.download_url', copy)


@pytest.fixture(scope='module')
def temp_archive(
    request: SubRequest, tmp_path_factory: TempPathFactory
) -> Generator[tuple[str, str], None, None]:
    dir_not_zipped = request.param
    dir_zipped = shutil.make_archive(
        tmp_path_factory.mktemp('archive') / dir_not_zipped,
        'zip',
        root_dir=dir_not_zipped,
    )
    yield dir_not_zipped, dir_zipped
    os.remove(dir_zipped)


@pytest.fixture
def aws(monkeypatch: MonkeyPatch) -> Executable:
    path = os.path.dirname(os.path.realpath(__file__))
    monkeypatch.setenv('PATH', path, prepend=os.pathsep)
    return which('aws')


@pytest.fixture
def azcopy(monkeypatch: MonkeyPatch) -> Executable:
    path = os.path.dirname(os.path.realpath(__file__))
    monkeypatch.setenv('PATH', path, prepend=os.pathsep)
    return which('azcopy')
