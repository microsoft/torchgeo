# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from typing import Any

import pytest
import torchvision.datasets.utils
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets.utils import Executable, Path, which


def copy(url: str, root: Path, *args: Any, **kwargs: Any) -> None:
    os.makedirs(root, exist_ok=True)
    shutil.copy(url, root)


@pytest.fixture(autouse=True)
def download_url(monkeypatch: MonkeyPatch, request: SubRequest) -> None:
    monkeypatch.setattr(torchvision.datasets.utils, 'download_url', copy)
    monkeypatch.setattr(torchgeo.datasets.utils, 'download_url', copy)
    _, filename = os.path.split(request.path)
    module = filename[5:-3]
    try:
        monkeypatch.setattr(f'torchgeo.datasets.{module}.download_url', copy)
    except AttributeError:
        pass
    monkeypatch.setattr('torchgeo.datasets.copernicus.lcz_s2.download_url', copy)


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
