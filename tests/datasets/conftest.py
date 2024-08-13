# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from pytest import MonkeyPatch

from torchgeo.datasets.utils import Executable, which


@pytest.fixture
def azcopy(monkeypatch: MonkeyPatch) -> Executable:
    path = os.path.dirname(os.path.realpath(__file__))
    monkeypatch.setenv('PATH', path, prepend=os.pathsep)
    monkeypatch.setenv('PATHEXT', '.py', prepend=os.pathsep)
    return which('azcopy')
