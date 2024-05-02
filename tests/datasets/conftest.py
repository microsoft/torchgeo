# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from collections.abc import Iterator
from typing import BinaryIO

import pytest
from pytest import MonkeyPatch


class ContainerClient:
    def __init__(self, account_url: str, container_name: str) -> None:
        self.account_url = account_url
        self.container_name = container_name

    def list_blob_names(self, name_starts_with: str = "") -> Iterator[str]:
        prefix = os.path.join(self.account_url, self.container_name)
        for root, dirs, files in os.walk(prefix):
            for file in files:
                name = os.path.join(root, file).replace(prefix + os.sep, "", 1)
                if name.startswith(name_starts_with):
                    yield name

    def download_blob(self, blob: str) -> BinaryIO:
        path = os.path.join(self.account_url, self.container_name, blob)
        # TODO: filehandle leak
        f = open(path, "rb", buffering=0)
        return f


@pytest.fixture
def container_client(monkeypatch: MonkeyPatch) -> None:
    asb = pytest.importorskip("azure.storage.blob", minversion="12.4")
    monkeypatch.setattr(asb, "ContainerClient", ContainerClient)
