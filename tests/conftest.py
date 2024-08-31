# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any

import pytest
import torch
import torchvision
from pytest import MonkeyPatch


def load(*args: Any, progress: bool = False, **kwargs: Any) -> Any:
    return torch.load(*args, **kwargs)


@pytest.fixture
def load_state_dict_from_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torchvision.models._api, 'load_state_dict_from_url', load)


@pytest.fixture(autouse=True)
def torch_hub(tmp_path: Path) -> None:
    torch.hub.set_dir(tmp_path)  # type: ignore[no-untyped-call]
