# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib
import pytest
import torch
import torchvision
from pytest import MonkeyPatch


def load(
    *args: Any, progress: bool = False, check_hash: bool = False, **kwargs: Any
) -> Any:
    return torch.load(*args, **kwargs)


@pytest.fixture
def load_state_dict_from_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torchvision.models._api, 'load_state_dict_from_url', load)


@pytest.fixture(autouse=True, scope='session')
def matplotlib_backend() -> None:
    matplotlib.use('agg')


@pytest.fixture(autouse=True)
def torch_hub(tmp_path: Path) -> None:
    torch.hub.set_dir(tmp_path)


@pytest.fixture(scope='session')
def test_data(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], str]:
    """Generate each requested fake dataset once in this pytest worker's temporary tree."""
    source = Path(__file__).parent / 'data'
    root = tmp_path_factory.mktemp('test-data')

    @cache
    def generate(name: str) -> str:
        destination = root / name
        shutil.copytree(source / name, destination)
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(
            [str(source.parent.parent), env.get('PYTHONPATH', '')]
        )
        subprocess.run(
            [sys.executable, str(destination / 'data.py')],
            cwd=destination,
            env=env,
            check=True,
        )
        return str(destination)

    def path(name: str) -> str:
        relative = Path(name)
        directory = next(
            parent
            for parent in [relative, *relative.parents]
            if (source / parent / 'data.py').is_file()
        )
        return str(Path(generate(str(directory))) / relative.relative_to(directory))

    return path
