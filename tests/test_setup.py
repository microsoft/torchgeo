import distutils.sysconfig
from pathlib import Path
import subprocess
import sys
from typing import Generator

from pytest import MonkeyPatch


def test_install(
    monkeypatch: Generator[MonkeyPatch, None, None], tmp_path: Path
) -> None:
    site_packages_dir = distutils.sysconfig.get_python_lib(prefix=str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", site_packages_dir, prepend=True)
    subprocess.run(
        [sys.executable, "setup.py", "install", "--prefix", str(tmp_path)], check=True
    )
