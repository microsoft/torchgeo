import distutils.sysconfig
import os
import subprocess
import sys
from pathlib import Path
from typing import Generator

from _pytest.monkeypatch import MonkeyPatch


def test_install(
    monkeypatch: Generator[MonkeyPatch, None, None], tmp_path: Path
) -> None:
    site_packages_dir = distutils.sysconfig.get_python_lib(prefix=str(tmp_path))
    os.makedirs(site_packages_dir)
    monkeypatch.setenv(  # type: ignore[attr-defined]
        "PYTHONPATH", site_packages_dir, prepend=os.pathsep
    )
    subprocess.run(
        [sys.executable, "setup.py", "install", "--prefix", str(tmp_path)], check=True
    )
