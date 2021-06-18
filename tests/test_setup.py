import pathlib
import subprocess

import pytest


def test_install(tmp_path: pathlib.Path) -> None:
    subprocess.run(["python3", "setup.py", "install", "--prefix", tmp_path], check=True)
