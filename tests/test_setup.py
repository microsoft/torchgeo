import distutils.sysconfig
import os
from pathlib import Path
import subprocess
import sys


def test_install(tmp_path: Path) -> None:
    site_packages_dir = distutils.sysconfig.get_python_lib(prefix=str(tmp_path))
    os.environ["PYTHONPATH"] = site_packages_dir
    subprocess.run(
        [sys.executable, "setup.py", "install", "--prefix", str(tmp_path)], check=True
    )
