import subprocess
import sys
from pathlib import Path


def test_install(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "setup.py", "build", "--build-base", str(tmp_path)], check=True
    )
