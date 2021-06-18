import pathlib
import subprocess


def test_install(tmp_path: pathlib.Path) -> None:
    subprocess.run(
        ["python3", "setup.py", "install", "--prefix", str(tmp_path)], check=True
    )
