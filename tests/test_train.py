import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


def test_required_args() -> None:
    args = [sys.executable, "train.py"]
    ps = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ps.returncode != 0
    assert b"Missing mandatory value" in ps.stderr


def test_output_file(tmp_path: Path) -> None:
    output_file = tmp_path / "output"
    output_file.touch()
    args = [
        sys.executable,
        "train.py",
        "program.experiment_name=test",
        f"program.output_dir={str(output_file)}",
        "task.name=test",
    ]
    ps = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ps.returncode != 0
    assert b"NotADirectoryError" in ps.stderr


def test_experiment_dir_not_empty(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    experiment_dir = output_dir / "test"
    experiment_dir.mkdir(parents=True)
    experiment_file = experiment_dir / "foo"
    experiment_file.touch()
    args = [
        sys.executable,
        "train.py",
        "program.experiment_name=test",
        f"program.output_dir={str(output_dir)}",
        "task.name=test",
    ]
    ps = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert ps.returncode != 0
    assert b"FileExistsError" in ps.stderr


def test_overwrite_experiment_dir(tmp_path: Path) -> None:
    experiment_name = "test"
    output_dir = tmp_path / "output"
    data_dir = os.path.join("tests", "data")
    log_dir = tmp_path / "logs"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True)
    experiment_file = experiment_dir / "foo"
    experiment_file.touch()
    args = [
        sys.executable,
        "train.py",
        "program.experiment_name=test",
        f"program.output_dir={str(output_dir)}",
        f"program.data_dir={data_dir}",
        f"program.log_dir={str(log_dir)}",
        "task.name=cyclone",
        "program.overwrite=True",
        "trainer.fast_dev_run=1",
    ]
    ps = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    assert re.search(
        b"The experiment directory, .*, already exists, we might overwrite data in it!",
        ps.stdout,
    )


@pytest.mark.parametrize("task", ["cyclone", "sen12ms"])
def test_tasks(task: str, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    data_dir = os.path.join("tests", "data")
    log_dir = tmp_path / "logs"
    args = [
        sys.executable,
        "train.py",
        "program.experiment_name=test",
        f"program.output_dir={str(output_dir)}",
        f"program.data_dir={data_dir}",
        f"program.log_dir={str(log_dir)}",
        "trainer.fast_dev_run=1",
        f"task.name={task}",
        "program.overwrite=True",
    ]
    subprocess.run(args, check=True)
