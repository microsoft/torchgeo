# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_required_args() -> None:
    args = [sys.executable, "train.py"]
    ps = subprocess.run(args, capture_output=True)
    assert ps.returncode != 0
    assert b"ConfigKeyError" in ps.stderr


def test_output_file(tmp_path: Path) -> None:
    output_file = tmp_path / "output"
    output_file.touch()
    args = [
        sys.executable,
        "train.py",
        "experiment.name=test",
        "program.output_dir=" + str(output_file),
        "experiment.task=test",
    ]
    ps = subprocess.run(args, capture_output=True)
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
        "experiment.name=test",
        "program.output_dir=" + str(output_dir),
        "experiment.task=test",
    ]
    ps = subprocess.run(args, capture_output=True)
    assert ps.returncode != 0
    assert b"FileExistsError" in ps.stderr


def test_overwrite_experiment_dir(tmp_path: Path) -> None:
    experiment_name = "test"
    output_dir = tmp_path / "output"
    data_dir = os.path.join("tests", "data", "cyclone")
    log_dir = tmp_path / "logs"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True)
    experiment_file = experiment_dir / "foo"
    experiment_file.touch()
    args = [
        sys.executable,
        "train.py",
        "experiment.name=test",
        "program.output_dir=" + str(output_dir),
        "program.data_dir=" + data_dir,
        "program.log_dir=" + str(log_dir),
        "experiment.task=cyclone",
        "experiment.datamodule.root_dir=" + data_dir,
        "program.overwrite=True",
        "trainer.fast_dev_run=1",
        "trainer.gpus=0",
    ]
    ps = subprocess.run(args, capture_output=True, check=True)
    assert re.search(
        b"The experiment directory, .*, already exists, we might overwrite data in it!",
        ps.stdout,
    )


def test_invalid_task(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    args = [
        sys.executable,
        "train.py",
        "experiment.name=foo",
        "program.output_dir=" + str(output_dir),
        "experiment.task=foo",
    ]
    ps = subprocess.run(args, capture_output=True)
    assert ps.returncode != 0
    assert b"ValueError" in ps.stderr


def test_missing_config_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    config_file = tmp_path / "config.yaml"
    args = [
        sys.executable,
        "train.py",
        "experiment.name=test",
        "program.output_dir=" + str(output_dir),
        "experiment.task=test",
        "config_file=" + str(config_file),
    ]
    ps = subprocess.run(args, capture_output=True)
    assert ps.returncode != 0
    assert b"FileNotFoundError" in ps.stderr


def test_config_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    data_dir = os.path.join("tests", "data", "cyclone")
    log_dir = tmp_path / "logs"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
program:
  output_dir: {output_dir}
  data_dir: {data_dir}
  log_dir: {log_dir}
experiment:
  name: test
  task: cyclone
  datamodule:
    root_dir: {data_dir}
trainer:
  fast_dev_run: true
  gpus: 0
"""
    )
    args = [sys.executable, "train.py", "config_file=" + str(config_file)]
    subprocess.run(args, check=True)
