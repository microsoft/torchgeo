# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import subprocess
import sys


def test_help() -> None:
    subprocess.run([sys.executable, '-m', 'torchgeo', '--help'], check=True)


def test_version() -> None:
    subprocess.run([sys.executable, '-m', 'torchgeo', '--version'], check=True)
