# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import sys


def test_help() -> None:
    subprocess.run([sys.executable, '-m', 'torchgeo', '--help'], check=True)
