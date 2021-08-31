# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_install(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "setup.py", "build", "--build-base", str(tmp_path)], check=True
    )
