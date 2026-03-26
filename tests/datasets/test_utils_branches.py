# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Additional branch coverage tests for datasets utilities."""

import os
import tempfile

from torchgeo.datasets.utils import check_integrity


class TestCheckIntegrityBranches:
    """Test branch coverage for check_integrity function."""

    def test_file_does_not_exist(self) -> None:
        """Test check_integrity when file doesn't exist - first branch."""
        result = check_integrity('/nonexistent/path/to/file.txt')
        assert result is False

    def test_file_exists_no_checksum(self) -> None:
        """Test check_integrity when file exists but no checksum - last branch."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test content')
            temp_path = f.name
        try:
            result = check_integrity(temp_path)
            assert result is True
        finally:
            os.unlink(temp_path)

    def test_file_exists_with_md5_match(self) -> None:
        """Test check_integrity with md5 checksum that matches - middle branch."""
        # md5 of 'test content' is 9473fdd0d880a43c21b3718e392a69a8
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test content')
            temp_path = f.name
        try:
            # MD5 checksum of 'test content'
            result = check_integrity(temp_path, md5='9473fdd0d880a43c21b3718e392a69a8')
            assert result is True
        finally:
            os.unlink(temp_path)

    def test_file_exists_with_md5_mismatch(self) -> None:
        """Test check_integrity with md5 checksum that doesn't match."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('test content')
            temp_path = f.name
        try:
            # Wrong MD5
            result = check_integrity(temp_path, md5='wrongmd5checksum')
            assert result is False
        finally:
            os.unlink(temp_path)
