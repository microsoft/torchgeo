# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for GeoTIFF writer."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from torchgeo.callbacks.writer import GeoTIFFWriter


class TestGeoTIFFWriter:
    """Tests for GeoTIFFWriter."""

    def test_write_single_chunk(self, tmp_path: Path) -> None:
        """Test writing single chunk."""
        output = tmp_path / 'test.tif'
        transform = Affine(1, 0, 0, 0, -1, 100)

        writer = GeoTIFFWriter(
            output_path=output,
            width=64,
            height=64,
            num_bands=1,
            crs='EPSG:32631',
            transform=transform,
        )

        data = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)

        with writer:
            writer.write_chunk(data, 0, 0)

        with rasterio.open(output) as src:
            assert src.width == 64
            assert src.height == 64
            assert src.crs.to_string() == 'EPSG:32631'
            assert src.transform == transform
            np.testing.assert_array_equal(src.read(1), data)

    def test_write_multiple_chunks(self, tmp_path: Path) -> None:
        """Test writing multiple chunks."""
        output = tmp_path / 'test.tif'
        transform = Affine(1, 0, 0, 0, -1, 100)

        writer = GeoTIFFWriter(
            output_path=output,
            width=128,
            height=128,
            num_bands=1,
            crs='EPSG:32631',
            transform=transform,
        )

        with writer:
            writer.write_chunk(np.ones((64, 64), dtype=np.uint8), 0, 0)
            writer.write_chunk(np.ones((64, 64), dtype=np.uint8) * 2, 0, 64)
            writer.write_chunk(np.ones((64, 64), dtype=np.uint8) * 3, 64, 0)
            writer.write_chunk(np.ones((64, 64), dtype=np.uint8) * 4, 64, 64)

        with rasterio.open(output) as src:
            data = src.read(1)
            assert data[0, 0] == 1
            assert data[0, 127] == 2
            assert data[127, 0] == 3
            assert data[127, 127] == 4

    def test_write_without_context_raises(self, tmp_path: Path) -> None:
        """Test writing without context manager raises error."""
        output = tmp_path / 'test.tif'
        transform = Affine(1, 0, 0, 0, -1, 100)

        writer = GeoTIFFWriter(
            output_path=output,
            width=64,
            height=64,
            num_bands=1,
            crs='EPSG:32631',
            transform=transform,
        )

        data = np.ones((64, 64), dtype=np.uint8)
        with pytest.raises(RuntimeError, match='Writer not opened'):
            writer.write_chunk(data, 0, 0)

    def test_finalize_with_overviews(self, tmp_path: Path) -> None:
        """Test finalize creates overviews when configured."""
        output = tmp_path / 'test_cog.tif'
        transform = Affine(1, 0, 0, 0, -1, 256)

        writer = GeoTIFFWriter(
            output_path=output,
            width=256,
            height=256,
            num_bands=1,
            crs='EPSG:32631',
            transform=transform,
            cog_config={'overviews': [2, 4], 'overview_resampling': 'nearest'},
        )

        data = np.ones((256, 256), dtype=np.uint8)
        with writer:
            writer.write_chunk(data, 0, 0)

        writer.finalize()

        with rasterio.open(output) as src:
            assert src.overviews(1) == [2, 4]
