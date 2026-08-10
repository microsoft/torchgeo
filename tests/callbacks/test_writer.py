# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for GeoTIFF writer."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

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

    def test_write_multi_band(self, tmp_path: Path) -> None:
        """Test writing multi-band chunks."""
        output = tmp_path / 'test.tif'
        transform = Affine(1, 0, 0, 0, -1, 100)

        writer = GeoTIFFWriter(
            output_path=output,
            width=64,
            height=64,
            num_bands=3,
            crs='EPSG:32631',
            transform=transform,
        )

        data = np.arange(3 * 64 * 64, dtype=np.uint8).reshape(3, 64, 64)

        with writer:
            writer.write_chunk(data, 0, 0)

        with rasterio.open(output) as src:
            assert src.count == 3
            np.testing.assert_array_equal(src.read(), data)

    def test_write_float(self, tmp_path: Path) -> None:
        """Test writing float chunks."""
        output = tmp_path / 'test.tif'
        transform = Affine(1, 0, 0, 0, -1, 100)

        writer = GeoTIFFWriter(
            output_path=output,
            width=64,
            height=64,
            num_bands=1,
            crs='EPSG:32631',
            transform=transform,
            dtype='float32',
        )

        data = np.random.rand(64, 64).astype(np.float32)

        with writer:
            writer.write_chunk(data, 0, 0)

        with rasterio.open(output) as src:
            assert src.dtypes[0] == 'float32'
            np.testing.assert_array_equal(src.read(1), data)

    def test_nodata(self, tmp_path: Path) -> None:
        """Test nodata value is written to the output."""
        output = tmp_path / 'test.tif'
        transform = Affine(1, 0, 0, 0, -1, 100)

        writer = GeoTIFFWriter(
            output_path=output,
            width=64,
            height=64,
            num_bands=1,
            crs='EPSG:32631',
            transform=transform,
            nodata=0,
        )

        with writer:
            writer.write_chunk(np.ones((64, 64), dtype=np.uint8), 0, 0)

        with rasterio.open(output) as src:
            assert src.nodata == 0

    def test_nodata_default_none(self, tmp_path: Path) -> None:
        """Test the default nodata of None does not break writing."""
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

        with writer:
            writer.write_chunk(np.ones((64, 64), dtype=np.uint8), 0, 0)

        with rasterio.open(output) as src:
            assert src.nodata is None

    def test_finalize_creates_cog(self, tmp_path: Path) -> None:
        """Test finalize produces a valid Cloud-Optimized GeoTIFF."""
        output = tmp_path / 'test_cog.tif'
        transform = Affine(1, 0, 0, 0, -1, 1024)

        writer = GeoTIFFWriter(
            output_path=output,
            width=1024,
            height=1024,
            num_bands=1,
            crs='EPSG:32631',
            transform=transform,
            overview_resampling='nearest',
        )

        data = np.ones((1024, 1024), dtype=np.uint8)
        with writer:
            writer.write_chunk(data, 0, 0)

        assert not writer.tmp_path.exists()
        with rasterio.open(output) as src:
            assert src.tags(ns='IMAGE_STRUCTURE')['LAYOUT'] == 'COG'
            assert src.overviews(1)

    def test_exception_cleans_up(self, tmp_path: Path) -> None:
        """Test an exception in the with block propagates and leaves no files."""
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

        with pytest.raises(ValueError, match='boom'), writer:
            writer.write_chunk(np.ones((64, 64), dtype=np.uint8), 0, 0)
            raise ValueError('boom')

        assert not writer.tmp_path.exists()
        assert not output.exists()
