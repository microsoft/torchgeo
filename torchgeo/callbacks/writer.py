# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""GeoTIFF writer with Cloud-Optimized GeoTIFF support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.transform import Affine
from rasterio.windows import Window


class GeoTIFFWriter:
    """GeoTIFF writer with chunked writing and COG support.

    Example::

        writer = GeoTIFFWriter(
            output_path='output.tif',
            width=1024,
            height=1024,
            num_bands=1,
            crs='EPSG:32631',
            transform=Affine(...),
        )

        with writer:
            for chunk_y, chunk_x, chunk_data in chunks:
                writer.write_chunk(chunk_data, chunk_y, chunk_x)

        writer.finalize()

    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        num_bands: int,
        crs: Any,
        transform: Affine,
        dtype: str = 'uint8',
        cog_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize writer.

        Args:
            output_path: Path to save GeoTIFF.
            width: Image width in pixels.
            height: Image height in pixels.
            num_bands: Number of bands.
            crs: Coordinate reference system.
            transform: Affine transform.
            dtype: Output data type.
            cog_config: Optional COG configuration.
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.num_bands = num_bands
        self.crs = crs
        self.transform = transform
        self.dtype = dtype
        self.cog_config = cog_config or {}

        self.tmp_path = self.output_path.with_suffix('.tmp.tif')
        self.dataset: Any = None

    def __enter__(self) -> GeoTIFFWriter:
        """Open GeoTIFF for writing.

        Returns:
            GeoTIFFWriter instance.
        """
        kwargs = {
            'driver': 'GTiff',
            'height': self.height,
            'width': self.width,
            'count': self.num_bands,
            'dtype': self.dtype,
            'crs': self.crs,
            'transform': self.transform,
            'tiled': True,
            'compress': self.cog_config.get('compress', 'lzw'),
            'BIGTIFF': 'IF_SAFER',
        }

        self.dataset = rasterio.open(self.tmp_path, 'w', **kwargs)
        return self

    def write_chunk(
        self, data: np.typing.NDArray[np.uint8], y_offset: int, x_offset: int
    ) -> None:
        """Write a chunk to the output GeoTIFF.

        Args:
            data: Chunk data of shape (H, W) for single band.
            y_offset: Row offset in output.
            x_offset: Column offset in output.
        """
        if self.dataset is None:
            raise RuntimeError('Writer not opened. Use with statement.')

        h, w = data.shape
        window = Window(x_offset, y_offset, w, h)  # ty: ignore[too-many-positional-arguments]
        self.dataset.write(data, 1, window=window)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close dataset on exit.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        if self.dataset:
            self.dataset.close()

    def finalize(self) -> None:
        """Translate the written GTiff into a Cloud-Optimized GeoTIFF.

        Streams the temporary GTiff through the GDAL COG driver, which builds
        overviews so it validates as a COG, without loading the full resolution
        image into memory.
        """
        rasterio.shutil.copy(
            self.tmp_path,
            self.output_path,
            driver='COG',
            compress=self.cog_config.get('compress', 'lzw'),
            overview_resampling=self.cog_config.get('overview_resampling', 'nearest'),
            BIGTIFF='IF_SAFER',
        )
        self.tmp_path.unlink()
