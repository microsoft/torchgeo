# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""GeoTIFF writer with Cloud-Optimized GeoTIFF support."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Self

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window


class GeoTIFFWriter:
    """GeoTIFF writer with chunked writing and COG support.

    .. versionadded:: 0.11

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

        self.dataset: Any = None

    def __enter__(self) -> Self:
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
        }

        self.dataset = rasterio.open(self.output_path, 'w', **kwargs)  # ty: ignore[no-matching-overload]
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
        window = Window(x_offset, y_offset, w, h)
        self.dataset.write(data, 1, window=window)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Close dataset on exit.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        if self.dataset:
            self.dataset.close()

    def finalize(self) -> None:
        """Build overview pyramids for Cloud-Optimized GeoTIFF.

        Builds internal overviews at the specified levels using the configured
        resampling method. Overviews enable efficient visualization at multiple
        zoom levels without loading the full resolution image.
        """
        if self.cog_config.get('overviews'):
            overview_levels = self.cog_config['overviews']
            resampling = self.cog_config.get('overview_resampling', 'nearest')

            from rasterio.enums import Resampling

            resampling_map = {
                'nearest': Resampling.nearest,
                'bilinear': Resampling.bilinear,
                'cubic': Resampling.cubic,
                'average': Resampling.average,
                'mode': Resampling.mode,
            }

            resampling_method = resampling_map.get(
                resampling.lower(), Resampling.nearest
            )

            with rasterio.open(self.output_path, 'r+') as dst:
                dst.build_overviews(overview_levels, resampling_method)
