# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""GeoTIFF writer with Cloud-Optimized GeoTIFF support."""

from __future__ import annotations

import contextlib
import pathlib
import types
from typing import Any, Self

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.crs import CRS
from rasterio.io import DatasetWriter
from rasterio.transform import Affine
from rasterio.windows import Window

from torchgeo.datasets.utils import Path


class GeoTIFFWriter(contextlib.AbstractContextManager['GeoTIFFWriter']):
    """GeoTIFF writer with chunked writing and COG support.

    Example::

        with GeoTIFFWriter(
            output_path='output.tif',
            width=1024,
            height=1024,
            num_bands=1,
            crs='EPSG:32631',
            transform=Affine(...),
        ) as writer:
            for chunk_y, chunk_x, chunk_data in chunks:
                writer.write_chunk(chunk_data, chunk_y, chunk_x)

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        num_bands: int,
        crs: CRS | str,
        transform: Affine,
        dtype: str = 'uint8',
        nodata: float | None = None,
        **kwargs: Any,
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
            nodata: Value to use for nodata pixels.
            **kwargs: Additional keyword arguments passed to the GDAL COG
                driver (e.g. ``compress``, ``overview_resampling``).
        """
        self.output_path = pathlib.Path(output_path)
        self.width = width
        self.height = height
        self.num_bands = num_bands
        self.crs = crs
        self.transform = transform
        self.dtype = dtype
        self.nodata = nodata
        self.kwargs = kwargs

        self.tmp_path = self.output_path.with_suffix('.tmp.tif')
        self.dataset: DatasetWriter | None = None

    def __enter__(self) -> Self:
        """Open GeoTIFF for writing.

        Returns:
            GeoTIFFWriter instance.
        """
        self.dataset = rasterio.open(
            self.tmp_path,
            'w',
            driver='GTiff',
            height=self.height,
            width=self.width,
            count=self.num_bands,
            dtype=self.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata,
            tiled=True,
            compress=self.kwargs.get('compress', 'lzw'),
            BIGTIFF='IF_SAFER',
        )
        return self

    def write_chunk(
        self, data: np.typing.NDArray[Any], y_offset: int, x_offset: int
    ) -> None:
        """Write a chunk to the output GeoTIFF.

        Args:
            data: Chunk data of shape (H, W) for single band or (C, H, W) for
                multi-band.
            y_offset: Row offset in output.
            x_offset: Column offset in output.
        """
        if data.ndim == 2:
            data = data[np.newaxis]
        _, h, w = data.shape
        window = Window(x_offset, y_offset, w, h)
        self.dataset.write(data, window=window)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Finalize the COG on exit and clean up the temporary file.

        Args:
            exc_type: Exception type.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        if self.dataset:
            self.dataset.close()
            self.dataset = None

        if exc_type is not None:
            self.tmp_path.unlink(missing_ok=True)
            return

        self._finalize()

    def _finalize(self) -> None:
        """Translate the written GTiff into a Cloud-Optimized GeoTIFF.

        Streams the temporary GTiff through the GDAL COG driver, which builds
        overviews so it validates as a COG, without loading the full resolution
        image into memory.
        """
        extra = {
            k: v
            for k, v in self.kwargs.items()
            if k not in ('compress', 'overview_resampling')
        }
        try:
            rasterio.shutil.copy(
                self.tmp_path,
                self.output_path,
                driver='COG',
                compress=self.kwargs.get('compress', 'lzw'),
                overview_resampling=self.kwargs.get('overview_resampling', 'nearest'),
                BIGTIFF='IF_SAFER',
                **extra,
            )
        except Exception:
            self.output_path.unlink(missing_ok=True)
            raise
        finally:
            self.tmp_path.unlink(missing_ok=True)
