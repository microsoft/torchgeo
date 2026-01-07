# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TileDataset for sampling patches from large raster files."""

import os
from collections.abc import Callable, Iterable, Sequence
from typing import cast

import rasterio
import rasterio.windows
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .utils import Path


class TileDataset(Dataset[dict[str, Tensor]]):
    """Dataset for sampling patches from large raster files.

    TileDataset provides a way to work with large raster imagery that doesn't require
    geospatial reprojection. Unlike :class:`~torchgeo.datasets.GeoDataset`, this dataset
    indexes tiles by (file_index, y, x, patch_size) tuples, enabling efficient random
    sampling of patches from large images without the overhead of on-the-fly warping.

    This is particularly useful for:

    * Large satellite scenes that are too big to load entirely into memory
    * Benchmark datasets with images of varying sizes
    * Training workflows that require random patch sampling from tiles
    * Datasets where geospatial metadata is not needed or not available

    .. versionadded:: 0.9
    """

    def __init__(
        self,
        image_paths: Sequence[Path | Sequence[Path]],
        mask_paths: Sequence[Path] | None = None,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
    ) -> None:
        """Initialize a new TileDataset instance.

        Args:
            image_paths: sequence of paths to image files. Each element can be either:

                * a single path to a GeoTIFF (or other rasterio-readable format)
                * a sequence of paths that will be concatenated channel-wise

                This allows multi-layer datasets where each sample consists of
                multiple source files (e.g., different spectral bands or sensors).
            mask_paths: optional sequence of paths to mask files, must be the same
                length as image_paths if provided. Each mask should be a single-channel
                GeoTIFF matching the spatial resolution of the corresponding image(s).
            transforms: a function/transform that takes a sample dict and returns
                a transformed version

        Raises:
            ValueError: if mask_paths is provided but has different length than
                image_paths
        """
        # Normalize image_paths: convert each entry to a list of strings
        self.image_paths: list[list[str]] = []
        for entry in image_paths:
            if isinstance(entry, (str, os.PathLike)):
                self.image_paths.append([str(entry)])
            else:
                paths = cast(Iterable[Path], entry)
                self.image_paths.append([str(p) for p in paths])

        self.mask_paths = (
            [str(p) for p in mask_paths] if mask_paths is not None else None
        )
        self.transforms = transforms

        if self.mask_paths is not None and len(self.image_paths) != len(
            self.mask_paths
        ):
            msg = (
                f'image_paths and mask_paths must have the same length, '
                f'got {len(self.image_paths)} and {len(self.mask_paths)}'
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of tiles in the dataset.

        Returns:
            number of tiles (image files) in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, query: tuple[int, int, int, int]) -> dict[str, Tensor]:
        """Retrieve a patch from a tile.

        Args:
            query: tuple of (file_index, y, x, patch_size) where:
                - file_index: index of the tile in image_paths
                - y: row offset (top edge of patch in pixels)
                - x: column offset (left edge of patch in pixels)
                - patch_size: size of the square patch to extract

        Returns:
            sample containing 'image' key with tensor of shape (C, H, W), and
            optionally 'mask' key with tensor of shape (H, W) if mask_paths provided

        Raises:
            IndexError: if file_index is out of bounds
        """
        file_index, y, x, patch_size = query

        if not 0 <= file_index < len(self.image_paths):
            msg = f'file_index {file_index} out of bounds for dataset with {len(self.image_paths)} tiles'
            raise IndexError(msg)

        window = rasterio.windows.Window(x, y, patch_size, patch_size)

        image_tensors = []
        for image_path in self.image_paths[file_index]:
            with rasterio.open(image_path) as src:
                data = src.read(window=window)
            image_tensors.append(torch.from_numpy(data).float())
        image = torch.cat(image_tensors, dim=0)
        sample: dict[str, Tensor] = {'image': image}

        if self.mask_paths is not None:
            mask_path = self.mask_paths[file_index]
            with rasterio.open(mask_path) as src:
                mask = src.read(window=window)
            sample['mask'] = torch.from_numpy(mask).long()[0]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def get_tile_size(self, file_index: int) -> tuple[int, int]:
        """Get the dimensions of a tile.

        Args:
            file_index: index of the tile in image_paths

        Returns:
            tuple of (height, width) in pixels

        Raises:
            IndexError: if file_index is out of bounds
        """
        if not 0 <= file_index < len(self.image_paths):
            msg = f'file_index {file_index} out of bounds for dataset with {len(self.image_paths)} tiles'
            raise IndexError(msg)

        with rasterio.open(self.image_paths[file_index][0]) as src:
            return src.height, src.width
