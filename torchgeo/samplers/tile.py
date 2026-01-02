# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo tile samplers."""

from collections.abc import Iterator

import torch
from torch import Generator
from torch.utils.data import Sampler

from ..datasets import TileDataset

TileQuery = tuple[int, int, int, int]


class TileSampler(Sampler[TileQuery]):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.TileDataset`."""


class RandomTileSampler(TileSampler):
    """Samples random patches from tiles, weighted by tile area.

    Larger tiles are sampled more frequently in proportion to their pixel area.
    This ensures uniform spatial coverage across all tiles regardless of size.

    .. versionadded:: 0.9
    """

    def __init__(
        self,
        dataset: TileDataset,
        size: int,
        length: int,
        generator: Generator | None = None,
    ) -> None:
        """Initialize a new RandomTileSampler instance.

        Args:
            dataset: tile dataset to sample from
            size: patch size in pixels (square patches)
            length: number of samples to draw per epoch
            generator: optional random number generator for reproducibility
        """
        self.size = size
        self.length = length
        self.generator = generator

        self.tile_heights: list[int] = []
        self.tile_widths: list[int] = []
        tile_areas: list[int] = []

        for i in range(len(dataset)):
            height, width = dataset.get_tile_size(i)
            self.tile_heights.append(height)
            self.tile_widths.append(width)
            tile_areas.append(height * width)

        self.weights = torch.tensor(tile_areas, dtype=torch.float)
        self.weights /= self.weights.sum()

    def __iter__(self) -> Iterator[TileQuery]:
        """Yield random patch queries.

        Yields:
            (file_index, y, x, patch_size) tuples
        """
        for _ in range(self.length):
            i = int(torch.multinomial(self.weights, 1, generator=self.generator).item())
            max_y = self.tile_heights[i] - self.size
            max_x = self.tile_widths[i] - self.size

            if self.generator is not None:
                y = int(
                    torch.randint(
                        0, max(1, max_y + 1), (1,), generator=self.generator
                    ).item()
                )
                x = int(
                    torch.randint(
                        0, max(1, max_x + 1), (1,), generator=self.generator
                    ).item()
                )
            else:
                y = int(torch.randint(0, max(1, max_y + 1), (1,)).item())
                x = int(torch.randint(0, max(1, max_x + 1), (1,)).item())

            yield (i, y, x, self.size)

    def __len__(self) -> int:
        """Return the number of samples per epoch.

        Returns:
            length of the epoch
        """
        return self.length


class GridTileSampler(TileSampler):
    """Samples patches in a regular grid pattern from all tiles.

    Useful for inference when complete coverage of all tiles is needed.

    .. versionadded:: 0.9
    """

    def __init__(
        self, dataset: TileDataset, size: int, stride: int | None = None
    ) -> None:
        """Initialize a new GridTileSampler instance.

        Args:
            dataset: tile dataset to sample from
            size: patch size in pixels (square patches)
            stride: step size between patches (defaults to size for non-overlapping)
        """
        self.size = size
        self.stride = stride if stride is not None else size

        self.indices: list[TileQuery] = []
        for i in range(len(dataset)):
            height, width = dataset.get_tile_size(i)

            if height < size or width < size:
                continue

            y_positions = list(range(0, height - size, self.stride))
            if height - size not in y_positions:
                y_positions.append(height - size)

            x_positions = list(range(0, width - size, self.stride))
            if width - size not in x_positions:
                x_positions.append(width - size)

            for y in y_positions:
                for x in x_positions:
                    self.indices.append((i, y, x, size))

    def __iter__(self) -> Iterator[TileQuery]:
        """Yield grid patch queries.

        Yields:
            (file_index, y, x, patch_size) tuples
        """
        yield from self.indices

    def __len__(self) -> int:
        """Return the total number of patches across all tiles.

        Returns:
            number of patches that will be sampled
        """
        return len(self.indices)
