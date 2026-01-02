# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from torchgeo.datasets import TileDataset
from torchgeo.samplers import GridTileSampler, RandomTileSampler, TileSampler


class TestTileSampler:
    def test_abstract(self) -> None:
        sampler = TileSampler()
        assert isinstance(sampler, TileSampler)


class TestRandomTileSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> TileDataset:
        root = os.path.join('tests', 'data', 'tile')
        image_paths = sorted(Path(root, 'images').glob('*.tif'))
        return TileDataset(image_paths=image_paths)

    def test_iter(self, dataset: TileDataset) -> None:
        sampler = RandomTileSampler(dataset, size=16, length=20)
        count = 0
        for file_index, y, x, patch_size in sampler:
            assert 0 <= file_index < len(dataset)
            assert patch_size == 16
            height, width = dataset.get_tile_size(file_index)
            assert 0 <= y <= height - patch_size
            assert 0 <= x <= width - patch_size
            count += 1
        assert count == 20

    def test_len(self, dataset: TileDataset) -> None:
        sampler = RandomTileSampler(dataset, size=16, length=50)
        assert len(sampler) == 50

    def test_weighted_sampling(self) -> None:
        """Test that larger tiles are sampled more frequently."""
        root = os.path.join('tests', 'data', 'tile')
        image_paths = sorted(Path(root, 'images').glob('*.tif'))
        dataset = TileDataset(image_paths=image_paths)

        sampler = RandomTileSampler(dataset, size=16, length=1000)

        tile_counts: dict[int, int] = {}
        for file_index, y, x, patch_size in sampler:
            tile_counts[file_index] = tile_counts.get(file_index, 0) + 1

        assert len(tile_counts) > 0

    def test_generator_reproducibility(self, dataset: TileDataset) -> None:
        """Test that using the same generator seed produces same results."""
        generator1 = torch.Generator().manual_seed(42)
        generator2 = torch.Generator().manual_seed(42)

        sampler1 = RandomTileSampler(dataset, size=16, length=10, generator=generator1)
        sampler2 = RandomTileSampler(dataset, size=16, length=10, generator=generator2)

        samples1 = list(sampler1)
        samples2 = list(sampler2)

        assert samples1 == samples2

    def test_no_generator(self, dataset: TileDataset) -> None:
        """Test that sampler works without a generator."""
        sampler = RandomTileSampler(dataset, size=16, length=5)
        samples = list(sampler)
        assert len(samples) == 5

    def test_small_tile(self) -> None:
        """Test sampling from tiles that are exactly patch_size."""
        root = os.path.join('tests', 'data', 'tile')
        image_paths = sorted(Path(root, 'images').glob('*.tif'))
        dataset = TileDataset(image_paths=image_paths)

        sampler = RandomTileSampler(dataset, size=32, length=10)
        for file_index, y, x, patch_size in sampler:
            dataset.get_tile_size(file_index)
            assert y >= 0
            assert x >= 0

    def test_dataloader(self, dataset: TileDataset) -> None:
        sampler = RandomTileSampler(dataset, size=16, length=10)
        dl = DataLoader(dataset, sampler=sampler, num_workers=0)
        for batch in dl:
            assert 'image' in batch
            break


class TestGridTileSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> TileDataset:
        root = os.path.join('tests', 'data', 'tile')
        image_paths = sorted(Path(root, 'images').glob('*.tif'))
        return TileDataset(image_paths=image_paths)

    def test_iter(self, dataset: TileDataset) -> None:
        sampler = GridTileSampler(dataset, size=16, stride=16)
        for file_index, y, x, patch_size in sampler:
            assert 0 <= file_index < len(dataset)
            assert patch_size == 16
            height, width = dataset.get_tile_size(file_index)
            assert 0 <= y <= height - patch_size
            assert 0 <= x <= width - patch_size

    def test_len(self, dataset: TileDataset) -> None:
        sampler = GridTileSampler(dataset, size=16, stride=16)
        assert len(sampler) > 0
        assert len(sampler) == len(list(sampler))

    def test_default_stride(self, dataset: TileDataset) -> None:
        """Test that stride defaults to size for non-overlapping patches."""
        sampler = GridTileSampler(dataset, size=16)
        assert sampler.stride == 16

    def test_overlapping_patches(self, dataset: TileDataset) -> None:
        """Test that stride < size produces overlapping patches."""
        sampler_no_overlap = GridTileSampler(dataset, size=16, stride=16)
        sampler_overlap = GridTileSampler(dataset, size=16, stride=8)

        assert len(sampler_overlap) > len(sampler_no_overlap)

    def test_small_tile_skipped(self) -> None:
        """Test that tiles smaller than patch size are skipped."""
        root = os.path.join('tests', 'data', 'tile')
        image_paths = sorted(Path(root, 'images').glob('*.tif'))
        dataset = TileDataset(image_paths=image_paths)

        sampler = GridTileSampler(dataset, size=64)

        for file_index, y, x, patch_size in sampler:
            height, width = dataset.get_tile_size(file_index)
            assert height >= 64 and width >= 64

    def test_edge_coverage(self, dataset: TileDataset) -> None:
        """Test that edges of tiles are covered by grid sampling."""
        sampler = GridTileSampler(dataset, size=16, stride=32)

        samples_by_tile: dict[int, list[tuple[int, int]]] = {}
        for file_index, y, x, patch_size in sampler:
            if file_index not in samples_by_tile:
                samples_by_tile[file_index] = []
            samples_by_tile[file_index].append((y, x))

        for file_index, positions in samples_by_tile.items():
            height, width = dataset.get_tile_size(file_index)
            if height < 16 or width < 16:
                continue

            ys = [p[0] for p in positions]
            xs = [p[1] for p in positions]

            assert height - 16 in ys
            assert width - 16 in xs

    def test_empty_result(self) -> None:
        """Test that all-too-small tiles result in empty sampler."""
        root = os.path.join('tests', 'data', 'tile')
        image_paths = sorted(Path(root, 'images').glob('*.tif'))
        dataset = TileDataset(image_paths=image_paths)

        sampler = GridTileSampler(dataset, size=200)
        assert len(sampler) == 0
        assert list(sampler) == []

    def test_dataloader(self, dataset: TileDataset) -> None:
        sampler = GridTileSampler(dataset, size=16, stride=32)
        dl = DataLoader(dataset, sampler=sampler, num_workers=0)
        for batch in dl:
            assert 'image' in batch
            break
