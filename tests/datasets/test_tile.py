# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import torch

from torchgeo.datasets import TileDataset


class TestTileDataset:
    @pytest.fixture
    def image_paths(self) -> list[str]:
        root = os.path.join('tests', 'data', 'tile', 'images')
        return sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.tif')]
        )

    @pytest.fixture
    def mask_paths(self) -> list[str]:
        root = os.path.join('tests', 'data', 'tile', 'masks')
        return sorted(
            [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.tif')]
        )

    @pytest.fixture
    def dataset(self, image_paths: list[str], mask_paths: list[str]) -> TileDataset:
        return TileDataset(image_paths, mask_paths)

    @pytest.fixture
    def dataset_images_only(self, image_paths: list[str]) -> TileDataset:
        return TileDataset(image_paths)

    def test_getitem(self, dataset: TileDataset) -> None:
        sample = dataset[(0, 0, 0, 16)]
        assert isinstance(sample, dict)
        assert 'image' in sample
        assert 'mask' in sample
        assert isinstance(sample['image'], torch.Tensor)
        assert isinstance(sample['mask'], torch.Tensor)
        assert sample['image'].shape == (3, 16, 16)
        assert sample['mask'].shape == (16, 16)

    def test_getitem_images_only(self, dataset_images_only: TileDataset) -> None:
        sample = dataset_images_only[(0, 0, 0, 16)]
        assert 'image' in sample
        assert 'mask' not in sample
        assert sample['image'].shape == (3, 16, 16)

    def test_len(self, dataset: TileDataset) -> None:
        assert len(dataset) == 3

    def test_get_tile_size(self, dataset: TileDataset) -> None:
        height, width = dataset.get_tile_size(0)
        assert height == 64
        assert width == 64

        height, width = dataset.get_tile_size(1)
        assert height == 32
        assert width == 48

        height, width = dataset.get_tile_size(2)
        assert height == 100
        assert width == 80

    def test_getitem_out_of_bounds(self, dataset: TileDataset) -> None:
        with pytest.raises(IndexError, match='out of bounds'):
            dataset[(10, 0, 0, 16)]
        with pytest.raises(IndexError, match='out of bounds'):
            dataset[(-1, 0, 0, 16)]

    def test_get_tile_size_out_of_bounds(self, dataset: TileDataset) -> None:
        with pytest.raises(IndexError, match='out of bounds'):
            dataset.get_tile_size(10)
        with pytest.raises(IndexError, match='out of bounds'):
            dataset.get_tile_size(-1)

    def test_transforms(self, image_paths: list[str], mask_paths: list[str]) -> None:
        def custom_transform(
            sample: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            sample['transformed'] = True
            return sample

        dataset = TileDataset(image_paths, mask_paths, transforms=custom_transform)
        sample = dataset[(0, 0, 0, 16)]
        assert sample.get('transformed') is True

    def test_mismatched_paths_length(self, image_paths: list[str]) -> None:
        with pytest.raises(ValueError, match='same length'):
            TileDataset(image_paths, image_paths[:1])

    def test_multi_path_per_sample(self, image_paths: list[str]) -> None:
        multi_paths: list[list[str]] = [[p, p] for p in image_paths]
        dataset = TileDataset(multi_paths)
        sample = dataset[(0, 0, 0, 16)]
        assert sample['image'].shape == (6, 16, 16)

    def test_mixed_single_and_multi_paths(
        self, image_paths: list[str], mask_paths: list[str]
    ) -> None:
        mixed_paths: list[str | list[str]] = [
            image_paths[0],
            [image_paths[1], image_paths[1]],
            image_paths[2],
        ]
        dataset = TileDataset(mixed_paths, mask_paths)
        assert len(dataset) == 3
        sample0 = dataset[(0, 0, 0, 16)]
        assert sample0['image'].shape == (3, 16, 16)
        sample1 = dataset[(1, 0, 0, 16)]
        assert sample1['image'].shape == (6, 16, 16)
