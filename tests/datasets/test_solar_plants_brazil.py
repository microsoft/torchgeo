# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests for the SolarPlantsBrazil dataset."""

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from matplotlib import pyplot as plt

from torchgeo.datasets import DatasetNotFoundError, SolarPlantsBrazil


class TestSolarPlantsBrazil:
    @pytest.fixture
    def dataset_root(self) -> str:
        return os.path.join('tests', 'data', 'solar_plants_brazil')

    @pytest.fixture
    def dataset(self, dataset_root: str) -> SolarPlantsBrazil:
        return SolarPlantsBrazil(root=dataset_root, split='train')

    def test_getitem(self, dataset: SolarPlantsBrazil) -> None:
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert isinstance(sample['image'], torch.Tensor)
        assert isinstance(sample['mask'], torch.Tensor)
        assert sample['image'].shape == (4, 256, 256)
        assert sample['mask'].shape == (256, 256)

    def test_len(self, dataset: SolarPlantsBrazil) -> None:
        assert len(dataset) == 1

    def test_plot(self, dataset: SolarPlantsBrazil) -> None:
        sample = dataset[0]
        sample['prediction'] = sample['mask']
        dataset.plot(sample, suptitle='Test')
        plt.close()

    def test_invalid_split(self) -> None:
        with pytest.raises(ValueError, match='Invalid split'):
            root = os.path.join('test', 'data', 'solar_plants_brazil')
            SolarPlantsBrazil(root=root, split='foo')  # type: ignore[arg-type]

    def test_missing_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError):
            SolarPlantsBrazil(root=tmp_path, split='train', download=False)

    def test_getitem_with_transform(self, dataset_root: str) -> None:
        def dummy_transform(sample: dict[str, Any]) -> dict[str, Any]:
            sample['image'] += 1
            return sample

        dataset = SolarPlantsBrazil(
            root=dataset_root, split='train', transforms=dummy_transform
        )
        sample = dataset[0]
        assert torch.all(sample['image'] > 0)

    def test_already_downloaded(self, dataset_root: str) -> None:
        SolarPlantsBrazil(root=dataset_root, split='train')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError):
            SolarPlantsBrazil(root=tmp_path, split='train', download=False)
