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
        assert sample['mask'].shape == (1, 256, 256)

    def test_len(self, dataset: SolarPlantsBrazil) -> None:
        assert len(dataset) == 1

    def test_plot(self, dataset: SolarPlantsBrazil) -> None:
        sample = dataset[0].copy()
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='Test')
        plt.close()

    def test_invalid_split(self) -> None:
        with pytest.raises(ValueError):
            SolarPlantsBrazil(root='tests/data/solar_plants_brazil', split='foo')

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


def test_download_called(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = {'triggered': False}

    def fake_download_and_extract_archive(*args: Any, **kwargs: Any) -> None:
        called['triggered'] = True
        # Simulate minimal dataset structure
        split = 'train'
        input_dir = tmp_path / split / 'input'
        label_dir = tmp_path / split / 'labels'
        input_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        # Create dummy files
        (input_dir / 'img(0).tif').touch()
        (label_dir / 'target(0).tif').touch()

    monkeypatch.setattr(
        'torchgeo.datasets.solar_plants_brazil.download_and_extract_archive',
        fake_download_and_extract_archive,
    )

    _ = SolarPlantsBrazil(root=tmp_path, split='train', download=True)

    assert called['triggered']


def test_missing_dataset_triggers_error(tmp_path: Path) -> None:
    dataset_root = tmp_path / 'non_existent_dataset'

    with pytest.raises(DatasetNotFoundError):
        SolarPlantsBrazil(root=dataset_root, split='train', download=False)


def test_empty_split_folder_triggers_error(tmp_path: Path) -> None:
    split = 'train'
    input_dir = tmp_path / split / 'input'
    label_dir = tmp_path / split / 'labels'
    input_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(DatasetNotFoundError):
        SolarPlantsBrazil(root=tmp_path, split=split, download=False)
