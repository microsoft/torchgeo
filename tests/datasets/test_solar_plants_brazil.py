# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests for the SolarPlantsBrazil dataset."""

from pathlib import Path

import pytest
import torch
from matplotlib import pyplot as plt
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, SolarPlantsBrazil


class TestSolarPlantsBrazil:
    @pytest.fixture
    def dataset_root(self) -> str:
        # Point to the test data folder you generated and committed
        return 'tests/data/solar_plants_brazil'

    @pytest.fixture(params=['train'])
    def dataset(
        self, dataset_root: str, request: pytest.FixtureRequest
    ) -> SolarPlantsBrazil:
        split = request.param
        return SolarPlantsBrazil(root=dataset_root, split=split)

    def test_getitem(self, dataset: SolarPlantsBrazil) -> None:
        sample = dataset[0]
        assert isinstance(sample, dict)
        assert isinstance(sample['image'], torch.Tensor)
        assert isinstance(sample['mask'], torch.Tensor)
        assert sample['image'].shape == (4, 32, 32)
        assert sample['mask'].shape == (1, 32, 32)

    def test_len(self, dataset: SolarPlantsBrazil) -> None:
        assert len(dataset) == 1

    def test_plot(self, dataset: SolarPlantsBrazil) -> None:
        sample = dataset[0].copy()
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='Test')
        plt.close()

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SolarPlantsBrazil(root='tests/data/solar_plants_brazil', split='foo')

    def test_missing_dataset_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SolarPlantsBrazil(root=tmp_path, split='train', download=False)

    def test_download_called(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        called = {'flag': False}

        from typing import Any

        def fake_download(self: Any) -> None:
            called['flag'] = True

        # Correctly patch the class method _download
        monkeypatch.setattr(
            'torchgeo.datasets.solar_plants_brazil.SolarPlantsBrazil._download',
            fake_download,
        )

        with pytest.raises(DatasetNotFoundError):
            SolarPlantsBrazil(root=tmp_path, split='train', download=True)

        assert called['flag']
