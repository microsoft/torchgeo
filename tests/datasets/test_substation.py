# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, Substation


class TestSubstation:
    @pytest.fixture(
        params=[
            {'bands': [1, 2, 3], 'mask_2d': True},
            {
                'bands': [1, 2, 3],
                'timepoint_aggregation': 'concat',
                'num_of_timepoints': 4,
                'mask_2d': False,
            },
            {
                'bands': [1, 2, 3],
                'timepoint_aggregation': 'median',
                'num_of_timepoints': 4,
                'mask_2d': True,
            },
            {
                'bands': [1, 2, 3],
                'timepoint_aggregation': 'first',
                'num_of_timepoints': 4,
                'mask_2d': False,
            },
            {
                'bands': [1, 2, 3],
                'timepoint_aggregation': None,
                'num_of_timepoints': 3,
                'mask_2d': False,
            },
            {
                'bands': [1, 2, 3],
                'timepoint_aggregation': None,
                'num_of_timepoints': 5,
                'mask_2d': False,
            },
            {
                'bands': [1, 2, 3],
                'timepoint_aggregation': 'random',
                'num_of_timepoints': 4,
                'mask_2d': True,
            },
            {'bands': [1, 2, 3], 'mask_2d': False},
            {
                'bands': [1, 2, 3],
                'timepoint_aggregation': 'first',
                'num_of_timepoints': 4,
                'mask_2d': False,
            },
            {
                'bands': [1, 2, 3],
                'timepoint_aggregation': 'random',
                'num_of_timepoints': 4,
                'mask_2d': True,
            },
        ]
    )
    def dataset(self, request: pytest.FixtureRequest, tmp_path: Path) -> Substation:
        """Fixture for the Substation with parameterization."""
        root = os.path.join('tests', 'data', 'substation')
        transforms = nn.Identity()
        return Substation(root, transforms=transforms, **request.param)

    def test_getitem(self, dataset: Substation) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert len(dataset) == 5

        match dataset.timepoint_aggregation:
            case 'concat':
                assert x['image'].shape == torch.Size([12, 32, 32])
            case 'median':
                assert x['image'].shape == torch.Size([3, 32, 32])
            case 'first' | 'random':
                assert x['image'].shape == torch.Size([3, 32, 32])
            case _:
                assert x['image'].shape == torch.Size(
                    [dataset.num_of_timepoints, 3, 32, 32]
                )

        if dataset.mask_2d:
            assert x['mask'].shape == torch.Size([2, 32, 32])
        else:
            assert x['mask'].shape == torch.Size([32, 32])

    def test_download(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        url = os.path.join('tests', 'data', 'substation')
        filename = Substation.filename_images
        maskname = Substation.filename_masks
        monkeypatch.setattr(Substation, 'url_for_images', os.path.join(url, filename))
        monkeypatch.setattr(Substation, 'url_for_masks', os.path.join(url, maskname))
        Substation(tmp_path, download=True)

    def test_extract(self, tmp_path: Path) -> None:
        root = os.path.join('tests', 'data', 'substation')
        filename = Substation.filename_images
        maskname = Substation.filename_masks
        shutil.copyfile(os.path.join(root, filename), tmp_path / filename)
        shutil.copyfile(os.path.join(root, maskname), tmp_path / maskname)
        Substation(tmp_path)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            Substation(tmp_path)

    def test_plot(self, dataset: Substation) -> None:
        sample = dataset[0]
        dataset.plot(sample, suptitle='Test')
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample)
        plt.close()
