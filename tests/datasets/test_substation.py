# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn

from torchgeo.datasets import DatasetNotFoundError, Substation


class TestSubstation:
    @pytest.fixture(
        params=[
            {'bands': [1, 2, 3], 'use_timepoints': False, 'mask_2d': True},
            {
                'bands': [1, 2, 3],
                'use_timepoints': True,
                'timepoint_aggregation': 'concat',
                'num_of_timepoints': 4,
                'mask_2d': False,
            },
            {
                'bands': [1, 2, 3],
                'use_timepoints': True,
                'timepoint_aggregation': 'median',
                'num_of_timepoints': 4,
                'mask_2d': True,
            },
            {
                'bands': [1, 2, 3],
                'use_timepoints': True,
                'timepoint_aggregation': 'first',
                'num_of_timepoints': 4,
                'mask_2d': False,
            },
            {
                'bands': [1, 2, 3],
                'use_timepoints': True,
                'timepoint_aggregation': None,
                'num_of_timepoints': 3,
                'mask_2d': False,
            },
            {
                'bands': [1, 2, 3],
                'use_timepoints': True,
                'timepoint_aggregation': None,
                'num_of_timepoints': 5,
                'mask_2d': False,
            },
            {
                'bands': [1, 2, 3],
                'use_timepoints': True,
                'timepoint_aggregation': 'random',
                'num_of_timepoints': 4,
                'mask_2d': True,
            },
            {'bands': [1, 2, 3], 'use_timepoints': False, 'mask_2d': False},
            {
                'bands': [1, 2, 3],
                'use_timepoints': False,
                'timepoint_aggregation': 'first',
                'num_of_timepoints': 4,
                'mask_2d': False,
            },
            {
                'bands': [1, 2, 3],
                'use_timepoints': False,
                'timepoint_aggregation': 'random',
                'num_of_timepoints': 4,
                'mask_2d': True,
            },
        ]
    )
    def dataset(
        self, request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Substation:
        """Fixture for the Substation with parameterization."""
        root = os.path.join(os.getcwd(), 'tests', 'data', 'substation')
        transforms = nn.Identity()
        return Substation(root=root, transforms=transforms, **request.param)

    def test_getitem_semantic(self, dataset: Substation) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert len(dataset) == 5

    def test_output_shape(self, dataset: Substation) -> None:
        """Test the output shape of the dataset."""
        x = dataset[0]
        if dataset.use_timepoints:
            if dataset.timepoint_aggregation == 'concat':
                assert x['image'].shape == torch.Size([12, 32, 32])
            elif dataset.timepoint_aggregation == 'median':
                assert x['image'].shape == torch.Size([3, 32, 32])
            else:
                assert x['image'].shape == torch.Size(
                    [dataset.num_of_timepoints, 3, 32, 32]
                )
        else:
            if (
                dataset.timepoint_aggregation == 'first'
                or dataset.timepoint_aggregation == 'random'
            ):
                assert x['image'].shape == torch.Size([3, 32, 32])
            else:
                assert x['image'].shape == torch.Size(
                    [dataset.num_of_timepoints, 3, 32, 32]
                )

        if dataset.mask_2d:
            assert x['mask'].shape == torch.Size([2, 32, 32])
        else:
            assert x['mask'].shape == torch.Size([32, 32])

    def test_plot(self, dataset: Substation) -> None:
        root = os.path.join(os.getcwd(), 'tests', 'data', 'substation')
        sample = dataset[0]
        dataset.plot(sample, suptitle='Test')
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample)
        plt.close()

    def test_already_downloaded(
        self, dataset: Substation, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the dataset doesn't re-download if already present."""
        url_for_images = os.path.join(
            'tests', 'data', 'substation', 'image_stack.tar.gz'
        )
        url_for_masks = os.path.join('tests', 'data', 'substation', 'mask.tar.gz')
        shutil.copy(url_for_images, tmp_path)
        shutil.copy(url_for_masks, tmp_path)

        monkeypatch.setattr(dataset, '_download', lambda: None)
        dataset._download()

    def test_download(
        self, dataset: Substation, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the _download method of the dataset."""
        monkeypatch.setattr(
            'torchgeo.datasets.substation.download_url', lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            'torchgeo.datasets.substation.extract_archive', lambda *args, **kwargs: None
        )
        dataset._download()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            Substation(
                bands=[1, 2, 3],
                use_timepoints=True,
                mask_2d=True,
                timepoint_aggregation='median',
                num_of_timepoints=4,
                root=tmp_path,
            )

    def test_not_downloaded_with_download(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filename = 'image_stack'
        maskname = 'mask'
        source_image_path = os.path.join('tests', 'data', 'substation', filename)
        source_mask_path = os.path.join('tests', 'data', 'substation', maskname)
        target_image_path = tmp_path / filename
        target_mask_path = tmp_path / maskname

        def mock_download(self: Substation) -> None:
            shutil.copytree(source_image_path, target_image_path)
            shutil.copytree(source_mask_path, target_mask_path)

        monkeypatch.setattr(
            'torchgeo.datasets.substation.Substation._download', mock_download
        )
        monkeypatch.setattr(
            'torchgeo.datasets.substation.Substation._extract', lambda self: None
        )

        Substation(
            bands=[1, 2, 3],
            use_timepoints=True,
            mask_2d=True,
            timepoint_aggregation='median',
            num_of_timepoints=4,
            root=tmp_path,
            download=True,
        )

    def test_extract(self, tmp_path: Path) -> None:
        filename = Substation.filename_images
        maskname = Substation.filename_masks
        shutil.copyfile(
            os.path.join('tests', 'data', 'substation', filename), tmp_path / filename
        )
        shutil.copyfile(
            os.path.join('tests', 'data', 'substation', maskname), tmp_path / maskname
        )
        Substation(
            bands=[1, 2, 3],
            use_timepoints=True,
            mask_2d=True,
            timepoint_aggregation='median',
            num_of_timepoints=4,
            root=tmp_path,
        )
