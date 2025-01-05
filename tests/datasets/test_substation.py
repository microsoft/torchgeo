# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pytest
import torch

from torchgeo.datasets import Substation


class TestSubstation:
    @pytest.fixture
    def dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Generator[Substation, None, None]:
        """Fixture for the Substation."""
        root = os.path.join(os.getcwd(), 'tests', 'data')

        yield Substation(
            root=root,
            bands=[1,2,3],
            use_timepoints=True,
            mask_2d=True,
            timepoint_aggregation='median',
            num_of_timepoints=4,
        )

    @pytest.mark.parametrize(
        'config',
        [
            {'bands': [1,2,3], 'use_timepoints': False, 'mask_2d': True},
            {
                'bands': [1,2,3],
                'use_timepoints': True,
                'timepoint_aggregation': 'concat',
                'num_of_timepoints': 4,
                'mask_2d': False,
            },
            {
                'bands': [1,2,3],
                'use_timepoints': True,
                'timepoint_aggregation': 'median',
                'num_of_timepoints': 4,
                'mask_2d': True,
            },
            {
                'bands': [1,2,3],
                'use_timepoints': True,
                'timepoint_aggregation': 'first',
                'num_of_timepoints': 4,
                'mask_2d': False,
            },
            {
                'bands': [1,2,3],
                'use_timepoints': True,
                'timepoint_aggregation': 'random',
                'num_of_timepoints': 4,
                'mask_2d': True,
            },
            {'bands': [1,2,3], 'use_timepoints': False, 'mask_2d': False},
            {
                'bands': [1,2,3],
                'use_timepoints': False,
                'timepoint_aggregation': 'first',
                'num_of_timepoints': 4,
                'mask_2d': False,
            },
            {
                'bands': [1,2,3],
                'use_timepoints': False,
                'timepoint_aggregation': 'random',
                'num_of_timepoints': 4,
                'mask_2d': True,
            },
        ],
    )
    def test_getitem_semantic(self, config: dict[str, Any]) -> None:
        root = os.path.join(os.getcwd(), 'tests', 'data')
        dataset = Substation(root=root, **config)

        x = dataset[0]
        assert isinstance(x, dict), f'Expected dict, got {type(x)}'
        assert isinstance(
            x['image'], torch.Tensor
        ), 'Expected image to be a torch.Tensor'
        assert isinstance(x['mask'], torch.Tensor), 'Expected mask to be a torch.Tensor'

    def test_len(self, dataset: Substation) -> None:
        """Test the length of the dataset."""
        assert len(dataset) == 5

    def test_output_shape(self, dataset: Substation) -> None:
        """Test the output shape of the dataset."""
        x = dataset[0]
        assert x['image'].shape == torch.Size([3, 32, 32])
        assert x['mask'].shape == torch.Size([2, 32, 32])

    def test_plot(self, dataset: Substation) -> None:
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
        # Simulating that files are already present by copying them to the target directory
        url_for_images = os.path.join(
            'tests', 'data', 'substation', 'image_stack.tar.gz'
        )
        url_for_masks = os.path.join('tests', 'data', 'substation', 'mask.tar.gz')

        # Copy files to the temporary directory to simulate already downloaded files
        shutil.copy(url_for_images, tmp_path)
        shutil.copy(url_for_masks, tmp_path)

        # No download should be attempted, since the files are already present
        # Mock the _download method to simulate the behavior
        monkeypatch.setattr(dataset, '_download', MagicMock())
        dataset._download()  # This will now call the mocked method

    def test_download(
        self, dataset: Substation, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test the _download method of the dataset."""
        # Mock the download_url and extract_archive functions
        mock_download_url = MagicMock()
        mock_extract_archive = MagicMock()
        monkeypatch.setattr(
            'torchgeo.datasets.substation.download_url', mock_download_url
        )
        monkeypatch.setattr(
            'torchgeo.datasets.substation.extract_archive', mock_extract_archive
        )

        # Call the _download method
        dataset._download()

        # Check that download_url was called twice
        mock_download_url.assert_called()
        assert mock_download_url.call_count == 2

        # Check that extract_archive was called twice
        mock_extract_archive.assert_called()
        assert mock_extract_archive.call_count == 2


if __name__ == '__main__':
    pytest.main([__file__])
