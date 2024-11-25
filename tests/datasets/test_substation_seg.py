import os
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torchvision.transforms as transforms

from torchgeo.datasets import SubstationDataset


class Args:
    """Mocked arguments for testing SubstationDataset."""

    def __init__(self) -> None:
        self.data_dir: str = os.path.join(os.getcwd(), 'tests', 'data')
        self.in_channels: int = 13
        self.use_timepoints: bool = True
        self.normalizing_type: str = 'percentile'
        self.mask_2d: bool = True
        self.model_type: str = 'vanilla_unet'
        self.timepoint_aggregation: str = 'median'
        self.color_transforms: bool = False
        self.geo_transforms: bool = False
        self.normalizing_factor: Any = np.array([[0, 0.5, 1.0]], dtype=np.float32)
        self.means: Any = np.array(
            [
                [[1431]],
                [[1233]],
                [[1209]],
                [[1192]],
                [[1448]],
                [[2238]],
                [[2609]],
                [[2537]],
                [[2828]],
                [[884]],
                [[20]],
                [[2226]],
                [[1537]],
            ],
            dtype=np.float32,
        )
        self.stds: Any = np.array(
            [
                [[157]],
                [[254]],
                [[290]],
                [[420]],
                [[363]],
                [[457]],
                [[575]],
                [[606]],
                [[630]],
                [[156]],
                [[3]],
                [[554]],
                [[523]],
            ],
            dtype=np.float32,
        )


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[SubstationDataset, None, None]:
    """Fixture for the SubstationDataset."""
    args = Args()
    image_files = ['image_0.npz', 'image_1.npz']

    yield SubstationDataset(args, image_files)


@pytest.mark.parametrize(
    'config',
    [
        {
            'normalizing_type': 'percentile',
            'in_channels': 3,
            'use_timepoints': False,
            'mask_2d': True,
        },
        {
            'normalizing_type': 'zscore',
            'in_channels': 9,
            'model_type': 'swin',
            'use_timepoints': True,
            'timepoint_aggregation': 'concat',
            'mask_2d': False,
        },
        {
            'normalizing_type': None,
            'in_channels': 12,
            'use_timepoints': True,
            'timepoint_aggregation': 'median',
            'mask_2d': True,
            'normalizing_factor': 1.0,
        },
        {
            'normalizing_type': None,
            'in_channels': 5,
            'use_timepoints': True,
            'timepoint_aggregation': 'first',
            'mask_2d': False,
            'normalizing_factor': 1.0,
        },
        {
            'normalizing_type': None,
            'in_channels': 4,
            'use_timepoints': True,
            'timepoint_aggregation': 'random',
            'mask_2d': True,
            'normalizing_factor': 1.0,
        },
        {
            'normalizing_type': 'zscore',
            'in_channels': 2,
            'use_timepoints': False,
            'mask_2d': False,
            'color_transforms': True,
            'geo_transforms': True,
        },
        {
            'normalizing_type': None,
            'in_channels': 5,
            'use_timepoints': False,
            'timepoint_aggregation': 'first',
            'mask_2d': False,
            'normalizing_factor': 1.0,
        },
        {
            'normalizing_type': None,
            'in_channels': 4,
            'use_timepoints': False,
            'timepoint_aggregation': 'random',
            'mask_2d': True,
            'normalizing_factor': 1.0,
        },
    ],
)
def test_getitem_semantic(config: dict[str, Any]) -> None:
    args = Args()
    for key, value in config.items():
        setattr(args, key, value)  # Dynamically set arguments for each config

    # Setting mock paths and creating dataset instance
    image_files = ['image_0.npz', 'image_1.npz']
    image_resize = transforms.Compose(
        [transforms.Resize(228, transforms.InterpolationMode.BICUBIC)]
    )
    mask_resize = transforms.Compose(
        [transforms.Resize(228, transforms.InterpolationMode.NEAREST)]
    )
    dataset = SubstationDataset(
        args, image_files, image_resize=image_resize, mask_resize=mask_resize
    )

    x = dataset[0]
    assert isinstance(x, dict), f'Expected dict, got {type(x)}'
    assert isinstance(x['image'], torch.Tensor), 'Expected image to be a torch.Tensor'
    assert isinstance(x['mask'], torch.Tensor), 'Expected mask to be a torch.Tensor'


def test_len(dataset: SubstationDataset) -> None:
    """Test the length of the dataset."""
    assert len(dataset) == 2


def test_output_shape(dataset: SubstationDataset) -> None:
    """Test the output shape of the dataset."""
    x = dataset[0]
    assert x['image'].shape == torch.Size([13, 228, 228])
    assert x['mask'].shape == torch.Size([2, 228, 228])


def test_plot(dataset: SubstationDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the plot method of the dataset."""
    # Mock plt.show to avoid showing the plot during the test
    mock_show = MagicMock()
    monkeypatch.setattr(plt, 'show', mock_show)

    # Mock np.random.randint to return a fixed index (e.g., 0)
    monkeypatch.setattr(
        np.random, 'randint', lambda low, high: 0
    )  # Correct the lambda to accept 2 arguments

    # Mock __getitem__ to return a sample with an image (3 channels) and a mask
    mock_image = torch.rand(3, 228, 228)  # Create a dummy 3-channel image (RGB)
    mock_mask = torch.randint(0, 4, (228, 228))  # Create a dummy mask
    monkeypatch.setattr(
        dataset, '__getitem__', lambda idx: {'image': mock_image, 'mask': mock_mask}
    )

    # Call the plot method
    dataset.plot()


def test_already_downloaded(
    dataset: SubstationDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_verify(dataset: SubstationDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the _verify method of the dataset."""
    # Mock os.path.exists to return False for the image and mask directories
    monkeypatch.setattr(os.path, 'exists', lambda path: False)

    # Mock the _download method to avoid actually downloading the dataset
    mock_download = MagicMock()
    monkeypatch.setattr(dataset, '_download', mock_download)

    # Call the _verify method
    dataset._verify()

    # Check that the _download method was called
    mock_download.assert_called_once()


def test_download(
    dataset: SubstationDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
