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
        self.normalizing_factor: Any = np.array([[0, 0.5, 1.0]], dtype=np.float32)
        self.means: Any = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.stds: Any = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[SubstationDataset, None, None]:
    """Fixture for the SubstationDataset."""
    args = Args()
    image_files = ['image_0.npz', 'image_1.npz']

    yield SubstationDataset(args, image_files)


def test_len(dataset: SubstationDataset) -> None:
    """Test the length of the dataset."""
    assert len(dataset) == 2


def test_getitem_semantic(dataset: SubstationDataset) -> None:
    x = dataset[0]
    assert isinstance(x, dict)
    assert isinstance(x['image'], torch.Tensor)
    assert isinstance(x['mask'], torch.Tensor)


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
        'tests', 'data', 'substation_seg', 'image_stack.tar.gz'
    )
    url_for_masks = os.path.join('tests', 'data', 'substation_seg', 'mask.tar.gz')

    # Copy files to the temporary directory to simulate already downloaded files
    shutil.copy(url_for_images, tmp_path)
    shutil.copy(url_for_masks, tmp_path)

    # No download should be attempted, since the files are already present
    # Mock the _download method to simulate the behavior
    monkeypatch.setattr(dataset, '_download', MagicMock())
    dataset._download()  # This will now call the mocked method
