import os
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
        self.data_dir: str = os.path.join(os.path.dirname(__file__), '../data/substation_seg')
        self.in_channels: int = 3
        self.use_timepoints: bool = True
        self.normalizing_type: str = "percentile"
        self.normalizing_factor: np.ndarray[Any, np.dtype[np.float_]] = np.array([[0, 0.5, 1.0]])
        self.mask_2d: bool = True
        self.model_type: str = "swin"
        self.timepoint_aggregation: str = "concat"
        self.means: np.ndarray[Any, np.dtype[np.float_]] = np.array([0.485, 0.456, 0.406])
        self.stds: np.ndarray[Any, np.dtype[np.float_]] = np.array([0.229, 0.224, 0.225])


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[SubstationDataset, None, None]:
    """Fixture for the SubstationDataset."""
    args = Args()
    image_files = ['image_001.npz', 'image_002.npz']
    
    # Mock os.path.exists to prevent actual file checking
    monkeypatch.setattr(os.path, 'exists', lambda x: True)
    
    # Mock numpy load to simulate image and mask data
    np_load_mock = MagicMock()
    np_load_mock.return_value = {'arr_0': np.random.rand(4, 13, 128, 128)}
    monkeypatch.setattr(np, 'load', np_load_mock)
    
    yield SubstationDataset(args, image_files)


def test_len(dataset: SubstationDataset) -> None:
    """Test the length of the dataset."""
    assert len(dataset) == 2


def test_getitem(dataset: SubstationDataset) -> None:
    """Test the __getitem__ method of the dataset."""
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert 'image' in sample and 'mask' in sample
    assert isinstance(sample['image'], torch.Tensor)
    assert isinstance(sample['mask'], torch.Tensor)
    assert sample['image'].shape[1:] == (128, 128)  # Ensure the final image dimensions
    assert sample['mask'].shape == (1, 128, 128)    # Ensure the mask is 2D with one channel


def test_plot(dataset: SubstationDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the plot method of the dataset."""
    mock_show = MagicMock()
    monkeypatch.setattr(plt, "show", mock_show)
    dataset.plot()
    mock_show.assert_called_once()


def test_verify_download(dataset: SubstationDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the verify and download logic."""
    mock_download = MagicMock()
    monkeypatch.setattr(dataset, "_download", mock_download)
    dataset._verify()
    mock_download.assert_not_called()  # Since paths are mocked as existing


def test_download(dataset: SubstationDataset, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the download functionality."""
    mock_download_url = MagicMock()
    mock_extract_archive = MagicMock()
    
    # Set the correct relative paths for the test dataset
    monkeypatch.setattr(dataset, 'url_for_images', os.path.join(os.path.dirname(__file__), '../data/substation_seg/image_stack.tar.gz'))
    monkeypatch.setattr(dataset, 'url_for_masks', os.path.join(os.path.dirname(__file__), '../data/substation_seg/mask.tar.gz'))

    monkeypatch.setattr('torchgeo.datasets.utils.download_url', mock_download_url)
    monkeypatch.setattr('torchgeo.datasets.utils.extract_archive', mock_extract_archive)

    dataset._download()
    
    # Check if download and extraction were triggered
    mock_download_url.assert_called()
    mock_extract_archive.assert_called()
