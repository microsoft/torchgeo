# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests for the SolarPlantsBrazil dataset."""

import os
import tempfile
import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
import rasterio
import torch
from rasterio.errors import NotGeoreferencedWarning

from torchgeo.datasets import SolarPlantsBrazil

# Suppress rasterio georeferencing warnings for dummy data
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)


def create_dummy_dataset_structure(root: str) -> None:
    """Create a dummy SolarPlantsBrazil-like folder with one image-mask pair."""
    input_dir = os.path.join(root, 'train', 'input')
    label_dir = os.path.join(root, 'train', 'labels')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    image = torch.randint(0, 255, (4, 256, 256), dtype=torch.uint8).numpy()
    mask = (torch.rand(256, 256) > 0.5).to(torch.uint8).numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
        with rasterio.open(
            os.path.join(input_dir, 'img(1).tif'),
            'w',
            driver='GTiff',
            height=256,
            width=256,
            count=4,
            dtype='uint8',
        ) as dst:
            dst.write(image)

        with rasterio.open(
            os.path.join(label_dir, 'target(1).tif'),
            'w',
            driver='GTiff',
            height=256,
            width=256,
            count=1,
            dtype='uint8',
        ) as dst:
            dst.write(mask, 1)


def test_solar_plants_brazil_getitem_and_len() -> None:
    """Test __getitem__ and __len__ for a minimal dataset."""
    with tempfile.TemporaryDirectory() as root:
        create_dummy_dataset_structure(root)
        dataset = SolarPlantsBrazil(root=root, split='train')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
            sample = dataset[0]

        assert len(dataset) == 1
        assert isinstance(sample, dict)
        assert 'image' in sample and 'mask' in sample
        assert sample['image'].shape == (4, 256, 256)
        assert sample['mask'].shape == (1, 256, 256)


def test_solar_plants_brazil_plot() -> None:
    """Test the plot method returns a matplotlib figure."""
    with tempfile.TemporaryDirectory() as root:
        create_dummy_dataset_structure(root)
        dataset = SolarPlantsBrazil(root=root, split='train')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
            sample = dataset[0]
            fig = dataset.plot(sample, suptitle='Test Sample')

        assert fig is not None
        plt.close(fig)


def test_missing_images_raises_error() -> None:
    """Test error raised when no input images exist."""
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, 'train', 'input'), exist_ok=True)
        os.makedirs(os.path.join(root, 'train', 'labels'), exist_ok=True)
        with pytest.raises(FileNotFoundError):
            SolarPlantsBrazil(root=root, split='train')


def test_verify_raises_runtimeerror_if_download_false() -> None:
    """Test error raised when dataset is missing and download is False."""
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(RuntimeError):
            SolarPlantsBrazil(root=root, split='train', download=False)


def test_transforms_are_applied() -> None:
    """Test that custom transforms are applied."""

    def dummy_transform(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample['image'] += 1
        return sample

    with tempfile.TemporaryDirectory() as root:
        create_dummy_dataset_structure(root)
        dataset = SolarPlantsBrazil(
            root=root, split='train', transforms=dummy_transform
        )
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
            sample = dataset[0]
        assert (sample['image'] > 0).all()


def test_download_invoked_if_missing() -> None:
    """Test that snapshot_download is invoked when dataset is missing."""
    with patch(
        'torchgeo.datasets.solar_plants_brazil.snapshot_download'
    ) as mock_download:
        with tempfile.TemporaryDirectory() as root:
            # Do NOT create any subfolders like 'train/input' beforehand.
            # This is intentional so that _verify() thinks the dataset is missing.
            try:
                SolarPlantsBrazil(root=root, split='train', download=True)
            except FileNotFoundError:
                # Expected since no files are actually written
                pass

            assert mock_download.called
