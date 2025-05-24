# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Unit tests for the SolarPlantsBrazil dataset."""

import os
import tempfile
import warnings

import matplotlib.pyplot as plt
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
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
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
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
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
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            sample = dataset[0]
            fig = dataset.plot(sample, suptitle='Test Sample')

        assert fig is not None
        plt.close(fig)
