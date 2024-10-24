import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from torchgeo.datasets import DatasetNotFoundError, SubstationDataset


@dataclass
class Args:
    data_dir: Path
    in_channels: int
    use_timepoints: bool
    normalizing_type: str
    normalizing_factor: NDArray[np.float64]
    means: NDArray[np.float64]
    stds: NDArray[np.float64]
    mask_2d: bool
    model_type: str


class TestSubstationDataset:
    @pytest.fixture(
        params=[
            {
                'image_files': ['image_1.npz', 'image_2.npz'],
                'geo_transforms': None,
                'color_transforms': None,
                'image_resize': None,
                'mask_resize': None,
            }
        ]
    )
    def dataset(self, tmp_path: Path, request: pytest.FixtureRequest) -> SubstationDataset:
        """
        Fixture to create a mock dataset with specified parameters.
        """
        args = Args(
            data_dir=tmp_path,
            in_channels=4,
            use_timepoints=False,
            normalizing_type='zscore',
            normalizing_factor=np.array([1.0]),
            means=np.array([0.5]),
            stds=np.array([0.1]),
            mask_2d=True,
            model_type='segmentation',
        )

        # Creating mock image and mask files
        for filename in request.param['image_files']:
            os.makedirs(os.path.join(tmp_path, 'image_stack'), exist_ok=True)
            os.makedirs(os.path.join(tmp_path, 'mask'), exist_ok=True)
            np.savez_compressed(os.path.join(tmp_path, 'image_stack', filename), arr_0=np.random.rand(4, 128, 128))
            np.savez_compressed(os.path.join(tmp_path, 'mask', filename), arr_0=np.random.randint(0, 4, (128, 128)))

        return SubstationDataset(
            args,
            image_files=request.param['image_files'],
            geo_transforms=request.param['geo_transforms'],
            color_transforms=request.param['color_transforms'],
            image_resize=request.param['image_resize'],
            mask_resize=request.param['mask_resize'],
        )

    def test_getitem(self, dataset: SubstationDataset) -> None:
        """Test that __getitem__ returns a valid image and mask tensor."""
        data = dataset[0]
        image = data["image"]
        mask = data["mask"]
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape[0] == 4  # Checking number of channels
        assert mask.shape == (1, 128, 128)

    def test_len(self, dataset: SubstationDataset) -> None:
        """Test that __len__ returns the correct length of the dataset."""
        assert len(dataset) == 2

    def test_already_downloaded(self, tmp_path: Path) -> None:
        """Test dataset initialization when data is already downloaded."""
        
        os.makedirs(os.path.join(tmp_path, 'image_stack'))
        os.makedirs(os.path.join(tmp_path, 'mask'))

        assert os.path.exists(os.path.join(tmp_path, 'image_stack'))
        assert os.path.exists(os.path.join(tmp_path, 'mask'))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        """Test dataset initialization when data is not downloaded, expecting DatasetNotFoundError."""
        args = Args(
            data_dir=tmp_path,
            in_channels=4,
            use_timepoints=False,
            normalizing_type='zscore',
            normalizing_factor=np.array([1.0]),
            means=np.array([0.5]),
            stds=np.array([0.1]),
            mask_2d=True,
            model_type='segmentation',
        )

        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SubstationDataset(args, image_files=[])

    def test_plot(self, dataset: SubstationDataset) -> None:
        """Test that the plot function runs without throwing exceptions."""
        dataset.plot()

    def test_corrupted(self, tmp_path: Path) -> None:
        """Test dataset loading with corrupted files."""
        args = Args(
            data_dir=tmp_path,
            in_channels=4,
            use_timepoints=False,
            normalizing_type='zscore',
            normalizing_factor=np.array([1.0]),
            means=np.array([0.5]),
            stds=np.array([0.1]),
            mask_2d=True,
            model_type='segmentation',
        )

        # Creating corrupted files
        os.makedirs(os.path.join(tmp_path, 'image_stack'))
        os.makedirs(os.path.join(tmp_path, 'mask'))
        with open(os.path.join(tmp_path, 'image_stack', 'image_1.npz'), 'w') as f:
            f.write('corrupted')

        with pytest.raises(Exception):
            SubstationDataset(args, image_files=['image_1.npz'])
