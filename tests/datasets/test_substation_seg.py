import os
from pathlib import Path

import numpy as np
import pytest
import torch
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, SubstationDataset


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
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest) -> SubstationDataset:
        """
        Fixture to create a mock dataset with specified parameters.
        """
        class Args:
            pass

        args = Args()
        args.data_dir = tmp_path
        args.in_channels = 4
        args.use_timepoints = False
        args.normalizing_type = 'zscore'
        args.normalizing_factor = np.array([1.0])
        args.means = np.array([0.5])
        args.stds = np.array([0.1])
        args.mask_2d = True
        args.model_type = 'segmentation'

        # Creating mock image and mask files
        for filename in request.param['image_files']:
            os.makedirs(os.path.join(tmp_path, 'image_stack'), exist_ok=True)
            os.makedirs(os.path.join(tmp_path, 'mask'), exist_ok=True)
            np.savez_compressed(os.path.join(tmp_path, 'image_stack', filename), arr_0=np.random.rand(4, 128, 128))
            np.savez_compressed(os.path.join(tmp_path, 'mask', filename), arr_0=np.random.randint(0, 4, (128, 128)))

        image_files = request.param['image_files']
        geo_transforms = request.param['geo_transforms']
        color_transforms = request.param['color_transforms']
        image_resize = request.param['image_resize']
        mask_resize = request.param['mask_resize']

        return SubstationDataset(
            args,
            image_files=image_files,
            geo_transforms=geo_transforms,
            color_transforms=color_transforms,
            image_resize=image_resize,
            mask_resize=mask_resize,
        )

    def test_getitem(self, dataset: SubstationDataset) -> None:
        image, mask = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert image.shape[0] == 4  # Checking number of channels
        assert mask.shape == (1, 128, 128)

    def test_len(self, dataset: SubstationDataset) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, tmp_path: Path) -> None:
        # Test to ensure dataset initialization doesn't download if data already exists
        class Args:
            pass

        args = Args()
        args.data_dir = tmp_path
        args.in_channels = 4
        args.use_timepoints = False
        args.normalizing_type = 'zscore'
        args.normalizing_factor = np.array([1.0])
        args.means = np.array([0.5])
        args.stds = np.array([0.1])
        args.mask_2d = True
        args.model_type = 'segmentation'

        os.makedirs(os.path.join(tmp_path, 'image_stack'))
        os.makedirs(os.path.join(tmp_path, 'mask'))

        # No need to assign `dataset` variable, just assert
        SubstationDataset(args, image_files=[])
        assert os.path.exists(os.path.join(tmp_path, 'image_stack'))
        assert os.path.exists(os.path.join(tmp_path, 'mask'))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        class Args:
            pass

        args = Args()
        args.data_dir = tmp_path
        args.in_channels = 4
        args.use_timepoints = False
        args.normalizing_type = 'zscore'
        args.normalizing_factor = np.array([1.0])
        args.means = np.array([0.5])
        args.stds = np.array([0.1])
        args.mask_2d = True
        args.model_type = 'segmentation'

        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SubstationDataset(args, image_files=[])

    def test_plot(self, dataset: SubstationDataset) -> None:
        dataset.plot()
        # No assertion, just ensuring that the plotting does not throw any exceptions.

    def test_corrupted(self, tmp_path: Path) -> None:
        class Args:
            pass

        args = Args()
        args.data_dir = tmp_path
        args.in_channels = 4
        args.use_timepoints = False
        args.normalizing_type = 'zscore'
        args.normalizing_factor = np.array([1.0])
        args.means = np.array([0.5])
        args.stds = np.array([0.1])
        args.mask_2d = True
        args.model_type = 'segmentation'

        # Creating corrupted files
        os.makedirs(os.path.join(tmp_path, 'image_stack'))
        os.makedirs(os.path.join(tmp_path, 'mask'))
        with open(os.path.join(tmp_path, 'image_stack', 'image_1.npz'), 'w') as f:
            f.write('corrupted')

        with pytest.raises(Exception):
            SubstationDataset(args, image_files=['image_1.npz'])
