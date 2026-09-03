# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn

from torchgeo.datasets import Landslide4Sense, RGBBandsMissingError

h5py = pytest.importorskip('h5py', minversion='3.10')


class TestLandslide4Sense:
    @pytest.fixture
    def dataset(self) -> Landslide4Sense:
        root = os.path.join('tests', 'data', 'landslide4sense')
        transforms = nn.Identity()
        return Landslide4Sense(root=root, split='train', transforms=transforms)

    def test_getitem(self, dataset: Landslide4Sense) -> None:
        sample = dataset[0]

        assert isinstance(sample, dict)
        assert isinstance(sample['image'], torch.Tensor)
        assert isinstance(sample['mask'], torch.Tensor)

        assert sample['image'].shape == (14, 32, 32)
        assert sample['mask'].shape == (1, 32, 32)
        assert sample['mask'].dtype == torch.int64

    def test_len_train(self, dataset: Landslide4Sense) -> None:
        assert len(dataset) == 2

    def test_len_val(self) -> None:
        root = os.path.join('tests', 'data', 'landslide4sense')
        dataset = Landslide4Sense(root=root, split='val')
        assert len(dataset) == 1

    def test_len_test(self) -> None:
        root = os.path.join('tests', 'data', 'landslide4sense')
        dataset = Landslide4Sense(root=root, split='test')
        assert len(dataset) == 1

    def test_invalid_split(self, dataset: Landslide4Sense) -> None:
        split = cast(Literal['train', 'val', 'test'], 'oops')
        with pytest.raises(ValueError, match=f"Invalid split '{split}'"):
            Landslide4Sense(root=dataset.root, split=split)

    def test_missing_root(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Landslide4Sense(root=tmp_path / 'does_not_exist', split='train')

    def test_missing_train_mask_dir(self, tmp_path: Path) -> None:
        root = tmp_path / 'landslide4sense'
        (root / 'TrainData' / 'img').mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match='Mask directory not found'):
            Landslide4Sense(root=root, split='train')

    def test_missing_h5_files(self, tmp_path: Path) -> None:
        root = tmp_path / 'landslide4sense'
        (root / 'TrainData' / 'img').mkdir(parents=True)
        (root / 'TrainData' / 'mask').mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match=r'No \.h5 files found'):
            Landslide4Sense(root=root, split='train')

    def test_no_samples_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _mock_load_samples(
            _self: Landslide4Sense,
        ) -> list[tuple[Path, Path | None]]:
            return []

        monkeypatch.setattr(Landslide4Sense, '_load_samples', _mock_load_samples)

        root = os.path.join('tests', 'data', 'landslide4sense')
        dataset = Landslide4Sense(root=root, split='train')
        assert len(dataset) == 0

    def test_val_sample_contains_only_image(self) -> None:
        root = os.path.join('tests', 'data', 'landslide4sense')
        dataset = Landslide4Sense(root=root, split='val')
        sample = dataset[0]

        assert set(sample.keys()) == {'image'}
        assert isinstance(sample['image'], torch.Tensor)
        assert sample['image'].shape == (14, 32, 32)

    def test_test_sample_contains_only_image(self) -> None:
        root = os.path.join('tests', 'data', 'landslide4sense')
        dataset = Landslide4Sense(root=root, split='test')
        sample = dataset[0]

        assert set(sample.keys()) == {'image'}
        assert isinstance(sample['image'], torch.Tensor)
        assert sample['image'].shape == (14, 32, 32)

    def test_band_subset(self, dataset: Landslide4Sense) -> None:
        dataset = Landslide4Sense(root=dataset.root, split='train', bands=[0, 3, 13])
        sample = dataset[0]

        assert sample['image'].shape == (3, 32, 32)
        assert sample['mask'].shape == (1, 32, 32)

    def test_missing_train_mask(self, tmp_path: Path) -> None:
        root = tmp_path / 'landslide4sense'
        train_img = root / 'TrainData' / 'img'
        train_mask = root / 'TrainData' / 'mask'
        val_img = root / 'ValidData' / 'img'
        test_img = root / 'TestData' / 'img'

        train_img.mkdir(parents=True)
        train_mask.mkdir(parents=True)
        val_img.mkdir(parents=True)
        test_img.mkdir(parents=True)

        source_root = Path('tests/data/landslide4sense')

        for src, dst in [
            (
                source_root / 'TrainData' / 'img' / 'image_1.h5',
                train_img / 'image_1.h5',
            ),
            (
                source_root / 'TrainData' / 'img' / 'image_2.h5',
                train_img / 'image_2.h5',
            ),
            (
                source_root / 'TrainData' / 'mask' / 'mask_2.h5',
                train_mask / 'mask_2.h5',
            ),
            (source_root / 'ValidData' / 'img' / 'image_1.h5', val_img / 'image_1.h5'),
            (source_root / 'TestData' / 'img' / 'image_1.h5', test_img / 'image_1.h5'),
        ]:
            dst.write_bytes(src.read_bytes())

        with pytest.raises(FileNotFoundError, match='Missing mask'):
            Landslide4Sense(root=root, split='train')

    def test_plot(self, dataset: Landslide4Sense) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='Prediction')
        plt.close()

    def test_plot_image_only(self) -> None:
        root = os.path.join('tests', 'data', 'landslide4sense')
        dataset = Landslide4Sense(root=root, split='val')
        dataset.plot(dataset[0], suptitle='Image Only')
        plt.close()

    def test_plot_rgb(self, dataset: Landslide4Sense) -> None:
        dataset = Landslide4Sense(root=dataset.root, split='train', bands=[0])
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            dataset.plot(dataset[0], suptitle='Single Band')
