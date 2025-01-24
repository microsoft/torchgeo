# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import BigEarthNetV2, DatasetNotFoundError


class TestBigEarthNetV2:
    @pytest.fixture(
        params=zip(['all', 's1', 's2'], [19, 19, 19], ['train', 'val', 'test'])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> BigEarthNetV2:
        data_dir = os.path.join('tests', 'data', 'bigearthnetv2')
        metadata = {
            's1': {
                'url': os.path.join(data_dir, 'BigEarthNet-S1.tar.zst'),
                'md5': 'a55eaa2cdf6a917e296bd6601ec1e348',
                'filename': 'BigEarthNet-S1.tar.zst',
                'directory': 'BigEarthNet-S1',
            },
            's2': {
                'url': os.path.join(data_dir, 'BigEarthNet-S2.tar.zst'),
                'md5': '2245ed2d1a93f6ce637d839bc856396e',
                'filename': 'BigEarthNet-S2.tar.zst',
                'directory': 'BigEarthNet-S2',
            },
            'maps': {
                'url': os.path.join(data_dir, 'Reference_Maps.tar.zst'),
                'md5': '95d85a222fa983faddcac51a19f28917',
                'filename': 'Reference_Maps.tar.zst',
                'directory': 'Reference_Maps',
            },
        }
        monkeypatch.setattr(BigEarthNetV2, 'metadata_locs', metadata)

        # Create dummy metadata.parquet
        dummy_metadata = {
            's2_base': 'S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP',
            'patch_id': 'S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57',
            's1_name': 'S1A_IW_GRDH_1SDV_20170613T165043_33UUP_61_39',
            's1_base': 'S1A_IW_GRDH_1SDV_20170613T165043',
            'split': request.param[2],
            'labels': ['Urban fabric', 'Industrial or commercial units'],
        }

        bands, num_classes, split = request.param
        root = tmp_path
        transforms = nn.Identity()
        return BigEarthNetV2(
            root, split, bands, num_classes, transforms, download=True, checksum=True
        )

    def test_getitem_s2(self, dataset: BigEarthNetV2) -> None:
        """Test loading S2 or combined data."""
        x = dataset[0]

        if dataset.bands in ['s2', 'all']:
            if dataset.bands == 's2':
                assert x['image'].shape == (12, 120, 120)
            else:
                assert x['image_s2'].shape == (12, 120, 120)

        if dataset.bands in ['s1', 'all']:
            if dataset.bands == 's1':
                assert x['image'].shape == (2, 120, 120)
            else:
                assert x['image_s1'].shape == (2, 120, 120)

        assert x['mask'].shape == (1, 120, 120)
        assert x['label'].shape == (dataset.num_classes,)

        assert x['mask'].dtype == torch.int64
        assert x['label'].dtype == torch.int64
        if 'image' in x:
            assert x['image'].dtype == torch.float32
        if 'image_s1' in x:
            assert x['image_s1'].dtype == torch.float32
        if 'image_s2' in x:
            assert x['image_s2'].dtype == torch.float32

    def test_len(self, dataset: BigEarthNetV2) -> None:
        """Test dataset length."""
        if dataset.split == 'train':
            assert len(dataset) == 2
        elif dataset.split == 'val':
            assert len(dataset) == 1
        else:
            assert len(dataset) == 1

    def test_not_downloaded(self, tmp_path: Path) -> None:
        """Test error handling when data not present."""
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            BigEarthNetV2(tmp_path)

    def test_invalid_split(self, tmp_path: Path) -> None:
        """Test error on invalid split."""
        with pytest.raises(AssertionError, match='split must be one of'):
            BigEarthNetV2(tmp_path, split='invalid')

    def test_invalid_bands(self, tmp_path: Path) -> None:
        """Test error on invalid bands selection."""
        with pytest.raises(AssertionError):
            BigEarthNetV2(tmp_path, bands='invalid')

    def test_invalid_num_classes(self, tmp_path: Path) -> None:
        """Test error on invalid number of classes."""
        with pytest.raises(AssertionError):
            BigEarthNetV2(tmp_path, num_classes=20)

    def test_plot(self, dataset: BigEarthNetV2) -> None:
        """Test plotting functionality."""
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()

        # Test without titles
        dataset.plot(x, show_titles=False)
        plt.close()

        # Test with prediction
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()
