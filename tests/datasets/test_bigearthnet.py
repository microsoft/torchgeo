# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import BigEarthNet, BigEarthNetV2, DatasetNotFoundError

pytest.importorskip('pyarrow')


class TestBigEarthNet:
    @pytest.fixture(
        params=zip(['all', 's1', 's2'], [43, 19, 19], ['train', 'val', 'test'])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> BigEarthNet:
        data_dir = os.path.join('tests', 'data', 'bigearthnet', 'v1')
        metadata = {
            's1': {
                'url': os.path.join(data_dir, 'BigEarthNet-S1-v1.0.tar.gz'),
                'md5': '5a64e9ce38deb036a435a7b59494924c',
                'filename': 'BigEarthNet-S1-v1.0.tar.gz',
                'directory': 'BigEarthNet-S1-v1.0',
            },
            's2': {
                'url': os.path.join(data_dir, 'BigEarthNet-S2-v1.0.tar.gz'),
                'md5': 'ef5f41129b8308ca178b04d7538dbacf',
                'filename': 'BigEarthNet-S2-v1.0.tar.gz',
                'directory': 'BigEarthNet-v1.0',
            },
        }
        splits_metadata = {
            'train': {
                'url': os.path.join(data_dir, 'bigearthnet-train.csv'),
                'filename': 'bigearthnet-train.csv',
                'md5': '167ac4d5de8dde7b5aeaa812f42031e7',
            },
            'val': {
                'url': os.path.join(data_dir, 'bigearthnet-val.csv'),
                'filename': 'bigearthnet-val.csv',
                'md5': 'aff594ba256a52e839a3b5fefeb9ef42',
            },
            'test': {
                'url': os.path.join(data_dir, 'bigearthnet-test.csv'),
                'filename': 'bigearthnet-test.csv',
                'md5': '851a6bdda484d47f60e121352dcb1bf5',
            },
        }
        monkeypatch.setattr(BigEarthNet, 'metadata', metadata)
        monkeypatch.setattr(BigEarthNet, 'splits_metadata', splits_metadata)
        bands, num_classes, split = request.param
        root = tmp_path
        transforms = nn.Identity()
        return BigEarthNet(
            root, split, bands, num_classes, transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: BigEarthNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['label'].shape == (dataset.num_classes,)
        assert x['image'].dtype == torch.float32
        assert x['label'].dtype == torch.int64

        if dataset.bands == 'all':
            assert x['image'].shape == (14, 120, 120)
        elif dataset.bands == 's1':
            assert x['image'].shape == (2, 120, 120)
        else:
            assert x['image'].shape == (12, 120, 120)

    def test_len(self, dataset: BigEarthNet) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 2
        elif dataset.split == 'val':
            assert len(dataset) == 1
        else:
            assert len(dataset) == 1

    def test_already_downloaded(self, dataset: BigEarthNet, tmp_path: Path) -> None:
        BigEarthNet(
            root=tmp_path,
            bands=dataset.bands,
            split=dataset.split,
            num_classes=dataset.num_classes,
            download=True,
        )

    def test_already_downloaded_not_extracted(
        self, dataset: BigEarthNet, tmp_path: Path
    ) -> None:
        if dataset.bands == 'all':
            shutil.rmtree(
                os.path.join(dataset.root, dataset.metadata['s1']['directory'])
            )
            shutil.rmtree(
                os.path.join(dataset.root, dataset.metadata['s2']['directory'])
            )
            shutil.copy(dataset.metadata['s1']['url'], tmp_path)
            shutil.copy(dataset.metadata['s2']['url'], tmp_path)
        elif dataset.bands == 's1':
            shutil.rmtree(
                os.path.join(dataset.root, dataset.metadata['s1']['directory'])
            )
            shutil.copy(dataset.metadata['s1']['url'], tmp_path)
        else:
            shutil.rmtree(
                os.path.join(dataset.root, dataset.metadata['s2']['directory'])
            )
            shutil.copy(dataset.metadata['s2']['url'], tmp_path)

        BigEarthNet(
            root=tmp_path,
            bands=dataset.bands,
            split=dataset.split,
            num_classes=dataset.num_classes,
            download=False,
        )

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            BigEarthNet(tmp_path)

    def test_plot(self, dataset: BigEarthNet) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()


class TestBigEarthNetV2:
    @pytest.fixture(params=zip(['all', 's1', 's2'], ['train', 'val', 'test']))
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> BigEarthNetV2:
        url = os.path.join('tests', 'data', 'bigearthnet', 'v2', '{}')
        monkeypatch.setattr(BigEarthNetV2, 'url', url)
        metadata = {
            's1': {
                'files': {
                    'BigEarthNet-S1.tar.gzaa': '8101e604552c010178af0bf3645cf391',
                    'BigEarthNet-S1.tar.gzab': 'e77d4408fc4594407b7b50ec0d43053c',
                }
            },
            's2': {
                'files': {
                    'BigEarthNet-S2.tar.gzaa': '9c611dc8598b20830d2d79f5a73df294',
                    'BigEarthNet-S2.tar.gzab': '4e6904c7f60504cceaf90a35401e4262',
                }
            },
            'maps': {
                'files': {'Reference_Maps.tar.gzaa': 'dd129c68c4902bfae48d7caada03fdc0'}
            },
            'metadata': {
                'files': {'metadata.parquet': 'ad100d6b020f2e693673f77ebbe57891'}
            },
        }
        monkeypatch.setattr(BigEarthNetV2, 'metadata_locs', metadata)

        bands, split = request.param

        root = tmp_path
        transforms = nn.Identity()
        return BigEarthNetV2(
            root, split, bands, transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: BigEarthNetV2) -> None:
        """Test loading data."""
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
        else:
            assert len(dataset) == 1

    def test_already_downloaded(self, dataset: BigEarthNetV2, tmp_path: Path) -> None:
        BigEarthNetV2(
            root=tmp_path, bands=dataset.bands, split=dataset.split, download=True
        )

    def test_not_downloaded(self, tmp_path: Path) -> None:
        """Test error handling when data not present."""
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            BigEarthNetV2(tmp_path)

    def test_already_downloaded_not_extracted(
        self, dataset: BigEarthNetV2, tmp_path: Path
    ) -> None:
        def rm_files(file_list: list[str]) -> None:
            for key, val in dataset.metadata_locs.items():
                if key in file_list:
                    if key != 'metadata':
                        shutil.rmtree(
                            os.path.join(dataset.root, dataset.dir_file_names[key])
                        )

        if dataset.bands == 'all':
            rm_files(['s1', 's2', 'maps', 'metadata'])
        elif dataset.bands == 's1':
            rm_files(['s1', 'metadata', 'maps'])
        else:
            rm_files(['s2', 'metadata', 'maps'])

        BigEarthNetV2(
            root=tmp_path, bands=dataset.bands, split=dataset.split, download=False
        )

    def test_invalid_split(self, tmp_path: Path) -> None:
        """Test error on invalid split."""
        with pytest.raises(AssertionError, match='split must be one of'):
            BigEarthNetV2(tmp_path, split='invalid')

    def test_invalid_bands(self, tmp_path: Path) -> None:
        """Test error on invalid bands selection."""
        with pytest.raises(AssertionError):
            BigEarthNetV2(tmp_path, bands='invalid')

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
