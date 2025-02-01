# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, DL4GAMAlps, RGBBandsMissingError

pytest.importorskip('xarray', minversion='2023.9')
pytest.importorskip('netCDF4', minversion='1.5.8')


class TestDL4GAMAlps:
    @pytest.fixture(
        params=zip(
            ['train', 'val', 'test'],
            [1, 3, 5],
            ['small', 'small', 'large'],
            [DL4GAMAlps.rgb_bands, DL4GAMAlps.rgb_nir_swir_bands, DL4GAMAlps.all_bands],
            [None, ['dem'], DL4GAMAlps.valid_extra_features],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> DL4GAMAlps:
        r_url = Path('tests', 'data', 'dl4gam_alps')
        download_metadata = {
            'dataset_small': {
                'url': str(r_url / 'dataset_small.tar.gz'),
                'checksum': '35f85360b943caa8661d9fb573b0f0b5',
            },
            'dataset_large': {
                'url': str(r_url / 'dataset_large.tar.gz'),
                'checksum': '636be5be35b8bd1e7771e9010503e4bc',
            },
            'splits_csv': {
                'url': str(r_url / 'splits.csv'),
                'checksum': '973367465c8ab322d0cf544a345b02f5',
            },
        }

        monkeypatch.setattr(DL4GAMAlps, 'download_metadata', download_metadata)
        root = tmp_path
        split, cv_iter, version, bands, extra_features = request.param
        transforms = nn.Identity()
        return DL4GAMAlps(
            root,
            split,
            cv_iter,
            version,
            bands,
            extra_features,
            transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: DL4GAMAlps) -> None:
        x = dataset[0]
        assert isinstance(x, dict)

        var_names = ['image', 'mask_glacier', 'mask_debris', 'mask_clouds_and_shadows']
        if dataset.extra_features:
            var_names += list(dataset.extra_features)
        for v in var_names:
            assert v in x
            assert isinstance(x[v], torch.Tensor)

            # check if all variables have the same spatial dimensions as the image
            assert x['image'].shape[-2:] == x[v].shape[-2:]

        # check the first dimension of the image tensor
        assert x['image'].shape[0] == len(dataset.bands)

    def test_len(self, dataset: DL4GAMAlps) -> None:
        num_glaciers_per_fold = 2 if dataset.split == 'train' else 1
        num_patches_per_glacier = 1 if dataset.version == 'small' else 2
        assert len(dataset) == num_glaciers_per_fold * num_patches_per_glacier

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            DL4GAMAlps(tmp_path)

    def test_already_downloaded_and_extracted(self, dataset: DL4GAMAlps) -> None:
        DL4GAMAlps(root=dataset.root, download=False, version=dataset.version)

    def test_already_downloaded_but_not_yet_extracted(self, tmp_path: Path) -> None:
        fp_archive = Path('tests', 'data', 'dl4gam_alps', 'dataset_small.tar.gz')
        shutil.copyfile(fp_archive, Path(str(tmp_path), fp_archive.name))
        fp_splits = Path('tests', 'data', 'dl4gam_alps', 'splits.csv')
        shutil.copyfile(fp_splits, Path(str(tmp_path), fp_splits.name))
        DL4GAMAlps(root=str(tmp_path), download=False)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            DL4GAMAlps(split='foo')

    def test_plot(self, dataset: DL4GAMAlps) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = torch.clone(sample['mask_glacier'])
        dataset.plot(sample, suptitle='Test with prediction')
        plt.close()

    def test_plot_wrong_bands(self, dataset: DL4GAMAlps) -> None:
        ds = DL4GAMAlps(
            root=dataset.root,
            split=dataset.split,
            cv_iter=dataset.cv_iter,
            version=dataset.version,
            bands=('B3',),
        )
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            ds.plot(dataset[0], suptitle='Single Band')
