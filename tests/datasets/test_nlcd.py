# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import (
    NLCD,
    DatasetNotFoundError,
    IntersectionDataset,
    UnionDataset,
)


class TestNLCD:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> NLCD:
        # Create zip files in test data directory for download_url mock to copy
        test_data_dir = os.path.join('tests', 'data', 'nlcd')
        for year in [2011, 2019]:
            tif_pathname = os.path.join(
                test_data_dir, f'Annual_NLCD_LndCov_{year}_CU_C1V1.tif'
            )
            zip_pathname = os.path.join(
                test_data_dir, f'Annual_NLCD_LndCov_{year}_CU_C1V1.zip'
            )
            if not os.path.exists(zip_pathname):
                with zipfile.ZipFile(zip_pathname, 'w') as zf:
                    zf.write(tif_pathname, f'Annual_NLCD_LndCov_{year}_CU_C1V1.tif')

        # Precalculated MD5 checksums of the zip files
        md5s = {
            2011: 'dadcb8af5b9eff117b9bb00b648594b4',
            2019: '6196bf5ae9fdd5858aaf71b76d388806',
        }
        monkeypatch.setattr(NLCD, 'md5s', md5s)

        url = os.path.join('tests', 'data', 'nlcd', 'Annual_NLCD_LndCov_{}_CU_C1V1.zip')
        monkeypatch.setattr(NLCD, 'url', url)
        monkeypatch.setattr(plt, 'show', lambda *args: None)
        root = tmp_path
        transforms = nn.Identity()
        return NLCD(
            root,
            transforms=transforms,
            download=True,
            checksum=True,
            years=[2011, 2019],
        )

    def test_getitem(self, dataset: NLCD) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: NLCD) -> None:
        assert len(dataset) == 2

    def test_classes(self) -> None:
        root = os.path.join('tests', 'data', 'nlcd')
        classes = list(NLCD.cmap.keys())[:5]
        ds = NLCD(root, years=[2019], classes=classes)
        sample = ds[ds.bounds]
        mask = sample['mask']
        assert mask.max() < len(classes)

    def test_and(self, dataset: NLCD) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: NLCD) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: NLCD) -> None:
        NLCD(dataset.paths, download=True, years=[2019])

    def test_already_downloaded(self, tmp_path: Path) -> None:
        # Create a zip file containing the tif file
        tif_pathname = os.path.join(
            'tests', 'data', 'nlcd', 'Annual_NLCD_LndCov_2019_CU_C1V1.tif'
        )
        zip_pathname = os.path.join(tmp_path, 'Annual_NLCD_LndCov_2019_CU_C1V1.zip')
        # The zip should contain a file with the C1V1 pattern expected by the code
        with zipfile.ZipFile(zip_pathname, 'w') as zf:
            zf.write(tif_pathname, 'Annual_NLCD_LndCov_2019_CU_C1V1.tif')
        NLCD(tmp_path, years=[2019])

    def test_invalid_year(self, tmp_path: Path) -> None:
        with pytest.raises(
            AssertionError,
            match='NLCD data product only exists for the following years:',
        ):
            NLCD(tmp_path, years=[1984])

    def test_invalid_classes(self) -> None:
        with pytest.raises(AssertionError):
            NLCD(classes=[-1])

        with pytest.raises(AssertionError):
            NLCD(classes=[11])

    def test_plot(self, dataset: NLCD) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: NLCD) -> None:
        query = dataset.bounds
        x = dataset[query]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            NLCD(tmp_path)

    def test_not_downloaded_explicit(self, tmp_path: Path) -> None:
        """Test that DatasetNotFoundError is raised when download=False and files don't exist."""
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            NLCD(tmp_path, download=False, years=[2019])

    def test_invalid_query(self, dataset: NLCD) -> None:
        with pytest.raises(
            IndexError, match=r'query: .* not found in index with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_invalid_checksum(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        # Create zip files in test data directory for download_url mock to copy
        test_data_dir = os.path.join('tests', 'data', 'nlcd')
        tif_pathname = os.path.join(
            test_data_dir, 'Annual_NLCD_LndCov_2019_CU_C1V1.tif'
        )
        zip_pathname = os.path.join(
            test_data_dir, 'Annual_NLCD_LndCov_2019_CU_C1V1.zip'
        )
        if not os.path.exists(zip_pathname):
            with zipfile.ZipFile(zip_pathname, 'w') as zf:
                zf.write(tif_pathname, 'Annual_NLCD_LndCov_2019_CU_C1V1.tif')

        # Set incorrect MD5 checksum
        md5s = {2019: '00000000000000000000000000000000'}
        monkeypatch.setattr(NLCD, 'md5s', md5s)
        url = os.path.join('tests', 'data', 'nlcd', 'Annual_NLCD_LndCov_{}_CU_C1V1.zip')
        monkeypatch.setattr(NLCD, 'url', url)

        with pytest.raises(RuntimeError, match='MD5 checksum mismatch'):
            NLCD(tmp_path, download=True, checksum=True, years=[2019])

    def test_invalid_checksum_already_downloaded(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        # Create zip file in tmp_path to simulate already downloaded file
        test_data_dir = os.path.join('tests', 'data', 'nlcd')
        tif_pathname = os.path.join(
            test_data_dir, 'Annual_NLCD_LndCov_2019_CU_C1V1.tif'
        )
        zip_pathname = os.path.join(tmp_path, 'Annual_NLCD_LndCov_2019_CU_C1V1.zip')
        with zipfile.ZipFile(zip_pathname, 'w') as zf:
            zf.write(tif_pathname, 'Annual_NLCD_LndCov_2019_CU_C1V1.tif')

        # Set incorrect MD5 checksum
        md5s = {2019: '00000000000000000000000000000000'}
        monkeypatch.setattr(NLCD, 'md5s', md5s)

        with pytest.raises(RuntimeError, match='MD5 checksum mismatch'):
            NLCD(tmp_path, download=False, checksum=True, years=[2019])

    def test_validate_checksums_year_not_in_md5s(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that validation skips when year is not in md5s."""
        test_data_dir = os.path.join('tests', 'data', 'nlcd')
        tif_pathname = os.path.join(
            test_data_dir, 'Annual_NLCD_LndCov_2019_CU_C1V1.tif'
        )
        zip_pathname = os.path.join(tmp_path, 'Annual_NLCD_LndCov_2019_CU_C1V1.zip')
        with zipfile.ZipFile(zip_pathname, 'w') as zf:
            zf.write(tif_pathname, 'Annual_NLCD_LndCov_2019_CU_C1V1.tif')

        dataset = NLCD(tmp_path, download=False, checksum=False, years=[2019])

        monkeypatch.setattr(NLCD, 'md5s', {})
        # Should not raise an error, validation should skip 2019
        # this enable user to potentially add future years without having to validate checksums for those years.
        dataset._validate_checksums()

    def test_validate_checksums_valid_checksum(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that validation passes with valid checksum."""
        test_data_dir = os.path.join('tests', 'data', 'nlcd')
        # Copy the existing zip file from test data directory to preserve MD5
        dest_zip = os.path.join(tmp_path, 'Annual_NLCD_LndCov_2019_CU_C1V1.zip')

        tif_pathname = os.path.join(
            test_data_dir, 'Annual_NLCD_LndCov_2019_CU_C1V1.tif'
        )
        with zipfile.ZipFile(dest_zip, 'w') as zf:
            zf.write(tif_pathname, 'Annual_NLCD_LndCov_2019_CU_C1V1.tif')
        # Calculate MD5 of created zip file
        with open(dest_zip, 'rb') as f:
            actual_md5 = hashlib.md5(f.read()).hexdigest()
        md5s = {2019: actual_md5}

        monkeypatch.setattr(NLCD, 'md5s', md5s)

        dataset = NLCD(tmp_path, download=False, checksum=True, years=[2019])
        assert dataset is not None
