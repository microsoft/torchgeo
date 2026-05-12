# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import shutil
import zipfile
from itertools import product
from pathlib import Path
from typing import Literal, TypedDict

import geopandas as gpd
import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import FLAIRHUB, DatasetNotFoundError, FLAIRHUBToy

FLAIRHUB_TEST_DATA_DIR = Path('tests') / 'data' / 'flair'
FLAIRHUB_DOMAIN_YEARS = {'D006': ['2020'], 'D012': ['2019'], 'D032': ['2019']}
FLAIRHUB_DOMAIN_YEARS_SINGLE = {'D006': ['2020']}


class _FLAIRHUBKwargs(TypedDict):
    bands: list[str]
    dataset_type: Literal['land_cover', 'crop_type', 'crop_type_2', 'crop_type_3']
    domain_years: dict[str, list[str]]


class _FLAIRHUBToyKwargs(TypedDict):
    bands: list[str]
    dataset_type: Literal['land_cover', 'crop_type']


_FLAIRHUB_KWARGS: _FLAIRHUBKwargs = {
    'bands': ['AERIAL_RGBI'],
    'dataset_type': 'land_cover',
    'domain_years': FLAIRHUB_DOMAIN_YEARS_SINGLE,
}
_FLAIRHUBTOY_KWARGS: _FLAIRHUBToyKwargs = {
    'bands': ['AERIAL_RGBI'],
    'dataset_type': 'land_cover',
}


class TestFLAIRHUB:
    @pytest.fixture(
        params=list(
            product(
                [FLAIRHUB, FLAIRHUBToy],
                [
                    (['AERIAL_RGBI', 'SENTINEL2_TS'], t)
                    for t in ['land_cover', 'crop_type', 'crop_type_2', 'crop_type_3']
                ],
            )
        )
        + [(cls, (['AERIAL_RGBI'], 'land_cover')) for cls in [FLAIRHUB, FLAIRHUBToy]]
        + [(cls, (None, 'land_cover')) for cls in [FLAIRHUB, FLAIRHUBToy]]
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> FLAIRHUB | FLAIRHUBToy:
        base_class: type[FLAIRHUB] | type[FLAIRHUBToy] = request.param[0]
        bands, dataset_type = request.param[1]
        if bands is None:
            monkeypatch.setattr(
                base_class, 'available_bands', ['AERIAL_RGBI', 'SENTINEL2_TS']
            )
        transforms = nn.Identity()

        if base_class is FLAIRHUB:
            monkeypatch.setattr(FLAIRHUB, 'download_link', str(FLAIRHUB_TEST_DATA_DIR))
            return FLAIRHUB(
                root=tmp_path,
                transforms=transforms,
                download=True,
                bands=bands,
                dataset_type=dataset_type,
                domain_years=FLAIRHUB_DOMAIN_YEARS,
            )
        else:
            toy_zip_path = FLAIRHUB_TEST_DATA_DIR / 'FLAIR-HUB_TOY_DATASET.zip'
            monkeypatch.setattr(FLAIRHUBToy, 'download_link', str(toy_zip_path))

            return FLAIRHUBToy(
                root=tmp_path,
                transforms=transforms,
                download=True,
                bands=bands,
                dataset_type=dataset_type,
            )

    def test_len(self, dataset: FLAIRHUB | FLAIRHUBToy) -> None:
        assert len(dataset) == 3

    def test_add(self, dataset: FLAIRHUB | FLAIRHUBToy) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2 * len(dataset)

    def test_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='Invalid band names'):
            FLAIRHUB(root=tmp_path, bands=['invalid_band'])
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FLAIRHUB(root=tmp_path, bands=['AERIAL_RGBI'], download=False)
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FLAIRHUB(root=tmp_path, download=False, **_FLAIRHUB_KWARGS)
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FLAIRHUBToy(root=tmp_path, download=False, **_FLAIRHUBTOY_KWARGS)
        with pytest.raises(AssertionError, match='split must be one of'):
            FLAIRHUB(root=tmp_path, split='invalid', bands=['AERIAL_RGBI'])
        with pytest.raises(AssertionError, match='split_column must be one of'):
            FLAIRHUB(
                root=tmp_path,
                split='train',
                split_column='invalid',
                bands=['AERIAL_RGBI'],
            )

    @pytest.mark.parametrize(
        'dataset_type,bands,suptitle',
        [
            ('land_cover', None, 'All modalities'),
            ('crop_type', ['AERIAL_RGBI'], 'LPIS 1 (crop_type)'),
            ('crop_type_2', ['AERIAL_RGBI'], 'LPIS 2 (crop_type_2)'),
            ('crop_type_3', ['AERIAL_RGBI'], 'LPIS 3 (crop_type_3)'),
        ],
    )
    def test_plot_all_modalities_and_lpis(
        self,
        tmp_path: Path,
        dataset_type: Literal['land_cover', 'crop_type', 'crop_type_2', 'crop_type_3'],
        bands: list[str] | None,
        suptitle: str,
    ) -> None:
        root = tmp_path / 'flair'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root)
        dataset = FLAIRHUB(
            root=root,
            download=False,
            bands=bands,
            dataset_type=dataset_type,
            domain_years=FLAIRHUB_DOMAIN_YEARS_SINGLE,
        )
        x = dataset[0]
        fig = dataset.plot(x, suptitle=suptitle)
        assert fig is not None
        plt.close()

    def test_getitem_with_transforms(self, tmp_path: Path) -> None:
        root = tmp_path / 'flair'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root)

        def custom_transform(
            sample: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            for key in sample:
                if key != 'mask':
                    sample[key] = sample[key] * 2.0
            return sample

        transformed_dataset = FLAIRHUB(
            root=root, transforms=custom_transform, download=False, **_FLAIRHUB_KWARGS
        )
        no_transform_dataset = FLAIRHUB(
            root=root, transforms=nn.Identity(), download=False, **_FLAIRHUB_KWARGS
        )

        x = no_transform_dataset[0]
        x_transformed = transformed_dataset[0]
        assert (
            abs(x_transformed['AERIAL_RGBI'].max() - x['AERIAL_RGBI'].max() * 2.0)
            < 1e-6
        )


class TestFLAIRHUBSpecific:
    def test_split_filters_and_default_loads_all(self, tmp_path: Path) -> None:
        """With split_column set, only that split is loaded; with default, all patches."""
        root = tmp_path / 'flair'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root)

        filtered = FLAIRHUB(
            root=root,
            download=False,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            split='train',
            split_column='split_1',
            domain_years=FLAIRHUB_DOMAIN_YEARS,
        )
        assert len(filtered) == 1

        all_patches = FLAIRHUB(
            root=root,
            download=False,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            domain_years=FLAIRHUB_DOMAIN_YEARS,
        )
        assert len(all_patches) == 3

    def test_zip_exists_but_not_extracted(self, tmp_path: Path) -> None:
        root = tmp_path / 'flair'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root)
        modality_dir = root / 'D006-2020_AERIAL_RGBI'
        modality_zip = root / 'D006-2020_AERIAL_RGBI.zip'
        shutil.rmtree(modality_dir)
        dataset = FLAIRHUB(
            root=root, transforms=nn.Identity(), download=False, **_FLAIRHUB_KWARGS
        )
        assert modality_dir.is_dir()
        assert not modality_zip.exists()
        assert dataset is not None

    def test_extract_file_not_found(self, tmp_path: Path) -> None:
        root = tmp_path / 'flair'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root)
        dataset = FLAIRHUB(root=root, download=False, **_FLAIRHUB_KWARGS)
        (root / 'D006-2020_AERIAL_RGBI.zip').unlink()
        with pytest.raises(FileNotFoundError, match='Archive not found'):
            dataset._extract('D006', '2020', 'AERIAL_RGBI')

    def test_ensure_splits_download(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """_ensure_splits_available triggers download when gpkg and zip missing."""
        root = tmp_path / 'flair'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root)
        shutil.rmtree(root / 'GLOBAL_ALL_MTD')
        zip_path = root / 'GLOBAL_ALL_MTD.zip'
        zip_src = FLAIRHUB_TEST_DATA_DIR / 'GLOBAL_ALL_MTD.zip'
        zip_path.unlink()

        def mock_download(url: str, root: Path) -> None:
            shutil.copy(zip_src, Path(root) / 'GLOBAL_ALL_MTD.zip')

        monkeypatch.setattr('torchgeo.datasets.flair.download_url', mock_download)
        dataset = FLAIRHUB(root=root, download=True, **_FLAIRHUB_KWARGS)
        path = dataset._ensure_splits_available()
        assert path.exists()


class TestFLAIRHUBToySpecific:
    def test_split_filters_and_default_loads_all(self, tmp_path: Path) -> None:
        """With split_column set, only that split is loaded; with default, all patches."""
        root = tmp_path
        shutil.copytree(
            FLAIRHUB_TEST_DATA_DIR / 'FLAIR-HUB_TOY', root / 'FLAIR-HUB_TOY'
        )

        filtered = FLAIRHUBToy(
            root=root,
            download=False,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            split='train',
            split_column='split_toy',
        )
        assert len(filtered) == 1

        all_patches = FLAIRHUBToy(
            root=root, download=False, bands=['AERIAL_RGBI'], dataset_type='land_cover'
        )
        assert len(all_patches) == 3

    def test_already_extracted(self, tmp_path: Path) -> None:
        toy_dir = tmp_path / 'FLAIR-HUB_TOY'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, toy_dir)
        FLAIRHUBToy(root=tmp_path, download=False, **_FLAIRHUBTOY_KWARGS)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        """Dataset initializes when zip is present and extracts with download=False."""
        toy_dir = tmp_path / 'FLAIR-HUB_TOY'
        toy_zip = tmp_path / 'FLAIR-HUB_TOY_DATASET.zip'
        with zipfile.ZipFile(toy_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in FLAIRHUB_TEST_DATA_DIR.rglob('*.tif'):
                arcname = Path('FLAIR-HUB_TOY') / file_path.relative_to(
                    FLAIRHUB_TEST_DATA_DIR
                )
                zipf.write(file_path, arcname)
        dataset = FLAIRHUBToy(root=tmp_path, download=False, **_FLAIRHUBTOY_KWARGS)
        assert toy_dir.is_dir()
        assert dataset is not None

    @pytest.mark.parametrize('splits_present', [True, False])
    def test_ensure_splits_available(
        self, tmp_path: Path, splits_present: bool
    ) -> None:
        """Splits present → returns path; splits missing and no download → raises."""
        toy_dir = tmp_path / 'FLAIR-HUB_TOY'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, toy_dir)
        if splits_present:
            gpkg_dir = toy_dir / 'GLOBAL_ALL_MTD'
            gdf = gpd.GeoDataFrame(
                {'patch_id': [], 'split_1': []}, geometry=[], crs='EPSG:4326'
            )
            gdf.to_file(gpkg_dir / 'GLOBAL_ALL_MTD_SPLIT.gpkg', driver='GPKG')
        else:
            shutil.rmtree(toy_dir / 'GLOBAL_ALL_MTD')
            (toy_dir / 'GLOBAL_ALL_MTD.zip').unlink()
        dataset = FLAIRHUBToy(root=tmp_path, download=False, **_FLAIRHUBTOY_KWARGS)
        if splits_present:
            path = dataset._ensure_splits_available()
            assert path == toy_dir / 'GLOBAL_ALL_MTD' / 'GLOBAL_ALL_MTD_SPLIT.gpkg'
            assert path.exists()
        else:
            with pytest.raises(DatasetNotFoundError):
                dataset._ensure_splits_available()
