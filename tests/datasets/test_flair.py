# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import shutil
import zipfile
from itertools import product
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import FLAIRHUB, DatasetNotFoundError, FLAIRHUBToy

FLAIRHUB_TEST_DATA_DIR = Path('tests') / 'data' / 'flairhub'
FLAIRHUB_DOMAIN_YEARS = {'D006': ['2020'], 'D012': ['2019'], 'D032': ['2019']}
FLAIRHUB_DOMAIN_YEARS_SINGLE = {'D006': ['2020']}


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

    def test_getitem(self, dataset: FLAIRHUB | FLAIRHUBToy) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert 'mask' in x
        assert isinstance(x['mask'], torch.Tensor)
        assert x['mask'].dtype == torch.int64

        if 'AERIAL_RGBI' in dataset.bands:
            assert x['AERIAL_RGBI'].shape[0] == 4
        if 'SENTINEL2_TS' in dataset.bands:
            assert x['SENTINEL2_TS'].shape == (2, 10, 10, 10)

    def test_len(self, dataset: FLAIRHUB | FLAIRHUBToy) -> None:
        assert len(dataset) > 0

    def test_add(self, dataset: FLAIRHUB | FLAIRHUBToy) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2 * len(dataset)

    def test_invalid_band_name(self, tmp_path: Path) -> None:
        """Test that ValueError is raised when invalid band names are provided."""
        with pytest.raises(ValueError, match='Invalid band names'):
            FLAIRHUB(root=tmp_path, bands=['invalid_band'])

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FLAIRHUB(root=tmp_path, bands=['AERIAL_RGBI'], download=False)

    def test_invalid_split_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='split must be one of'):
            FLAIRHUB(root=tmp_path, split='invalid', bands=['AERIAL_RGBI'])

    def test_invalid_split_column_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='split_column must be one of'):
            FLAIRHUB(
                root=tmp_path,
                split='train',
                split_column='invalid',
                bands=['AERIAL_RGBI'],
            )

    @pytest.mark.parametrize('cls', [FLAIRHUB, FLAIRHUBToy])
    def test_split_filters_and_default_loads_all(
        self, tmp_path: Path, cls: type[FLAIRHUB] | type[FLAIRHUBToy]
    ) -> None:
        """With split_column set, only that split is loaded; with default, all patches."""
        if cls is FLAIRHUBToy:
            root = tmp_path
            shutil.copytree(
                FLAIRHUB_TEST_DATA_DIR / 'FLAIR-HUB_TOY',
                root / 'FLAIR-HUB_TOY',
                dirs_exist_ok=True,
            )
        else:
            root = tmp_path / 'flairhub'
            shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root, dirs_exist_ok=True)
        kwargs: dict = {'domain_years': FLAIRHUB_DOMAIN_YEARS}

        split_column = 'split_toy' if cls is FLAIRHUBToy else 'split_1'
        common = {
            'root': root,
            'download': False,
            'bands': ['AERIAL_RGBI'],
            'dataset_type': 'land_cover',
        }
        extra = kwargs if cls is FLAIRHUB else {}
        filtered = cls(**common, split='train', split_column=split_column, **extra)
        assert len(filtered) == 1
        all_patches = cls(**common, **extra)
        assert len(all_patches) == 3

    @pytest.mark.parametrize(
        'dataset_type,bands,suptitle',
        [
            ('land_cover', None, 'All modalities'),
            ('crop_type_2', ['AERIAL_RGBI'], 'LPIS 2 (crop_type_2)'),
            ('crop_type_3', ['AERIAL_RGBI'], 'LPIS 3 (crop_type_3)'),
        ],
    )
    def test_plot_all_modalities_and_lpis(
        self, tmp_path: Path, dataset_type: str, bands: list[str] | None, suptitle: str
    ) -> None:
        root = tmp_path / 'flairhub'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root, dirs_exist_ok=True)
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


class TestFLAIRHUBSpecific:
    def test_zip_exists_but_not_extracted(self, tmp_path: Path) -> None:
        root = tmp_path / 'flairhub'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root, dirs_exist_ok=True)
        modality_dir = root / 'D006-2020_AERIAL_RGBI'
        modality_zip = root / 'D006-2020_AERIAL_RGBI.zip'
        shutil.rmtree(modality_dir)
        dataset = FLAIRHUB(
            root=root,
            transforms=nn.Identity(),
            download=False,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            domain_years=FLAIRHUB_DOMAIN_YEARS_SINGLE,
        )

        assert modality_dir.is_dir()
        assert not modality_zip.exists()
        assert dataset is not None

    def test_getitem_with_transforms(self, tmp_path: Path) -> None:
        root = tmp_path / 'flairhub'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root, dirs_exist_ok=True)

        def custom_transform(
            sample: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            for key in sample:
                if key != 'mask':
                    sample[key] = sample[key] * 2.0
            return sample

        transformed_dataset = FLAIRHUB(
            root=root,
            transforms=custom_transform,
            download=False,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            domain_years=FLAIRHUB_DOMAIN_YEARS_SINGLE,
        )
        no_transform_dataset = FLAIRHUB(
            root=root,
            transforms=nn.Identity(),
            download=False,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            domain_years=FLAIRHUB_DOMAIN_YEARS_SINGLE,
        )

        x = no_transform_dataset[0]
        x_transformed = transformed_dataset[0]
        assert (
            abs(x_transformed['AERIAL_RGBI'].max() - x['AERIAL_RGBI'].max() * 2.0)
            < 1e-6
        )

    def test_extract_file_not_found(self, tmp_path: Path) -> None:
        root = tmp_path / 'flairhub'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root, dirs_exist_ok=True)
        dataset = FLAIRHUB(
            root=root,
            download=False,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            domain_years=FLAIRHUB_DOMAIN_YEARS_SINGLE,
        )

        (root / 'D006-2020_AERIAL_RGBI.zip').unlink()

        with pytest.raises(FileNotFoundError, match='Archive not found'):
            dataset._extract('D006', '2020', 'AERIAL_RGBI')

    def test_ensure_splits_download(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """_ensure_splits_available triggers download when gpkg and zip missing."""
        root = tmp_path / 'flairhub'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, root, dirs_exist_ok=True)
        gpkg_dir = root / 'GLOBAL_ALL_MTD'
        if gpkg_dir.exists():
            shutil.rmtree(gpkg_dir)
        zip_path = root / 'GLOBAL_ALL_MTD.zip'
        zip_src = FLAIRHUB_TEST_DATA_DIR / 'GLOBAL_ALL_MTD.zip'
        zip_path.unlink(missing_ok=True)

        def mock_download(url: str, root: Path) -> None:
            shutil.copy(zip_src, Path(root) / 'GLOBAL_ALL_MTD.zip')

        monkeypatch.setattr('torchgeo.datasets.flair.download_url', mock_download)
        dataset = FLAIRHUB(
            root=root,
            download=True,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            domain_years=FLAIRHUB_DOMAIN_YEARS_SINGLE,
        )
        path = dataset._ensure_splits_available()
        assert path.exists()


class TestFLAIRHUBToySpecific:
    def test_already_extracted(self, tmp_path: Path) -> None:
        toy_dir = tmp_path / 'FLAIR-HUB_TOY'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, toy_dir, dirs_exist_ok=True)

        FLAIRHUBToy(
            root=tmp_path,
            download=False,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
        )

    def test_already_downloaded(self, tmp_path: Path) -> None:
        """Test that dataset can be initialized when zip is already downloaded."""
        toy_dir = tmp_path / 'FLAIR-HUB_TOY'
        toy_zip = tmp_path / 'FLAIR-HUB_TOY_DATASET.zip'

        with zipfile.ZipFile(toy_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in FLAIRHUB_TEST_DATA_DIR.rglob('*.tif'):
                arcname = Path('FLAIR-HUB_TOY') / file_path.relative_to(
                    FLAIRHUB_TEST_DATA_DIR
                )
                zipf.write(file_path, arcname)

        if toy_dir.exists():
            shutil.rmtree(toy_dir)

        dataset = FLAIRHUBToy(
            root=tmp_path,
            download=False,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
        )
        assert toy_dir.is_dir()
        assert dataset is not None

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FLAIRHUBToy(
                root=tmp_path,
                download=False,
                bands=['AERIAL_RGBI'],
                dataset_type='land_cover',
            )

    @pytest.mark.parametrize('splits_present', [True, False])
    def test_ensure_splits_available(
        self, tmp_path: Path, splits_present: bool
    ) -> None:
        """Splits present → returns path; splits missing and no download → raises."""
        toy_dir = tmp_path / 'FLAIR-HUB_TOY'
        shutil.copytree(FLAIRHUB_TEST_DATA_DIR, toy_dir, dirs_exist_ok=True)
        if splits_present:
            gpkg_dir = toy_dir / 'GLOBAL_ALL_MTD'
            gpkg_dir.mkdir(exist_ok=True)
            gdf = gpd.GeoDataFrame(
                {'patch_id': [], 'split_1': []}, geometry=[], crs='EPSG:4326'
            )
            gdf.to_file(gpkg_dir / 'GLOBAL_ALL_MTD_SPLIT.gpkg', driver='GPKG')
        else:
            shutil.rmtree(toy_dir / 'GLOBAL_ALL_MTD', ignore_errors=True)
            (toy_dir / 'GLOBAL_ALL_MTD.zip').unlink(missing_ok=True)

        dataset = FLAIRHUBToy(
            root=tmp_path,
            download=False,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
        )
        if splits_present:
            path = dataset._ensure_splits_available()
            assert path == toy_dir / 'GLOBAL_ALL_MTD' / 'GLOBAL_ALL_MTD_SPLIT.gpkg'
            assert path.exists()
        else:
            with pytest.raises(DatasetNotFoundError):
                dataset._ensure_splits_available()
