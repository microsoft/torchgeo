# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch.nn as nn
from pytest import MonkeyPatch
from torch import Tensor

from torchgeo.datasets import FLAIRHUB, DatasetNotFoundError, FLAIRHUBToy

_TEST_DATA = Path('tests') / 'data' / 'flair'
_DOMAIN_YEARS = {'D006': ['2020'], 'D012': ['2019'], 'D032': ['2019']}


class TestFLAIRHUB:
    def test_not_downloaded(self, tmp_path: Path) -> None:
        """Covered by torchgeo.datasets.errors.DatasetNotFoundError."""
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FLAIRHUB(tmp_path, download=False)
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FLAIRHUBToy(tmp_path, download=False)

    def test_all_modalities_plot(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Every modality loads and renders — covers all plot branches and time series.
        Basic dataset init/getitem/len exercised through the trainer already."""
        root = tmp_path / 'flair'
        shutil.copytree(_TEST_DATA, root)
        monkeypatch.setattr(FLAIRHUB, 'domain_years', {'D006': ['2020']})

        bands = [
            'AERIAL_RGBI',
            'SPOT_RGBI',
            'SENTINEL2_TS',
            'SENTINEL2_MSK-SC',
            'SENTINEL1-ASC_TS',
            'SENTINEL1-DESC_TS',
            'DEM_ELEV',
            'AERIAL-RLT_PAN',
        ]
        ds = FLAIRHUB(root=root, bands=bands, dataset_type='land_cover')
        x = ds[0]

        for band in bands:
            key = ds.modality_key_map[band]
            assert key in x
            assert isinstance(x[key], Tensor)

        fig = ds.plot(x, suptitle='All modalities')
        assert fig is not None
        plt.close()

    def test_crop_type_levels(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """crop_type_2 and crop_type_3 mask loading (deeper LPIS levels).
        Basic crop_type already tested by trainer/flairhub_croptype config."""
        root = tmp_path / 'flair'
        shutil.copytree(_TEST_DATA, root)
        monkeypatch.setattr(FLAIRHUB, 'domain_years', dict(_DOMAIN_YEARS))

        for dtype in ('crop_type_2', 'crop_type_3'):
            ds = FLAIRHUB(root=root, bands=['AERIAL_RGBI'], dataset_type=dtype)  # type: ignore[arg-type]
            x = ds[0]
            assert isinstance(x['mask'], Tensor)
            assert isinstance(x['image_aerial_rgbi'], Tensor)

    def test_zip_reload(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Missing directory is re-extracted from existing zip."""
        root = tmp_path / 'flair'
        shutil.copytree(_TEST_DATA, root)
        monkeypatch.setattr(FLAIRHUB, 'domain_years', dict(_DOMAIN_YEARS))

        dir_to_remove = root / 'D006-2020_AERIAL_RGBI'
        shutil.rmtree(dir_to_remove)
        assert not dir_to_remove.is_dir()

        ds = FLAIRHUB(
            root=root, download=False, bands=['AERIAL_RGBI'], dataset_type='land_cover'
        )
        assert dir_to_remove.is_dir()
        assert not (root / 'D006-2020_AERIAL_RGBI.zip').exists()
        assert ds is not None

    def test_toy_init(self, tmp_path: Path) -> None:
        """FLAIRHUBToy init edge cases not covered by trainer test."""
        shutil.copytree(_TEST_DATA / 'FLAIR-HUB_TOY', tmp_path / 'FLAIR-HUB_TOY')
        ds = FLAIRHUBToy(
            root=tmp_path,
            split='train',
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
        )
        assert len(ds) == 1

        shutil.rmtree(tmp_path / 'FLAIR-HUB_TOY')
        shutil.copy(
            _TEST_DATA / 'FLAIR-HUB_TOY_DATASET.zip',
            tmp_path / 'FLAIR-HUB_TOY_DATASET.zip',
        )
        ds = FLAIRHUBToy(
            root=tmp_path,
            split='train',
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
        )
        assert (tmp_path / 'FLAIR-HUB_TOY').is_dir()
        assert len(ds) == 1

        shutil.rmtree(tmp_path / 'FLAIR-HUB_TOY' / 'GLOBAL_ALL_MTD')
        (tmp_path / 'FLAIR-HUB_TOY' / 'GLOBAL_ALL_MTD.zip').unlink(missing_ok=True)
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FLAIRHUBToy(
                root=tmp_path,
                split='train',
                bands=['AERIAL_RGBI'],
                dataset_type='land_cover',
            )

    def test_download_url_paths(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Cover download_url in _verify, _ensure_splits_available, and Toy._verify."""
        monkeypatch.setattr(FLAIRHUB, 'domain_years', dict(_DOMAIN_YEARS))
        monkeypatch.setattr(FLAIRHUB, 'download_link', str(_TEST_DATA))
        monkeypatch.setattr(
            FLAIRHUBToy, 'download_link', str(_TEST_DATA / 'FLAIR-HUB_TOY_DATASET.zip')
        )

        ds = FLAIRHUB(
            root=tmp_path,
            download=True,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
        )
        assert len(ds) == 1
        x = ds[0]
        assert isinstance(x['mask'], Tensor)

    def test_toy_download_url(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Cover the download_url branch in FLAIRHUBToy._verify."""
        monkeypatch.setattr(
            FLAIRHUBToy, 'download_link', str(_TEST_DATA / 'FLAIR-HUB_TOY_DATASET.zip')
        )
        ds = FLAIRHUBToy(
            root=tmp_path,
            download=True,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
        )
        assert len(ds) == 1
        assert isinstance(ds[0]['mask'], Tensor)

    def test_transforms_branch(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Cover the ``if self.transforms is not None`` branch in __getitem__."""
        root = tmp_path / 'flair'
        shutil.copytree(_TEST_DATA, root)
        monkeypatch.setattr(FLAIRHUB, 'domain_years', dict(_DOMAIN_YEARS))

        ds = FLAIRHUB(
            root=root,
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            transforms=nn.Identity(),
        )
        x = ds[0]
        assert isinstance(x['mask'], Tensor)
