# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import shutil
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pytest
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import Tensor, nn
from torch.utils.data import ConcatDataset

from torchgeo.datasets import FLAIRHUB, DatasetNotFoundError, FLAIRHUBToy
from torchgeo.datasets.flair import AvailableBands

_DOMAIN_YEARS = {'D006': ['2020'], 'D012': ['2019'], 'D032': ['2019']}


class TestFLAIRHUB:
    @pytest.fixture(params=product([FLAIRHUB, FLAIRHUBToy], ['train', 'val', 'test']))
    def dataset(
        self,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        request: SubRequest,
        test_data: Callable[[str], str],
    ) -> FLAIRHUB | FLAIRHUBToy:
        dataset_class: type[FLAIRHUB | FLAIRHUBToy] = request.param[0]
        split = request.param[1]

        if dataset_class is FLAIRHUB:
            monkeypatch.setattr(FLAIRHUB, 'domain_years', dict(_DOMAIN_YEARS))
            monkeypatch.setattr(
                FLAIRHUB, 'download_link', str(Path(test_data('flair')))
            )
        else:
            monkeypatch.setattr(
                FLAIRHUBToy,
                'download_link',
                str(Path(test_data('flair')) / 'FLAIR-HUB_TOY_DATASET.zip'),
            )

        return dataset_class(
            tmp_path,
            split=split,
            transforms=nn.Identity(),
            download=True,
            bands=['AERIAL_RGBI'],
        )

    def test_getitem(self, dataset: FLAIRHUB | FLAIRHUBToy) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image_aerial_rgbi'], Tensor)
        assert isinstance(x['mask'], Tensor)

    def test_len(self, dataset: FLAIRHUB | FLAIRHUBToy) -> None:
        assert len(dataset) == 1

    def test_add(self, dataset: FLAIRHUB | FLAIRHUBToy) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2

    def test_already_downloaded(
        self, dataset: FLAIRHUB | FLAIRHUBToy, tmp_path: Path
    ) -> None:
        type(dataset)(tmp_path, split=dataset.split, bands=dataset.bands)

    @pytest.mark.parametrize('dataset_class', [FLAIRHUB, FLAIRHUBToy])
    def test_not_downloaded(
        self, dataset_class: type[FLAIRHUB] | type[FLAIRHUBToy], tmp_path: Path
    ) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            dataset_class(tmp_path, download=False)

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError, match='Invalid band name: invalidband'):
            FLAIRHUB(bands=['invalidband'])  # ty: ignore[invalid-argument-type]

    def test_all_modalities_plot(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, test_data: Callable[[str], str]
    ) -> None:
        root = tmp_path / 'flair'
        shutil.copytree(Path(test_data('flair')), root)
        monkeypatch.setattr(FLAIRHUB, 'domain_years', {'D006': ['2020']})

        bands: list[AvailableBands] = [
            'AERIAL_RGBI',
            'SPOT_RGBI',
            'SENTINEL2_TS',
            'SENTINEL2_MSK-SC',
            'SENTINEL1-ASC_TS',
            'SENTINEL1-DESC_TS',
            'DEM_ELEV',
            'AERIAL-RLT_PAN',
        ]
        dataset = FLAIRHUB(root=root, bands=bands, dataset_type='land_cover')
        x = dataset[0]

        for band in bands:
            key = dataset.modality_key_map[band]
            assert key in x
            assert isinstance(x[key], Tensor)

        dataset.plot(x, suptitle='All modalities')
        plt.close()

    @pytest.mark.parametrize('dataset_type', ['crop_type_2', 'crop_type_3'])
    def test_crop_type_levels(
        self,
        dataset_type: Literal['crop_type_2', 'crop_type_3'],
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
        test_data: Callable[[str], str],
    ) -> None:
        """crop_type_2 and crop_type_3 mask loading (deeper LPIS levels).
        Basic crop_type already tested by trainer/flairhub_croptype config."""
        root = tmp_path / 'flair'
        shutil.copytree(Path(test_data('flair')), root)
        monkeypatch.setattr(FLAIRHUB, 'domain_years', dict(_DOMAIN_YEARS))

        dataset = FLAIRHUB(root=root, bands=['AERIAL_RGBI'], dataset_type=dataset_type)
        x = dataset[0]
        assert isinstance(x['mask'], Tensor)
        assert isinstance(x['image_aerial_rgbi'], Tensor)

    def test_zip_reload(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, test_data: Callable[[str], str]
    ) -> None:
        """Missing directory is re-extracted from existing zip."""
        root = tmp_path / 'flair'
        shutil.copytree(Path(test_data('flair')), root)
        monkeypatch.setattr(FLAIRHUB, 'domain_years', dict(_DOMAIN_YEARS))

        directory = root / 'D006-2020_AERIAL_RGBI'
        shutil.rmtree(directory)

        FLAIRHUB(root=root, bands=['AERIAL_RGBI'], checksum=False)
        assert directory.is_dir()
        assert not (root / 'D006-2020_AERIAL_RGBI.zip').exists()

    def test_corrupted(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, test_data: Callable[[str], str]
    ) -> None:
        root = tmp_path / 'flair'
        shutil.copytree(Path(test_data('flair')), root)
        monkeypatch.setattr(FLAIRHUB, 'domain_years', {'D006': ['2020']})

        shutil.rmtree(root / 'GLOBAL_ALL_MTD')
        (root / 'GLOBAL_ALL_MTD.zip').write_text('breaking_SHA256')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            FLAIRHUB(root=root, checksum=True, bands=['AERIAL_RGBI'])

    def test_corrupted_modality(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, test_data: Callable[[str], str]
    ) -> None:
        root = tmp_path / 'flair'
        shutil.copytree(Path(test_data('flair')), root)
        monkeypatch.setattr(FLAIRHUB, 'domain_years', {'D006': ['2020']})

        directory = root / 'D006-2020_AERIAL_RGBI'
        shutil.rmtree(directory)
        (root / 'D006-2020_AERIAL_RGBI.zip').write_text('breaking_SHA256')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            FLAIRHUB(root=root, checksum=True, bands=['AERIAL_RGBI'])

    def test_toy_checksum(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, test_data: Callable[[str], str]
    ) -> None:
        monkeypatch.setattr(
            FLAIRHUBToy,
            'download_link',
            str(Path(test_data('flair')) / 'FLAIR-HUB_TOY_DATASET.zip'),
        )
        monkeypatch.setattr(
            FLAIRHUBToy,
            'sha256',
            'f7c19caa216fe37afac70ada47207d7cf1cbd2ce2da24654606bb68c261f3473',
        )
        dataset = FLAIRHUBToy(
            root=tmp_path,
            split='train',
            download=True,
            checksum=True,
            bands=['AERIAL_RGBI'],
        )
        assert len(dataset) == 1

        toy_dir = tmp_path / 'FLAIR-HUB_TOY'
        shutil.rmtree(toy_dir)
        (tmp_path / 'FLAIR-HUB_TOY_DATASET.zip').write_text('breaking_SHA256')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            FLAIRHUBToy(root=tmp_path, checksum=True, bands=['AERIAL_RGBI'])

    def test_toy_reextract_and_missing_splits(
        self, tmp_path: Path, test_data: Callable[[str], str]
    ) -> None:
        shutil.copy(
            Path(test_data('flair')) / 'FLAIR-HUB_TOY_DATASET.zip',
            tmp_path / 'FLAIR-HUB_TOY_DATASET.zip',
        )
        dataset = FLAIRHUBToy(
            root=tmp_path,
            split='train',
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            checksum=False,
        )
        toy_dir = tmp_path / 'FLAIR-HUB_TOY'
        shutil.rmtree(toy_dir)

        dataset = FLAIRHUBToy(
            root=tmp_path,
            split='train',
            bands=['AERIAL_RGBI'],
            dataset_type='land_cover',
            checksum=False,
        )
        assert len(dataset) == 1

        shutil.rmtree(toy_dir / 'GLOBAL_ALL_MTD')
        (toy_dir / 'GLOBAL_ALL_MTD.zip').unlink(missing_ok=True)
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FLAIRHUBToy(
                root=tmp_path,
                split='train',
                bands=['AERIAL_RGBI'],
                dataset_type='land_cover',
            )
