# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import shutil
from pathlib import Path

import pytest
import torch

from tests.data.tessera_cdl.data import ensure_tessera_cdl_data
from torchgeo.datamodules import TesseraCDLDataModule
from torchgeo.datamodules.utils import collate_fn_embeddings
from torchgeo.datasets import CDL, GeoDataset

EMBEDDINGS_DIM = 128


class TestCollateFunction:
    """Test the collate function for embeddings."""

    def test_empty_batch(self) -> None:
        """Test collating an empty batch."""
        result = collate_fn_embeddings([])
        assert result['embeddings'].shape == (0, 0)
        assert result['labels'].shape == (0,)

    def test_single_sample(self) -> None:
        """Test collating a single sample."""
        image = torch.randn(EMBEDDINGS_DIM, 4, 4)
        mask = torch.randint(0, 10, (4, 4)).float()
        result = collate_fn_embeddings([{'image': image, 'mask': mask}])

        assert result['embeddings'].shape == (16, EMBEDDINGS_DIM)
        assert result['labels'].shape == (16,)

    def test_multiple_samples(self) -> None:
        """Test collating multiple samples."""
        batch = [
            {
                'image': torch.randn(EMBEDDINGS_DIM, 4, 4),
                'mask': torch.randint(0, 10, (4, 4)).float(),
            }
            for _ in range(3)
        ]
        result = collate_fn_embeddings(batch)
        assert result['embeddings'].shape == (48, EMBEDDINGS_DIM)
        assert result['labels'].shape == (48,)


class TestTesseraCDLDataModule:
    """Test the TesseraCDLDataModule."""

    @pytest.fixture
    def datamodule(self, tmp_path: Path) -> TesseraCDLDataModule:
        """Create test datamodule with generated test data in a temp dir."""
        tessera_root = tmp_path / 'tessera_cdl'
        cdl_root = tmp_path / 'cdl'
        ensure_tessera_cdl_data(tessera_root, cdl_root)

        return TesseraCDLDataModule(
            data_dir=str(cdl_root),
            tessera_root=str(tessera_root),
            year=2023,
            batch_size=2,
            num_workers=0,
            num_train_patches=2,
            patch_size=16,
            download=False,
        )

    def test_initialization(self, datamodule: TesseraCDLDataModule) -> None:
        """Test datamodule parameters are set correctly."""
        assert datamodule.year == 2023
        assert datamodule.batch_size == 2
        assert datamodule.patch_size == 16
        assert datamodule.num_train_patches == 2

    def test_setup(self, datamodule: TesseraCDLDataModule) -> None:
        """Test setup creates train and validation datasets."""
        datamodule.setup('fit')
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None

    @pytest.mark.parametrize('stage', ['fit', 'validate'])
    def test_dataloader(self, datamodule: TesseraCDLDataModule, stage: str) -> None:
        """Test dataloaders return batches with correct format."""
        datamodule.setup(stage)
        loader = (
            datamodule.train_dataloader()
            if stage == 'fit'
            else datamodule.val_dataloader()
        )

        batch = next(iter(loader))

        assert set(batch) == {'embeddings', 'labels'}
        assert batch['embeddings'].ndim == 2
        assert batch['labels'].ndim == 1
        assert batch['embeddings'].shape[0] == batch['labels'].shape[0]
        assert batch['embeddings'].shape[1] == EMBEDDINGS_DIM

    def test_pin_memory(self, datamodule: TesseraCDLDataModule) -> None:
        """Test pin_memory follows cuda availability."""
        datamodule.setup('fit')
        loader = datamodule.train_dataloader()
        assert loader.pin_memory == torch.cuda.is_available()

    def test_on_after_batch_transfer(self, datamodule: TesseraCDLDataModule) -> None:
        """Test batch transfer returns batch unchanged."""
        batch = {
            'embeddings': torch.randn(10, EMBEDDINGS_DIM),
            'labels': torch.randint(0, 4, (10,)),
        }
        result = datamodule.on_after_batch_transfer(batch, dataloader_idx=0)

        assert torch.equal(result['embeddings'], batch['embeddings'])
        assert torch.equal(result['labels'], batch['labels'])

    def test_setup_missing_train_tessera_dir(self, tmp_path: Path) -> None:
        """Test setup raises when the train Tessera directory is missing."""
        cdl_root = tmp_path / 'cdl'
        ensure_tessera_cdl_data(tmp_path / 'tessera_cdl', cdl_root)

        bad_tessera_root = tmp_path / 'missing_tessera'
        datamodule = TesseraCDLDataModule(
            data_dir=str(cdl_root),
            tessera_root=str(bad_tessera_root),
            year=2023,
            batch_size=2,
            num_workers=0,
            num_train_patches=2,
            patch_size=16,
            download=False,
        )

        with pytest.raises(FileNotFoundError, match='Tessera directory not found'):
            datamodule.setup('fit')

    def test_setup_missing_val_tessera_dir(self, tmp_path: Path) -> None:
        """Test setup raises when the validation Tessera directory is missing."""
        tessera_root = tmp_path / 'tessera_cdl'
        cdl_root = tmp_path / 'cdl'
        ensure_tessera_cdl_data(tessera_root, cdl_root)

        val_dir = tessera_root / 'val'
        if val_dir.exists():
            shutil.rmtree(val_dir)

        datamodule = TesseraCDLDataModule(
            data_dir=str(cdl_root),
            tessera_root=str(tessera_root),
            year=2023,
            batch_size=2,
            num_workers=0,
            num_train_patches=2,
            patch_size=16,
            download=False,
        )

        with pytest.raises(FileNotFoundError, match='Tessera directory not found'):
            datamodule.setup('fit')

    def test_explicit_split_dirs(self, tmp_path: Path) -> None:
        """Test that explicit train/val/test dirs override the default layout."""
        tessera_root = tmp_path / 'tessera_cdl'
        cdl_root = tmp_path / 'cdl'
        ensure_tessera_cdl_data(tessera_root, cdl_root)

        train_dir = tessera_root / 'train' / 'global_0.1_degree_representation' / '2023'
        val_dir = tessera_root / 'val' / 'global_0.1_degree_representation' / '2023'

        datamodule = TesseraCDLDataModule(
            data_dir=str(cdl_root),
            tessera_root=str(tmp_path / 'does_not_exist'),
            year=2023,
            batch_size=2,
            num_workers=0,
            num_train_patches=2,
            patch_size=16,
            download=False,
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            test_dir=str(val_dir),
        )

        datamodule.setup('fit')
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None

        datamodule.setup('test')
        assert datamodule.test_dataset is not None
        assert datamodule.test_sampler is not None

    def test_setup_train_dataset_none(
        self, datamodule: TesseraCDLDataModule, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test setup raises if the train dataset could not be built."""
        original_build_dataset = datamodule._build_dataset

        def fake_build_dataset(split: str, cdl: CDL) -> GeoDataset | None:
            if split == 'train':
                return None
            return original_build_dataset(split, cdl)

        monkeypatch.setattr(datamodule, '_build_dataset', fake_build_dataset)

        with pytest.raises(FileNotFoundError, match='Train dataset could not be built'):
            datamodule.setup('fit')
