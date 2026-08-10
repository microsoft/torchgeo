# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
import torch

from tests.data.tessera_cdl.data import ensure_tessera_cdl_data
from torchgeo.datamodules import TesseraCDLDataModule
from torchgeo.datamodules.utils import collate_fn_embeddings

EMBEDDINGS_DIM = 128
CDL_ROOT = str(Path('tests') / 'data' / 'cdl')


class TestCollateFunction:
    """Test the collate function for embeddings."""

    def test_samples(self) -> None:
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
        """Create test datamodule with generated Tessera data in a temp dir."""
        tessera_root = tmp_path / 'tessera_cdl'
        ensure_tessera_cdl_data(tessera_root)

        return TesseraCDLDataModule(
            data_dir=CDL_ROOT,
            tessera_root=str(tessera_root),
            year=2023,
            batch_size=2,
            num_workers=0,
            num_train_patches=2,
            patch_size=16,
            download=False,
        )

    def test_setup(self, datamodule: TesseraCDLDataModule) -> None:
        """Test setup creates the test dataset and sampler."""
        datamodule.setup('test')
        assert datamodule.test_dataset is not None
        assert datamodule.test_sampler is not None

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

    def test_on_after_batch_transfer(self, datamodule: TesseraCDLDataModule) -> None:
        """Test batch transfer returns batch unchanged."""
        batch = {
            'embeddings': torch.randn(10, EMBEDDINGS_DIM),
            'labels': torch.randint(0, 4, (10,)),
        }
        result = datamodule.on_after_batch_transfer(batch, dataloader_idx=0)

        assert torch.equal(result['embeddings'], batch['embeddings'])
        assert torch.equal(result['labels'], batch['labels'])
