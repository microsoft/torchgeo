import os
from itertools import product
from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datamodules import MMFloodDataModule
from torchgeo.datasets import MMFlood


class TestMMFloodDataModule:
    @pytest.fixture(params=product([True, False], ['mean', 'median']))
    def datamodule(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> MMFloodDataModule:
        dataset_root = os.path.join('tests', 'data', 'mmflood/')
        # url = os.path.join(dataset_root)

        # monkeypatch.setattr(MMFlood, 'url', url)
        monkeypatch.setattr(MMFlood, '_nparts', 2)

        include_dem, normalization = request.param
        # root = tmp_path
        return MMFloodDataModule(
            batch_size=2,
            patch_size=8,
            normalization=normalization,
            root=dataset_root,
            include_dem=include_dem,
            transforms=nn.Identity(),
            download=True,
            checksum=True,
        )

    def test_fit_stage(self, datamodule: MMFloodDataModule) -> None:
        datamodule.setup(stage='fit')
        datamodule.setup(stage='fit')
        if datamodule.trainer:
            datamodule.trainer.training = True
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        nchannels = 3 if datamodule.kwargs['include_dem'] else 2
        assert batch['image'].shape == (2, nchannels, 8, 8)
        assert batch['mask'].shape == (2, 8, 8)
        return

    def test_validate_stage(self, datamodule: MMFloodDataModule) -> None:
        datamodule.setup(stage='validate')
        datamodule.setup(stage='validate')
        if datamodule.trainer:
            datamodule.trainer.validating = True
        batch = next(iter(datamodule.val_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        nchannels = 3 if datamodule.kwargs['include_dem'] else 2
        assert batch['image'].shape == (2, nchannels, 8, 8)
        assert batch['mask'].shape == (2, 8, 8)
        return

    def test_test_stage(self, datamodule: MMFloodDataModule) -> None:
        datamodule.setup(stage='test')
        datamodule.setup(stage='test')
        if datamodule.trainer:
            datamodule.trainer.testing = True
        batch = next(iter(datamodule.test_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
        nchannels = 3 if datamodule.kwargs['include_dem'] else 2
        assert batch['image'].shape == (2, nchannels, 8, 8)
        assert batch['mask'].shape == (2, 8, 8)
        return
