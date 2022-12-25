# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import USAVarsDataModule
from torchgeo.datasets import unbind_samples


class TestUSAVarsDataModule:
    @pytest.fixture()
    def datamodule(self, request: SubRequest) -> USAVarsDataModule:
        pytest.importorskip("pandas", minversion="0.23.2")
        root = os.path.join("tests", "data", "usavars")
        batch_size = 1
        num_workers = 0

        dm = USAVarsDataModule(
            root=root, batch_size=batch_size, num_workers=num_workers, download=True
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: USAVarsDataModule) -> None:
        assert len(datamodule.train_dataloader()) == 3
        sample = next(iter(datamodule.train_dataloader()))
        assert sample["image"].shape[0] == datamodule.batch_size

    def test_val_dataloader(self, datamodule: USAVarsDataModule) -> None:
        assert len(datamodule.val_dataloader()) == 2
        sample = next(iter(datamodule.val_dataloader()))
        assert sample["image"].shape[0] == datamodule.batch_size

    def test_test_dataloader(self, datamodule: USAVarsDataModule) -> None:
        assert len(datamodule.test_dataloader()) == 1
        sample = next(iter(datamodule.test_dataloader()))
        assert sample["image"].shape[0] == datamodule.batch_size

    def test_plot(self, datamodule: USAVarsDataModule) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        sample = unbind_samples(batch)[0]
        datamodule.plot(sample)
        plt.close()
