# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import GID15DataModule
from torchgeo.datasets import unbind_samples


class TestGID15DataModule:
    @pytest.fixture(scope="class", params=[0.0, 0.5])
    def datamodule(self, request: SubRequest) -> GID15DataModule:
        root = os.path.join("tests", "data", "gid15")
        batch_size = 2
        num_workers = 0
        val_split_size = request.param
        dm = GID15DataModule(
            root=root,
            train_batch_size=batch_size,
            num_workers=num_workers,
            val_split_pct=val_split_size,
            num_tiles_per_batch=1,
            download=True,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_batch_size_warning(self, datamodule: GID15DataModule) -> None:
        match = "The effective batch size will differ"
        with pytest.warns(UserWarning, match=match):
            GID15DataModule(
                root=datamodule.test_dataset.root,
                train_batch_size=3,
                num_tiles_per_batch=2,
                num_workers=datamodule.num_workers,
                val_split_pct=datamodule.val_split_pct,
            )

    def test_train_dataloader(self, datamodule: GID15DataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: GID15DataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: GID15DataModule) -> None:
        next(iter(datamodule.test_dataloader()))

    def test_plot(self, datamodule: GID15DataModule) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        sample = unbind_samples(batch)[0]
        datamodule.plot(sample)
        plt.close()
