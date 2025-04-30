# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest

from torchgeo.datamodules import PatternNetDataModule
from torchgeo.datasets import unbind_samples


class TestPatternNetDataModule:
    @pytest.fixture
    def datamodule(self) -> PatternNetDataModule:
        root = os.path.join("tests", "data", "patternnet")
        batch_size = 1
        num_workers = 0
        dm = PatternNetDataModule(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            val_split_pct=0.2,
            test_split_pct=0.1,
            download=False,
        )
        dm.prepare_data()
        return dm

    def test_train_dataloader(self, datamodule: PatternNetDataModule) -> None:
        datamodule.setup("fit")
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: PatternNetDataModule) -> None:
        datamodule.setup("validate")
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: PatternNetDataModule) -> None:
        datamodule.setup("test")
        next(iter(datamodule.test_dataloader()))

    def test_plot(self, datamodule: PatternNetDataModule) -> None:
        datamodule.setup("validate")
        batch = next(iter(datamodule.val_dataloader()))
        sample = unbind_samples(batch)[0]
        datamodule.plot(sample)
        plt.close()
