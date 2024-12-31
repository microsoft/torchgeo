# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest

from torchgeo.datamodules import FLAIR2DataModule


class TestFLAIR2DataModule:
    @pytest.fixture
    def datamodule(self) -> FLAIR2DataModule:
        root = os.path.join('tests', 'data', 'flair2', 'FLAIR2')
        batch_size = 2
        num_workers = 0
        dm = FLAIR2DataModule(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            use_sentinel=False,
        )  # FIXME: use_sentinel=True does not work due to varying dimensions
        return dm

    def test_train_dataloader(self, datamodule: FLAIR2DataModule) -> None:
        datamodule.setup('fit')
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: FLAIR2DataModule) -> None:
        datamodule.setup('validate')
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: FLAIR2DataModule) -> None:
        datamodule.setup('test')
        next(iter(datamodule.test_dataloader()))

    def test_plot(self, datamodule: FLAIR2DataModule) -> None:
        datamodule.setup('validate')
        batch = next(iter(datamodule.val_dataloader()))
        sample = {'image': batch['image'][0], 'mask': batch['mask'][0]}
        datamodule.plot(sample)
        plt.close()
