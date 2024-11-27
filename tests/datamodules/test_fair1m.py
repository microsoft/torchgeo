# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import matplotlib.pyplot as plt
import pytest

from torchgeo.datamodules import FAIR1MDataModule


class TestFAIR1MDataModule:
    @pytest.fixture
    def datamodule(self) -> FAIR1MDataModule:
        root = os.path.join('tests', 'data', 'fair1m')
        batch_size = 2
        num_workers = 0
        dm = FAIR1MDataModule(root=root, batch_size=batch_size, num_workers=num_workers)
        return dm

    def test_train_dataloader(self, datamodule: FAIR1MDataModule) -> None:
        datamodule.setup('fit')
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: FAIR1MDataModule) -> None:
        datamodule.setup('validate')
        next(iter(datamodule.val_dataloader()))

    def test_predict_dataloader(self, datamodule: FAIR1MDataModule) -> None:
        datamodule.setup('predict')
        next(iter(datamodule.predict_dataloader()))

    def test_plot(self, datamodule: FAIR1MDataModule) -> None:
        datamodule.setup('validate')
        batch = next(iter(datamodule.val_dataloader()))
        sample = {
            'image': batch['image'][0],
            'boxes': batch['boxes'][0],
            'label': batch['label'][0],
        }
        datamodule.plot(sample)
        plt.close()
