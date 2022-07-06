# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import InriaAerialImageLabelingDataModule

TEST_DATA_DIR = os.path.join("tests", "data", "inria")


class TestInriaAerialImageLabelingDataModule:
    @pytest.fixture(
        params=zip([0.2, 0.2, 0.0], [0.2, 0.0, 0.0], ["test", "test", "test"])
    )
    def datamodule(self, request: SubRequest) -> InriaAerialImageLabelingDataModule:
        val_split_pct, test_split_pct, predict_on = request.param
        patch_size = 2  # (2,2)
        num_patches_per_tile = 2
        root = TEST_DATA_DIR
        batch_size = 1
        num_workers = 0
        dm = InriaAerialImageLabelingDataModule(
            root,
            batch_size,
            num_workers,
            val_split_pct,
            test_split_pct,
            patch_size,
            num_patches_per_tile,
            predict_on=predict_on,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(
        self, datamodule: InriaAerialImageLabelingDataModule
    ) -> None:
        sample = next(iter(datamodule.train_dataloader()))
        assert sample["image"].shape[-2:] == sample["mask"].shape[-2:] == (2, 2)
        assert sample["image"].shape[0] == sample["mask"].shape[0] == 2
        assert sample["image"].shape[1] == 3
        assert sample["mask"].shape[1] == 1

    def test_val_dataloader(
        self, datamodule: InriaAerialImageLabelingDataModule
    ) -> None:
        sample = next(iter(datamodule.val_dataloader()))
        if datamodule.val_split_pct > 0.0:
            assert sample["image"].shape[-2:] == sample["mask"].shape[-2:] == (2, 2)
            assert sample["image"].shape[0] == sample["mask"].shape[0] == 2

    def test_test_dataloader(
        self, datamodule: InriaAerialImageLabelingDataModule
    ) -> None:
        sample = next(iter(datamodule.test_dataloader()))
        if datamodule.test_split_pct > 0.0:
            assert sample["image"].shape[-2:] == sample["mask"].shape[-2:] == (2, 2)
            assert sample["image"].shape[0] == sample["mask"].shape[0] == 2

    def test_predict_dataloader(
        self, datamodule: InriaAerialImageLabelingDataModule
    ) -> None:
        sample = next(iter(datamodule.predict_dataloader()))
        assert len(sample["image"].shape) == 5
        assert sample["image"].shape[-2:] == (2, 2)
        assert sample["image"].shape[2] == 3
