# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, cast

import pytest
import torch
from omegaconf import OmegaConf

from torchgeo.datamodules import ChesapeakeCVPRDataModule


class TestChesapeakeCVPRDataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> ChesapeakeCVPRDataModule:
        conf = OmegaConf.load(os.path.join("tests", "conf", "chesapeake_cvpr_5.yaml"))
        kwargs = OmegaConf.to_object(conf.experiment.datamodule)
        kwargs = cast(Dict[str, Any], kwargs)

        datamodule = ChesapeakeCVPRDataModule(**kwargs)
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule

    def test_nodata_check(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        nodata_check = datamodule.nodata_check(4)
        sample = {"image": torch.ones(1, 2, 2), "mask": torch.ones(2, 2)}
        out = nodata_check(sample)
        assert torch.equal(out["image"], torch.zeros(1, 4, 4))
        assert torch.equal(out["mask"], torch.zeros(4, 4))

    def test_invalid_param_config(self) -> None:
        with pytest.raises(ValueError, match="The pre-generated prior labels"):
            ChesapeakeCVPRDataModule(
                root=os.path.join("tests", "data", "chesapeake", "cvpr"),
                train_splits=["de-test"],
                val_splits=["de-test"],
                test_splits=["de-test"],
                patch_size=32,
                patches_per_tile=2,
                batch_size=2,
                num_workers=0,
                class_set=7,
                use_prior_labels=True,
            )
