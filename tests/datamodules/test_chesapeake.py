# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest

from torchgeo.datamodules import ChesapeakeCVPRDataModule


class TestChesapeakeCVPRDataModule:
    def test_invalid_param_config(self) -> None:
        with pytest.raises(ValueError, match='The pre-generated prior labels'):
            ChesapeakeCVPRDataModule(
                root=os.path.join('tests', 'data', 'chesapeake', 'cvpr'),
                train_splits=['de-test'],
                val_splits=['de-test'],
                test_splits=['de-test'],
                batch_size=2,
                patch_size=32,
                length=4,
                num_workers=0,
                class_set=7,
                use_prior_labels=True,
            )
