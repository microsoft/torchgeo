# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pickle
from functools import partial
from typing import cast

import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datamodules import PASTIS100DataModule, PASTISDataModule
from torchgeo.datasets.utils import pad_across_batches


class TestPASTISDataModule:
    @pytest.fixture(params=[PASTISDataModule, PASTIS100DataModule])
    def datamodule(self, request: SubRequest) -> PASTISDataModule | PASTIS100DataModule:
        datamodule_class = cast(
            type[PASTISDataModule | PASTIS100DataModule], request.param
        )
        return datamodule_class(root='tests/data/pastis', padding_length=9)

    def test_collate_fn(
        self, datamodule: PASTISDataModule | PASTIS100DataModule
    ) -> None:
        assert isinstance(datamodule.collate_fn, partial)
        assert datamodule.collate_fn.func is pad_across_batches
        assert datamodule.collate_fn.keywords == {'padding_length': 9}
        pickle.dumps(datamodule.collate_fn)
