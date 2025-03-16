# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest

from torchgeo.datasets import CopernicusBench, CopernicusBenchBase


class TestCopernicusBench:
    @pytest.fixture(params=[('cloud_s2', 'l1_cloud_s2')])
    def dataset(self, request: SubRequest) -> CopernicusBenchBase:
        dataset, directory = request.param
        root = os.path.join('tests', 'data', 'copernicus', directory)
        transforms = nn.Identity()
        return CopernicusBench(dataset, root, transforms=transforms)

    def test_getitem(self, dataset: CopernicusBenchBase) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: CopernicusBenchBase) -> None:
        assert len(dataset) == 1
