# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models import FCN


class TestFCN:
    def test_in_channels(self) -> None:
        model = FCN(in_channels=5, classes=4, num_filters=10)
        x = torch.randn(2, 5, 64, 64)
        model(x)

        model = FCN(in_channels=3, classes=4, num_filters=10)
        match = 'to have 3 channels, but got 5 channels instead'
        with pytest.raises(RuntimeError, match=match):
            model(x)

    def test_classes(self) -> None:
        model = FCN(in_channels=5, classes=4, num_filters=10)
        x = torch.randn(2, 5, 64, 64)
        y = model(x)

        assert y.shape[1] == 4
        assert model.last.out_channels == 4

    def test_model_size(self) -> None:
        model = FCN(in_channels=5, classes=4, num_filters=10)

        assert len(model.backbone) == 10

    def test_model_filters(self) -> None:
        model = FCN(in_channels=5, classes=4, num_filters=10)

        conv_layers = [
            model.backbone[0],
            model.backbone[2],
            model.backbone[4],
            model.backbone[6],
            model.backbone[8],
        ]
        for conv_layer in conv_layers:
            assert conv_layer.out_channels == 10
