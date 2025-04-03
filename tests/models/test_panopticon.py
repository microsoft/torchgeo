# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models import Panopticon_Weights, panopticon_vitb14


class TestPanopticon:
    @pytest.mark.slow
    def test_panopticon(self) -> None:
        # from https://github.com/Panopticon-FM/panopticon?tab=readme-ov-file#using-panopticon

        model = panopticon_vitb14(Panopticon_Weights.VIT_BASE14)

        # generate example input
        x_dict = dict(
            imgs=torch.randn(2, 3, 224, 224),  # (B, C, H, W)
            chn_ids=torch.tensor([[664, 559, 493]]).repeat(
                2, 1
            ),  # (B, C), RGB wavelengths in nm
        )

        # get image-level features (for classification, regression, ...)
        normed_cls_token = model(x_dict)
        assert tuple(normed_cls_token.shape) == (2, 768)