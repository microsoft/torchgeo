# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models import panopticon_vitb14


class TestPanopticon:
    @pytest.mark.slow
    def test_panopticon(self) -> None:
        # from https://github.com/Panopticon-FM/panopticon?tab=readme-ov-file#using-panopticon

        model = panopticon_vitb14()

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

        # get patch-level features (for segmentation)
        blk_indices = [3, 5, 7, 11]
        blocks = model.get_intermediate_layers(
            x_dict, n=blk_indices, return_class_token=True
        )
        assert len(blocks) == 4
        cls_tokens = [blk[1] for blk in blocks]
        patch_tokens = [blk[0] for blk in blocks]
        assert tuple(cls_tokens[0].shape) == (2, 768)
        assert tuple(patch_tokens[0].shape) == (2, (224 / 14) ** 2, 768)
