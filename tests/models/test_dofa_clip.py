# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from unittest.mock import patch

import torch

from torchgeo.models import (
    DOFA_CLIP,
    DOFA_CLIP_Base16_Weights,
    dofa_clip_base_patch16_224,
)


class TestDOFA_CLIP:
    def test_dofa_clip(self) -> None:
        model = DOFA_CLIP(
            embed_dim=128,
            image_embed_dim=384,
            image_depth=2,
            image_num_heads=6,
            text_width=128,
            text_layers=2,
            text_heads=8,
        )
        image = torch.rand(2, 4, 224, 224)
        text = torch.randint(0, 100, (3, 77), dtype=torch.long)
        wavelengths = [0.665, 0.56, 0.49, 0.842]
        logits_per_image, logits_per_text = model(image, text, wavelengths)
        assert logits_per_image.shape == torch.Size([2, 3])
        assert logits_per_text.shape == torch.Size([3, 2])

        image_features = model.encode_image(image, wavelengths)
        text_features = model.encode_text(text)
        assert image_features.shape == torch.Size([2, 128])
        assert text_features.shape == torch.Size([3, 128])
        assert torch.allclose(image_features.norm(dim=-1), torch.ones(2), atol=1e-5)
        assert torch.allclose(text_features.norm(dim=-1), torch.ones(3), atol=1e-5)

    def test_dofa_clip_builder(self) -> None:
        model = dofa_clip_base_patch16_224(
            embed_dim=128,
            image_embed_dim=384,
            image_depth=2,
            image_num_heads=6,
            text_width=128,
            text_layers=2,
            text_heads=8,
        )
        image = torch.rand(1, 4, 224, 224)
        text = torch.randint(0, 100, (1, 77), dtype=torch.long)
        wavelengths = [0.665, 0.56, 0.49, 0.842]
        logits_per_image, logits_per_text = model(image, text, wavelengths)
        assert logits_per_image.shape == torch.Size([1, 1])
        assert logits_per_text.shape == torch.Size([1, 1])

    def test_dofa_clip_builder_defaults(self) -> None:
        model = dofa_clip_base_patch16_224()
        assert isinstance(model, DOFA_CLIP)

    def test_dofa_clip_builder_loads_weights(self) -> None:
        model = dofa_clip_base_patch16_224()
        state_dict = model.state_dict()
        prefixed_state_dict = {
            f'module.{key}': value for key, value in state_dict.items()
        }

        with patch.object(
            DOFA_CLIP_Base16_Weights.DOFA_CLIP,
            'get_state_dict',
            return_value=prefixed_state_dict,
        ):
            loaded_model = dofa_clip_base_patch16_224(
                weights=DOFA_CLIP_Base16_Weights.DOFA_CLIP
            )

        assert isinstance(loaded_model, DOFA_CLIP)
